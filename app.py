import io
import json
import logging
import os
import re
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytesseract
from flask import (
    Flask,
    abort,
    flash,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)
from werkzeug.utils import secure_filename


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
RESULTS_FOLDER = BASE_DIR / "results"
DATABASE_PATH = RESULTS_FOLDER / "invoices.sqlite3"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "tif", "tiff", "bmp", "pdf"}
MAX_IMAGE_PIXELS = 8_000_000
EXTRACTION_VERSION = 8

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.environ.get("FLASK_SECRET_KEY") or os.urandom(32),
    UPLOAD_FOLDER=str(UPLOAD_FOLDER),
    MAX_CONTENT_LENGTH=20 * 1024 * 1024,
)

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

configured_tesseract = os.environ.get("TESSERACT_CMD")
if configured_tesseract:
    pytesseract.pytesseract.tesseract_cmd = configured_tesseract

ner_pipeline = None
ner_status = "loading"
ner_model_name = os.environ.get("INVOICE_NER_MODEL", "dslim/bert-base-NER")



def load_ner_model():
    global ner_pipeline, ner_status

    # Skip the memory-intensive NER model when disabled on Render.
    if os.environ.get("DISABLE_NER", "").lower() in {"1", "true", "yes"}:
        ner_status = "unavailable"
        logger.info("NER disabled; using rule-based invoice extraction.")
        return

    try:

        from transformers import pipeline

        loaded_pipeline = pipeline(
            "ner", model=ner_model_name, aggregation_strategy="simple"
        )
    except Exception as exc:
        ner_status = "unavailable"
        logger.warning(
            "NER model %s could not be loaded (%s); invoice extraction will continue without NER.",
            ner_model_name,
            type(exc).__name__,
        )
        logger.debug("NER model initialization failure details", exc_info=True)
    else:
        ner_pipeline = loaded_pipeline
        ner_status = "ready"
        logger.info("NER model loaded: %s", ner_model_name)


threading.Thread(
    target=load_ner_model, name="invoice-ner-loader", daemon=True
).start()


@contextmanager
def get_db():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_database():
    with get_db() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS invoices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL,
                fields_json TEXT NOT NULL,
                entities_json TEXT NOT NULL,
                extracted_text TEXT NOT NULL,
                image_file TEXT NOT NULL,
                warnings_json TEXT NOT NULL,
                field_quality_json TEXT NOT NULL DEFAULT '{}',
                extraction_version INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(invoices)")
        }
        if "field_quality_json" not in columns:
            connection.execute(
                "ALTER TABLE invoices ADD COLUMN field_quality_json "
                "TEXT NOT NULL DEFAULT '{}'"
            )
        if "extraction_version" not in columns:
            connection.execute(
                "ALTER TABLE invoices ADD COLUMN extraction_version "
                "INTEGER NOT NULL DEFAULT 0"
            )


def allowed_file(filename):
    return (
        bool(filename)
        and "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def pdf_to_image(pdf_path):
    try:
        from pdf2image import convert_from_path
    except ImportError as exc:
        raise RuntimeError(
            "PDF support requires pdf2image and Poppler. Install both, then retry."
        ) from exc

    try:
        pages = convert_from_path(
            pdf_path, first_page=1, last_page=1, dpi=180, thread_count=1
        )
    except Exception as exc:
        raise ValueError(
            "The PDF could not be read. Check that it is valid and that Poppler is installed."
        ) from exc
    if not pages:
        raise ValueError("The PDF contains no readable pages.")
    return cv2.cvtColor(np.array(pages[0].convert("RGB")), cv2.COLOR_RGB2BGR)


def read_invoice_image(filepath):
    if filepath.lower().endswith(".pdf"):
        image = pdf_to_image(filepath)
    else:
        image = cv2.imread(filepath, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("The uploaded file could not be read as an image or PDF.")
    height, width = image.shape[:2]
    if height * width > MAX_IMAGE_PIXELS:
        scale = (MAX_IMAGE_PIXELS / (height * width)) ** 0.5
        image = cv2.resize(
            image,
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    return image


def _resize_for_ocr(gray):
    _, width = gray.shape
    if width < 1500:
        scale = min(1500 / width, 3)
        return cv2.resize(
            gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )
    return gray


def preprocess_image(image):
    if image is None or image.size == 0:
        raise ValueError("The uploaded document contains no readable image.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = _resize_for_ocr(gray)
    if np.std(gray) < 12:
        raise ValueError("The uploaded image appears blank or unreadable.")

    return cv2.adaptiveThreshold(
        cv2.GaussianBlur(gray, (3, 3), 0),
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )


def _normalized_lines(text):
    normalized = []
    for raw_line in text.splitlines():
        line = raw_line.replace("：", ":").replace("＃", "#").replace("—", "-")
        line = re.sub(r"[ \t]+", " ", line).strip()
        if line:
            normalized.append(line)
    return normalized


def _value_on_or_after(lines, line_index, suffix):
    if suffix.strip(" \t:#-"):
        return suffix.strip(" \t:#-")
    if line_index + 1 >= len(lines):
        return ""
    next_line = lines[line_index + 1]
    if re.search(
        r"\b(invoice|inv\.?|date|due date|subtotal|sub\s*total|tax|vat|"
        r"grand total|total due|amount due|balance due|bill to|ship to|"
        r"po\s*(?:number|no\.?|#)|purchase order|order number|account|"
        r"bank|routing|reference)\b",
        next_line,
        re.IGNORECASE,
    ):
        return ""
    return next_line


def _valid_calendar_date(value):
    value = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", value, flags=re.IGNORECASE)
    value = re.sub(r"\b([A-Za-z]{3})\.(?=\s)", r"\1", value)
    formats = (
        "%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d",
        "%m/%d/%Y", "%d/%m/%Y", "%m-%d-%Y", "%d-%m-%Y",
        "%m.%d.%Y", "%d.%m.%Y",
        "%m/%d/%y", "%d/%m/%y", "%m-%d-%y", "%d-%m-%y",
        "%b %d, %Y", "%B %d, %Y", "%b %d %Y", "%B %d %Y",
        "%d %b %Y", "%d %B %Y",
    )
    for date_format in formats:
        try:
            return datetime.strptime(value, date_format).date()
        except ValueError:
            continue
    return None


def _find_date(value):
    value = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", value, flags=re.IGNORECASE)
    patterns = (
        r"\b\d{4}[./-]\d{1,2}[./-]\d{1,2}\b",
        r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b",
        r"\b[A-Za-z]{3,9}\.?\s+\d{1,2},?\s+\d{4}\b",
        r"\b\d{1,2}\s+[A-Za-z]{3,9}\.?\s+\d{4}\b",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, value):
            candidate = re.sub(
                r"\b([A-Za-z]{3,9})\.(?=\s)", r"\1", match.group(0)
            )
            candidate = re.sub(
                r"\bSept\b", "Sep", candidate, flags=re.IGNORECASE
            )
            parsed = _valid_calendar_date(candidate)
            if parsed:
                return match.group(0), parsed
    return None, None


def _normalize_amount(value):
    amount = re.sub(r"[$€£\s]", "", value)
    if "," in amount and "." in amount:
        if amount.rfind(",") > amount.rfind("."):
            amount = amount.replace(".", "").replace(",", ".")
        else:
            amount = amount.replace(",", "")
    elif "," in amount:
        tail = amount.rsplit(",", 1)[1]
        amount = amount.replace(",", ".") if len(tail) == 2 else amount.replace(",", "")
    elif "." in amount and len(amount.rsplit(".", 1)[1]) == 3:
        amount = amount.replace(".", "")
    return amount


def _amount_candidates(value):
    pattern = re.compile(
        r"(?<![A-Za-z0-9])(?:[$€£]\s*)?"
        r"-?(?:\d{1,3}(?:[,. ]\d{3})+|\d+)(?:[.,]\d{2})?"
        r"(?![A-Za-z0-9])"
    )
    matches = []
    for match in pattern.finditer(value):
        raw_amount = match.group(0)
        if re.match(r"\s*%", value[match.end():]):
            continue
        priority = (
            bool(re.search(r"[$€£]", raw_amount)),
            bool(re.search(r"[.,]\d{2}$", raw_amount)),
            bool(re.search(r"\d{1,3}[,. ]\d{3}", raw_amount)),
        )
        matches.append((priority, match.start(), _normalize_amount(raw_amount)))
    if not matches:
        return []
    best_priority = max(priority for priority, _position, _amount in matches)
    best_matches = [
        (position, amount)
        for priority, position, amount in matches
        if priority == best_priority
    ]
    return [amount for _position, amount in reversed(best_matches)]


def _field_result(value, status, reason):
    return {"value": value, "status": status, "reason": reason}


def _extract_invoice_data(text):
    lines = _normalized_lines(text)
    matches = {}

    number_label = re.compile(
        r"\b(?:invoice|inv\.?)\s*(?:number|no\.?|num(?:ber)?\.?|id|#)(?!\w)",
        re.IGNORECASE,
    )
    invoice_number = None
    number_evidence = None
    for index, line in enumerate(lines):
        label = number_label.search(line)
        if not label:
            continue
        suffix = _value_on_or_after(lines, index, line[label.end():])
        value_match = re.match(
            r"[#:\s-]*([A-Z0-9][A-Z0-9./_-]*(?:\s+[A-Z0-9][A-Z0-9./_-]*)?)",
            suffix,
            re.IGNORECASE,
        )
        if not value_match:
            continue
        candidate = value_match.group(1).strip()
        candidate = re.sub(r"\s+", "-", candidate)
        if not re.search(r"\d", candidate):
            continue
        if re.match(r"^(?:PO|P\.O\.|ACCOUNT|ACCT|ORDER|BANK)", candidate, re.IGNORECASE):
            continue
        if _find_date(candidate)[0]:
            continue
        if candidate.casefold() in {"business", "invoice", "number", "unknown", "none"}:
            continue
        invoice_number = candidate
        number_evidence = "A numeric identifier follows an invoice-number label."
        break
    matches["Invoice Number"] = _field_result(
        invoice_number,
        "matched" if invoice_number else "not_found",
        number_evidence or "No numeric identifier was found after an invoice-number label.",
    )

    date_label = re.compile(
        r"\b(?:invoice\s+date|date\s+of\s+invoice|date\s+of\s+issue|"
        r"issue\s+date|issued\s+date|issued\s+on|date)\b",
        re.IGNORECASE,
    )
    invoice_date = None
    date_reason = "No valid calendar date was found beside an invoice date label."
    for index, line in enumerate(lines):
        if re.search(
            r"\b(?:due|delivery|ship(?:ment)?|purchase\s+order|po|order|"
            r"payment)\s+date\b",
            line,
            re.IGNORECASE,
        ):
            continue
        label = date_label.search(line)
        if not label:
            continue
        value = _value_on_or_after(lines, index, line[label.end():])
        date_match, _parsed = _find_date(value)
        if date_match:
            invoice_date = date_match
            date_reason = "A valid calendar date follows a date label."
            break
    matches["Invoice Date"] = _field_result(
        invoice_date,
        "matched" if invoice_date else "not_found",
        date_reason,
    )

    supplier = None
    supplier_reason = "No supplier name could be distinguished from labels or placeholders."
    supplier_label = re.compile(
        r"\b(?:(?:supplier|vendor|seller|company)(?:\s+name)?|from)"
        r"\s*[:#-]\s*(.+)$",
        re.IGNORECASE,
    )
    placeholder = re.compile(
        r"\[.*?\]|\{.*?\}|<.*?>|your\s+(?:name|business|logo)|"
        r"name\s*/\s*business\s+name|business\s+address|account\s+holder|"
        r"swift\s*/?\s*bic|iban|sort\s+code|\blogo\b|"
        r"company\s+name|supplier\s+name|vendor\s+name|"
        r"standard\s+business\s+invoice|sample\s+invoice|"
        r"\bbilled?\s+to\b|\bship(?:ped)?\s+to\b|\bremit\s+to\b|"
        r"^\s*(?:business|company|supplier|vendor)\s*$",
        re.IGNORECASE,
    )
    for line in lines[:12]:
        labeled_supplier = supplier_label.search(line)
        if labeled_supplier:
            candidate = labeled_supplier.group(1).strip()
            if candidate and not placeholder.search(candidate):
                supplier = candidate[:120]
                supplier_reason = "Supplier name follows an explicit supplier/vendor label."
                break
    if not supplier:
        for line in lines[:8]:
            if (
                len(line) >= 3
                and not line.endswith(":")
                and not placeholder.search(line)
                and not re.search(
                    r"\b(invoice|date|bill\s+to|ship\s+to|subtotal|tax|total|"
                    r"amount|phone|email|www\.|address|supplier|vendor|"
                    r"po\s*(?:number|no\.?|#)|purchase\s+order|order\s+number|"
                    r"account\s+(?:number|holder)|reference|bank|routing|"
                    r"swift|bic|iban|sort\s+code|"
                    r"payment\s+terms|customer\s+(?:name|address))\b",
                    line,
                    re.IGNORECASE,
                )
                and not re.search(r"\b(?:no\.?|number|#)\s*[:#]?\s*$", line, re.IGNORECASE)
                and not re.search(r"\d{3,}", line)
            ):
                supplier = line[:120]
                supplier_reason = "Inferred from a non-placeholder heading near the top of the document; verify it."
                break
    matches["Supplier"] = _field_result(
        supplier,
        "matched" if supplier and "explicit" in supplier_reason else (
            "inferred" if supplier else "not_found"
        ),
        supplier_reason,
    )

    def find_amount(label_patterns, missing_reason, reject_patterns=()):
        for label_pattern in label_patterns:
            label = re.compile(
                r"(?<!\w)(?:" + label_pattern + r")(?!\w)",
                re.IGNORECASE,
            )
            for index, line in enumerate(lines):
                if any(
                    re.search(pattern, line, re.IGNORECASE)
                    for pattern in reject_patterns
                ):
                    continue
                match = label.search(line)
                if not match:
                    continue
                suffix = line[match.end():]
                next_label = re.search(
                    r"\b(?:sub\s*total|sales\s+tax|tax|vat|amount\s+due|"
                    r"balance\s+due|total\s+due|grand\s+total|invoice\s+total|"
                    r"total\s+amount|invoice\s+date|due\s+date)\b",
                    suffix,
                    re.IGNORECASE,
                )
                if next_label:
                    suffix = suffix[:next_label.start()]
                candidates = _amount_candidates(suffix)
                if not candidates and index + 1 < len(lines):
                    next_line = lines[index + 1]
                    if not re.search(
                        r"\b(subtotal|tax|vat|grand total|total due|amount due|"
                        r"balance due|invoice total|invoice|date|po\s*(?:number|no\.?|#)|"
                        r"purchase order|order number|account|bank|routing|reference)\b",
                        next_line,
                        re.IGNORECASE,
                    ):
                        candidates = _amount_candidates(next_line)
                if candidates:
                    return candidates[0], "A monetary value follows an explicit label."
        return None, missing_reason

    subtotal, subtotal_reason = find_amount(
        (r"sub\s*total", r"subtotal"),
        "No amount was found beside a subtotal label.",
    )
    tax, tax_reason = find_amount(
        (r"sales\s+tax", r"tax", r"vat"),
        "No amount was found beside a tax/VAT label.",
        reject_patterns=(r"\btax\s+(?:id|number|no\.?)\b",),
    )
    total, total_reason = find_amount(
        (
            r"amount\s+due",
            r"balance\s+due",
            r"total\s+due",
            r"grand\s+total",
            r"invoice\s+total",
            r"total\s+amount",
            r"total",
        ),
        "No payable amount was found beside a total/due label.",
        reject_patterns=(
            r"\b(?:sub\s*total|subtotal|tax\s+total|total\s+tax|"
            r"total\s+(?:before\s+tax|items|quantity|savings|discount)|"
            r"(?:account|bank|purchase\s+order)\s+total|"
            r"(?:account|bank)\s+balance)\b",
        ),
    )
    matches["Subtotal"] = _field_result(
        subtotal, "matched" if subtotal else "not_found", subtotal_reason
    )
    matches["Tax"] = _field_result(
        tax, "matched" if tax else "not_found", tax_reason
    )
    matches["Total Amount"] = _field_result(
        total, "matched" if total else "not_found", total_reason
    )

    fields = {key: result["value"] for key, result in matches.items()}
    quality = {
        key: {"status": result["status"], "reason": result["reason"]}
        for key, result in matches.items()
    }
    return fields, quality


def extract_invoice_fields(text):
    fields, _quality = _extract_invoice_data(text)
    return fields


def upgrade_saved_extractions():
    with get_db() as connection:
        rows = connection.execute(
            "SELECT id, fields_json, warnings_json, extracted_text "
            "FROM invoices WHERE extraction_version < ?",
            (EXTRACTION_VERSION,),
        ).fetchall()
        for row in rows:
            fields, quality = _extract_invoice_data(row["extracted_text"])
            try:
                old_warnings = json.loads(row["warnings_json"])
            except (TypeError, json.JSONDecodeError):
                old_warnings = []
            warnings = [
                f"{name}: {item['reason']}"
                for name, item in quality.items()
                if item["status"] in {"not_found", "inferred"}
            ]
            warnings.extend(
                warning
                for warning in old_warnings
                if warning.startswith(("The NER model", "NER prediction"))
            )
            connection.execute(
                """
                UPDATE invoices
                SET fields_json = ?, field_quality_json = ?, warnings_json = ?,
                    status = ?, extraction_version = ?
                WHERE id = ?
                """,
                (
                    json.dumps(fields),
                    json.dumps(quality),
                    json.dumps(warnings),
                    "review" if warnings else "success",
                    EXTRACTION_VERSION,
                    row["id"],
                ),
            )
            logger.info(
                "Updated saved invoice extraction id=%s; fields requiring review=%d",
                row["id"],
                sum(item["status"] != "matched" for item in quality.values()),
            )


initialize_database()
upgrade_saved_extractions()


def get_ner_entities(text):
    if ner_pipeline is None:
        if ner_status == "loading":
            return [], "The NER model is still loading; rule-based invoice fields are shown."
        return [], "The NER model is unavailable; rule-based invoice fields are still shown."
    try:
        result = ner_pipeline(text[:20_000])
        entities = [
            {
                "word": str(entity.get("word", "")),
                "label": str(entity.get("entity_group", entity.get("entity", "Unknown"))),
                "score": round(float(entity["score"]), 4)
                if entity.get("score") is not None
                else None,
            }
            for entity in result
        ]
    except Exception:
        logger.exception("NER prediction failed")
        return [], "NER prediction failed; rule-based invoice fields are still shown."
    return entities, None


def process_invoice(filepath, image_output_path):
    image = read_invoice_image(filepath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if np.std(gray) < 12:
        raise ValueError("The uploaded image appears blank or unreadable.")
    original_candidate = _resize_for_ocr(gray)
    candidates = (
    ("grayscale, page segmentation 6", original_candidate, "--oem 3 --psm 6"),
)
    best_candidate = None
    try:
        for description, candidate_image, config in candidates:
            data = pytesseract.image_to_data(
                candidate_image,
                config=config,
                output_type=pytesseract.Output.DICT,
            )
            confidences = [
                float(confidence)
                for confidence, word in zip(data["conf"], data["text"])
                if word.strip() and float(confidence) >= 0
            ]
            word_count = sum(bool(word.strip()) for word in data["text"])
            mean_confidence = (
                sum(confidences) / len(confidences) if confidences else 0.0
            )
            score = mean_confidence + min(word_count, 100) * 0.02
            logger.debug(
                "OCR candidate evaluated method=%s words=%d mean_confidence=%.1f",
                description,
                word_count,
                mean_confidence,
            )
            if best_candidate is None or score > best_candidate[0]:
                best_candidate = (
                    score, description, candidate_image, data, mean_confidence
                )
    except pytesseract.TesseractNotFoundError as exc:
        raise RuntimeError(
            "Tesseract OCR is not installed or could not be found. Install it and retry."
        ) from exc
    except pytesseract.TesseractError as exc:
        logger.exception("Tesseract failed to process the invoice")
        raise RuntimeError(
            "OCR could not process this document. Please retry with a clearer scan."
        ) from exc

    _, selected_method, ocr_image, ocr_data, mean_confidence = best_candidate
    logger.info(
        "Selected OCR method=%s mean_confidence=%.1f",
        selected_method,
        mean_confidence,
    )

    ocr_lines = {}
    for index, word in enumerate(ocr_data["text"]):
        if isinstance(word, str) and word.strip():
            line_key = (
                ocr_data["page_num"][index],
                ocr_data["block_num"][index],
                ocr_data["par_num"][index],
                ocr_data["line_num"][index],
            )
            ocr_lines.setdefault(line_key, []).append(word.strip())
    extracted_text = "\n".join(
        " ".join(words) for words in ocr_lines.values()
    )
    if not extracted_text:
        raise ValueError("No text could be extracted. Try a clearer or higher-resolution scan.")
    if logger.isEnabledFor(logging.DEBUG) and os.environ.get("LOG_OCR_TEXT") == "true":
        logger.debug("OCR text (sensitive invoice contents):\n%s", extracted_text)

    preview = image.copy()
    scale_x = image.shape[1] / ocr_image.shape[1]
    scale_y = image.shape[0] / ocr_image.shape[0]
    for index, word in enumerate(ocr_data["text"]):
        if not isinstance(word, str) or not word.strip():
            continue
        confidence = float(ocr_data["conf"][index])
        if confidence < 0:
            continue
        x = round(int(ocr_data["left"][index]) * scale_x)
        y = round(int(ocr_data["top"][index]) * scale_y)
        width = round(int(ocr_data["width"][index]) * scale_x)
        height = round(int(ocr_data["height"][index]) * scale_y)
        cv2.rectangle(
            preview, (x, y), (x + width, y + height), (79, 70, 229), 2
        )
    if not cv2.imwrite(str(image_output_path), preview):
        raise RuntimeError("The highlighted invoice preview could not be saved.")

    fields, field_quality = _extract_invoice_data(extracted_text)
    entities, ner_warning = get_ner_entities(extracted_text)
    warnings = [
        f"{field}: {result['reason']}"
        for field, result in field_quality.items()
        if result["status"] in {"not_found", "inferred"}
    ]
    if ner_warning:
        warnings.append(ner_warning)
    status = "review" if warnings else "success"
    return fields, field_quality, entities, extracted_text, warnings, status


def invoice_from_row(row):
    invoice = dict(row)
    invoice["fields"] = json.loads(invoice.pop("fields_json"))
    invoice["entities"] = json.loads(invoice.pop("entities_json"))
    invoice["warnings"] = json.loads(invoice.pop("warnings_json"))
    quality_json = invoice.pop("field_quality_json", "{}")
    invoice["field_quality"] = json.loads(quality_json or "{}")
    invoice.pop("extraction_version", None)
    return invoice


def fetch_invoice(invoice_id):
    with get_db() as connection:
        row = connection.execute(
            "SELECT * FROM invoices WHERE id = ?", (invoice_id,)
        ).fetchone()
    return invoice_from_row(row) if row else None


def get_recent_invoices(limit=5):
    with get_db() as connection:
        rows = connection.execute(
            "SELECT * FROM invoices ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [invoice_from_row(row) for row in rows]


def dashboard_metrics():
    with get_db() as connection:
        row = connection.execute(
            """
            SELECT COUNT(*) AS total,
                   SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) AS successful,
                   SUM(CASE WHEN status = 'review' THEN 1 ELSE 0 END) AS review
            FROM invoices
            """
        ).fetchone()
    return {
        "total": row["total"] or 0,
        "successful": row["successful"] or 0,
        "review": row["review"] or 0,
    }


@app.route("/")
def home():
    return render_template(
        "index.html",
        metrics=dashboard_metrics(),
        history=get_recent_invoices(),
        page_title="Dashboard",
    )


@app.route("/upload", methods=["POST"])
def upload_file():
    uploaded_file = request.files.get("invoice")
    if uploaded_file is None or not uploaded_file.filename:
        flash("Choose an invoice file before submitting.", "error")
        return redirect(url_for("home"))
    if not allowed_file(uploaded_file.filename):
        flash("Unsupported format. Upload a PDF or a PNG, JPG, TIFF, or BMP image.", "error")
        return redirect(url_for("home"))

    original_name = secure_filename(uploaded_file.filename)
    if not original_name:
        flash("The selected file has an invalid filename.", "error")
        return redirect(url_for("home"))

    extension = original_name.rsplit(".", 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{extension}"
    upload_folder = Path(app.config["UPLOAD_FOLDER"])
    source_path = upload_folder / unique_name
    preview_name = f"{uuid.uuid4().hex}-highlighted.png"
    preview_path = upload_folder / preview_name
    try:
        uploaded_file.save(source_path)
    except OSError:
        logger.exception("Could not save uploaded invoice")
        flash("The uploaded file could not be saved. Please retry.", "error")
        return redirect(url_for("home"))
    if source_path.stat().st_size == 0:
        source_path.unlink(missing_ok=True)
        flash("The selected file is empty.", "error")
        return redirect(url_for("home"))

    try:
        fields, field_quality, entities, extracted_text, warnings, status = process_invoice(
            str(source_path), preview_path
        )
    except (ValueError, RuntimeError) as exc:
        logger.info("Invoice processing rejected: %s", exc)
        source_path.unlink(missing_ok=True)
        preview_path.unlink(missing_ok=True)
        flash(str(exc), "error")
        return redirect(url_for("home"))
    except Exception:
        logger.exception("Unexpected invoice processing failure")
        source_path.unlink(missing_ok=True)
        preview_path.unlink(missing_ok=True)
        flash("Invoice processing failed unexpectedly. Please check the file and try again.", "error")
        return redirect(url_for("home"))
    finally:
        source_path.unlink(missing_ok=True)

    try:
        with get_db() as connection:
            cursor = connection.execute(
                """
                INSERT INTO invoices
                    (filename, created_at, status, fields_json, entities_json,
                     extracted_text, image_file, warnings_json,
                     field_quality_json, extraction_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    original_name,
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    status,
                    json.dumps(fields),
                    json.dumps(entities),
                    extracted_text,
                    preview_name,
                    json.dumps(warnings),
                    json.dumps(field_quality),
                    EXTRACTION_VERSION,
                ),
            )
            invoice_id = cursor.lastrowid
    except sqlite3.Error:
        preview_path.unlink(missing_ok=True)
        logger.exception("Could not save invoice history")
        flash("The invoice was processed but its history could not be saved. Please retry.", "error")
        return redirect(url_for("home"))
    return redirect(url_for("invoice_detail", invoice_id=invoice_id))


@app.route("/invoices/<int:invoice_id>")
def invoice_detail(invoice_id):
    invoice = fetch_invoice(invoice_id)
    if invoice is None:
        abort(404)
    return render_template(
        "entities.html", invoice=invoice, page_title="Invoice details"
    )


@app.route("/history")
def history():
    search = request.args.get("q", "").strip()
    status = request.args.get("status", "").strip()
    date_from = request.args.get("date_from", "").strip()
    date_to = request.args.get("date_to", "").strip()
    sort = request.args.get("sort", "created_at")
    direction = request.args.get("direction", "desc").lower()
    page = max(request.args.get("page", 1, type=int) or 1, 1)
    per_page = 20
    where = []
    parameters = []
    if search:
        where.append(
            "(filename LIKE ? OR fields_json LIKE ?)"
        )
        parameters.extend([f"%{search}%", f"%{search}%"])
    if status in {"success", "review"}:
        where.append("status = ?")
        parameters.append(status)
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_from):
        where.append("date(created_at) >= date(?)")
        parameters.append(date_from)
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_to):
        where.append("date(created_at) <= date(?)")
        parameters.append(date_to)
    clause = f"WHERE {' AND '.join(where)}" if where else ""
    sort_columns = {
        "created_at": "id",
        "filename": "filename COLLATE NOCASE",
        "invoice_number": "json_extract(fields_json, '$.\"Invoice Number\"') COLLATE NOCASE",
        "supplier": "json_extract(fields_json, '$.\"Supplier\"') COLLATE NOCASE",
        "invoice_date": "json_extract(fields_json, '$.\"Invoice Date\"') COLLATE NOCASE",
        "total_amount": "CAST(json_extract(fields_json, '$.\"Total Amount\"') AS REAL)",
    }
    if sort not in sort_columns:
        sort = "created_at"
    if direction not in {"asc", "desc"}:
        direction = "desc"
    with get_db() as connection:
        total = connection.execute(
            f"SELECT COUNT(*) FROM invoices {clause}", parameters
        ).fetchone()[0]
        rows = connection.execute(
            f"SELECT * FROM invoices {clause} "
            f"ORDER BY {sort_columns[sort]} {direction.upper()}, id DESC "
            "LIMIT ? OFFSET ?",
            [*parameters, per_page, (page - 1) * per_page],
        ).fetchall()
    return render_template(
        "history.html",
        history=[invoice_from_row(row) for row in rows],
        search=search,
        status=status,
        date_from=date_from,
        date_to=date_to,
        sort=sort,
        direction=direction,
        page=page,
        pages=max((total + per_page - 1) // per_page, 1),
        total=total,
        page_title="Invoice history",
    )


@app.route("/analytics")
def analytics():
    return render_template(
        "analytics.html",
        metrics=dashboard_metrics(),
        latest=get_recent_invoices(1),
        page_title="Reports & exports",
    )


@app.route("/about")
def about():
    return render_template("about.html", page_title="About InvoiceAI")


@app.route("/export/<int:invoice_id>.json")
def export_json(invoice_id):
    invoice = fetch_invoice(invoice_id)
    if invoice is None:
        abort(404)
    payload = {
        "id": invoice["id"],
        "filename": invoice["filename"],
        "processed_at": invoice["created_at"],
        "status": invoice["status"],
        "fields": invoice["fields"],
        "field_quality": invoice["field_quality"],
        "entities": invoice["entities"],
        "extracted_text": invoice["extracted_text"],
        "warnings": invoice["warnings"],
    }
    response = app.response_class(
        json.dumps(payload, indent=2, ensure_ascii=False),
        mimetype="application/json",
    )
    response.headers["Content-Disposition"] = (
        f'attachment; filename="invoice-{invoice_id}.json"'
    )
    return response


@app.route("/export.xlsx")
def export_excel():
    invoice_id = request.args.get("id", type=int)
    with get_db() as connection:
        if invoice_id is None:
            rows = connection.execute(
                "SELECT * FROM invoices ORDER BY id DESC"
            ).fetchall()
        else:
            rows = connection.execute(
                "SELECT * FROM invoices WHERE id = ?", (invoice_id,)
            ).fetchall()
    if invoice_id is not None and not rows:
        abort(404)
    if not rows:
        flash("There are no processed invoices to export yet.", "error")
        return redirect(url_for("history"))
    try:
        export_rows = []
        for row in rows:
            invoice = invoice_from_row(row)
            export_rows.append(
                {
                    "File name": invoice["filename"],
                    "Processed at": invoice["created_at"],
                    "Status": invoice["status"],
                    **invoice["fields"],
                }
            )
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            pd.DataFrame(export_rows).to_excel(
                writer, index=False, sheet_name="Invoices"
            )
            quality_rows = []
            for row in rows:
                invoice = invoice_from_row(row)
                for field, quality in invoice["field_quality"].items():
                    quality_rows.append(
                        {
                            "File name": invoice["filename"],
                            "Field": field,
                            "Status": quality["status"],
                            "Reason": quality["reason"],
                        }
                    )
            pd.DataFrame(quality_rows).to_excel(
                writer, index=False, sheet_name="Field quality"
            )
        output.seek(0)
    except Exception:
        logger.exception("Excel export failed")
        flash("Excel export failed. Please try again.", "error")
        return redirect(url_for("history"))
    filename = f"invoice-{invoice_id}.xlsx" if invoice_id else "invoices.xlsx"
    return send_file(
        output,
        as_attachment=True,
        download_name=filename,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@app.errorhandler(413)
def request_entity_too_large(_error):
    limit_bytes = app.config["MAX_CONTENT_LENGTH"]
    limit_label = (
        f"{limit_bytes // (1024 * 1024)} MB"
        if limit_bytes % (1024 * 1024) == 0
        else f"{limit_bytes} bytes"
    )
    return render_template(
        "error.html",
        message=f"This file exceeds the {limit_label} upload limit.",
        page_title="Upload too large",
    ), 413


@app.errorhandler(404)
def not_found(_error):
    return render_template(
        "error.html",
        message="The requested invoice or page could not be found.",
        page_title="Not found",
    ), 404


if __name__ == "__main__":
    app.run(
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "5000")),
        debug=os.environ.get("FLASK_DEBUG", "").lower() == "true",
    )
