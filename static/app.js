(() => {
  const form = document.getElementById("upload-form");
  if (!form) return;

  const input = document.getElementById("invoice");
  const dropZone = document.getElementById("drop-zone");
  const title = document.getElementById("file-title");
  const description = document.getElementById("file-description");
  const hint = document.getElementById("upload-hint");
  const submit = document.getElementById("submit-upload");
  const remove = document.getElementById("remove-file");
  const maxBytes = 20 * 1024 * 1024;
  const extensions = new Set(["pdf", "png", "jpg", "jpeg", "tif", "tiff", "bmp"]);

  const setFile = (file) => {
    if (!file) {
      input.value = "";
      title.textContent = "";
      title.append("Drop an invoice here, or ");
      const browse = document.createElement("span");
      browse.textContent = "browse files";
      title.append(browse);
      description.textContent = "PDF, PNG, JPG, TIFF, or BMP · Maximum 20 MB";
      hint.textContent = "Your files are processed locally by this application.";
      submit.disabled = true;
      remove.hidden = true;
      return;
    }
    const extension = file.name.split(".").pop().toLowerCase();
    if (!extensions.has(extension)) {
      setFile(null);
      hint.textContent = "Unsupported format. Choose a PDF or supported image file.";
      return;
    }
    if (file.size === 0 || file.size > maxBytes) {
      setFile(null);
      hint.textContent = file.size === 0
        ? "This file is empty. Choose a different invoice."
        : "This file exceeds the 20 MB limit. Choose a smaller file.";
      return;
    }
    title.textContent = file.name;
    description.textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB · Ready to process`;
    hint.textContent = "Select Process invoice to begin OCR and extraction.";
    submit.disabled = false;
    remove.hidden = false;
  };

  input.addEventListener("change", () => setFile(input.files[0]));
  remove.addEventListener("click", () => setFile(null));
  dropZone.addEventListener("dragover", (event) => {
    event.preventDefault();
    dropZone.classList.add("dragging");
  });
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragging"));
  dropZone.addEventListener("drop", (event) => {
    event.preventDefault();
    dropZone.classList.remove("dragging");
    const file = event.dataTransfer.files[0];
    if (!file) return;
    const transfer = new DataTransfer();
    transfer.items.add(file);
    input.files = transfer.files;
    setFile(file);
  });
  form.addEventListener("submit", (event) => {
    if (!input.files.length || submit.disabled) {
      event.preventDefault();
      hint.textContent = "Choose a valid invoice file before processing.";
      return;
    }
    submit.disabled = true;
    submit.textContent = "Processing invoice...";
    hint.textContent = "Processing has started. The page will update when the results are ready.";
  });
})();
