labels = [
    "O",
    "B-INVOICE_NO",
    "I-INVOICE_NO",
    "B-DATE",
    "I-DATE",
    "B-TOTAL",
    "I-TOTAL",
    "B-COMPANY",
    "I-COMPANY"
]

label2id = {label: i for i, label in enumerate(labels)}

id2label = {i: label for i, label in enumerate(labels)}

print(label2id)