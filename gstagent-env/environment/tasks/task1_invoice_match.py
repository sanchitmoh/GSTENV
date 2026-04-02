"""Task 1: Invoice Matching (Easy) — identify which invoices appear in GSTR-2B."""


def get_task1_config() -> dict:
    return {
        "task_id": "invoice_match",
        "description": (
            "Identify which invoices from your purchase register appear in GSTR-2B "
            "and which are missing. Classify each invoice as 'present' or 'missing'."
        ),
        "max_steps": 8,
        "data_file": "invoices_easy.json",
        "difficulty": "easy",
    }
