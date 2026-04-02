"""Task 3: Full Reconciliation (Hard) — complete GST reconciliation with report."""


def get_task3_config() -> dict:
    return {
        "task_id": "full_recon",
        "description": (
            "Given the full month purchase register and GSTR-2B, produce a complete "
            "reconciliation report with: total eligible ITC, discrepancy list with "
            "supplier GSTINs, and recommended actions for each discrepancy."
        ),
        "max_steps": 20,
        "data_file": "invoices_hard.json",
        "difficulty": "hard",
    }
