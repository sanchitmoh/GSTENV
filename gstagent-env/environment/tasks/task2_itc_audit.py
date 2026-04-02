"""Task 2: ITC Audit (Medium) — determine ITC eligibility for each invoice."""


def get_task2_config() -> dict:
    return {
        "task_id": "itc_audit",
        "description": (
            "Determine ITC eligibility for each invoice: eligible, partial, or "
            "ineligible. Consider amount mismatches and missing GSTR-2B entries."
        ),
        "max_steps": 12,
        "data_file": "invoices_medium.json",
        "difficulty": "medium",
    }
