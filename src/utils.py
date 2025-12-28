import pandas as pd

def build_doc(row: pd.Series) -> str:
    """
    This is the text that gets embedded.
    It's intentionally redundant to improve retrieval quality.
    """
    domain = str(row.get("domain", "")).strip()
    table_name = str(row.get("table_name", "")).strip()
    column_name = str(row.get("column_name", "")).strip()
    data_type = str(row.get("data_type", "")).strip()
    description = str(row.get("description", "")).strip()
    sample_values = str(row.get("sample_values", "")).strip()

    return (
        f"Domain: {domain}\n"
        f"Table: {table_name}\n"
        f"Column: {column_name}\n"
        f"Type: {data_type}\n"
        f"Description: {description}\n"
        f"Sample Values: {sample_values}\n"
    )

def stable_id(domain: str, table_name: str, column_name: str) -> str:
    return f"{domain}.{table_name}.{column_name}".lower().replace(" ", "_")
