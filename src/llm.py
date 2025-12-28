def build_sql_prompt(question: str, context_lines: list[str]) -> str:
    """
    Keeps prompt tight and forces "SQL only" output (no markdown).
    """
    context_block = "\n".join(f"- {line}" for line in context_lines)

    return f"""
You are a data engineer. Generate a SQL query for the user's request using ONLY the schema hints provided.
If a detail is missing, make a reasonable assumption and keep it simple.

Rules:
- Output SQL only (no explanations, no markdown).
- Prefer SELECT queries.
- Use table and column names exactly as given.
- If multiple candidate tables exist, choose the best one.

User request:
{question}

Schema hints (most relevant):
{context_block}
""".strip()
