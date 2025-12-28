def is_sql_request(question: str) -> bool:
    q = question.lower()
    # simple heuristic: treat these as SQL requests
    sql_words = ["select", "sql", "query", "total", "sum", "count", "group by", "join", "top", "last", "where", "show me", "give me", "list"]
    # treat these as discovery (not necessarily SQL output)
    discovery_words = ["which table", "what table", "where is", "which column", "what column", "find column", "find table"]
    if any(w in q for w in discovery_words) and not any(w in q for w in ["join", "group by", "sum", "count", "total"]):
        return False
    return any(w in q for w in sql_words)

def build_discovery_prompt(question: str, context_lines: list[str]) -> str:
    context_block = "\n".join(f"- {line}" for line in context_lines)
    return f"""
You are helping a data engineer locate the right tables/columns.

Rules:
- Do NOT write SQL.
- Return a short answer listing best tables/columns and why.
- Use ONLY the hints provided.

User question:
{question}

Hints:
{context_block}
""".strip()

def build_sql_prompt(question: str, context_lines: list[str], relationship_lines: list[str]) -> str:
    context_block = "\n".join(f"- {line}" for line in context_lines)
    rel_block = "\n".join(f"- {line}" for line in relationship_lines) if relationship_lines else "(none)"

    return f"""
You are a data engineer. Generate a SQL query for the user's request using ONLY the schema hints and relationship hints provided.

Hard rules:
- Output SQL only (no explanations, no markdown).
- ONLY use tables/columns that appear in Schema Hints or Relationship Hints.
- If the request needs a column not present in hints (example: state) and no join path exists, output exactly: INSUFFICIENT_SCHEMA
- Use ANSI SQL. Use table aliases.
- If grouping is implied (by state/month/etc.), include GROUP BY.

User request:
{question}

Schema Hints (most relevant columns):
{context_block}

Relationship Hints (join paths you may use):
{rel_block}
""".strip()
