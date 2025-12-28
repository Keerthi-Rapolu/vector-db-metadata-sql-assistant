import os
import sys
from dotenv import load_dotenv

from src.ollama_client import ollama_embed, ollama_chat
from src.vectordb import get_client, get_collection
from src.llm import build_sql_prompt

def format_hint(meta: dict) -> str:
    return f"{meta.get('domain')}.{meta.get('table_name')}.{meta.get('column_name')} ({meta.get('data_type')}) - {meta.get('description')}"

def main():
    load_dotenv()

    if len(sys.argv) < 2:
        print('Usage: python src/query.py "your question here"')
        sys.exit(1)

    question = sys.argv[1]
    embed_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    chat_model = os.getenv("OLLAMA_CHAT_MODEL", "llama3.1")
    top_k = int(os.getenv("TOP_K", "6"))

    client = get_client()
    col = get_collection(client)

    q_emb = ollama_embed(question, model=embed_model)

    results = col.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["metadatas", "distances"],
    )

    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not metadatas:
        print("No results returned from vector search. Did you run ingest_metadata.py?")
        sys.exit(1)

    print("\nTop matches (vector search):")
    context_lines = []
    for i, (m, d) in enumerate(zip(metadatas, distances), start=1):
        hint = format_hint(m)
        context_lines.append(hint)
        print(f"{i}. {hint}  | distance={d:.4f}")

    prompt = build_sql_prompt(question=question, context_lines=context_lines)
    sql = ollama_chat(prompt, model=chat_model).strip()

    print("\nGenerated SQL:")
    print(sql)
    print()

if __name__ == "__main__":
    main()
