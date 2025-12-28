import os
import pandas as pd
from dotenv import load_dotenv

from src.ollama_client import ollama_embed
from src.vectordb import get_client, reset_collection
from src.utils import build_doc, stable_id

def main():
    load_dotenv()

    embed_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    csv_path = os.path.join("data", "metadata.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing {csv_path}. Create data/metadata.csv first.")

    df = pd.read_csv(csv_path).fillna("")
    client = get_client()
    col = reset_collection(client)

    ids, docs, metadatas, embeddings = [], [], [], []

    for _, row in df.iterrows():
        domain = str(row["domain"]).strip()
        table_name = str(row["table_name"]).strip()
        column_name = str(row["column_name"]).strip()

        doc = build_doc(row)
        emb = ollama_embed(doc, model=embed_model)

        _id = stable_id(domain, table_name, column_name)

        ids.append(_id)
        docs.append(doc)
        embeddings.append(emb)
        metadatas.append({
            "domain": domain,
            "table_name": table_name,
            "column_name": column_name,
            "data_type": str(row.get("data_type", "")).strip(),
            "description": str(row.get("description", "")).strip(),
        })

    col.add(ids=ids, documents=docs, embeddings=embeddings, metadatas=metadatas)
    print(f"Ingested {len(ids)} metadata rows into Chroma collection '{os.getenv('CHROMA_COLLECTION')}'.")

if __name__ == "__main__":
    main()
