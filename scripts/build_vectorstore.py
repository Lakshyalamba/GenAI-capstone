from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.retrieval import build_vectorstore
from src.utils import KNOWLEDGE_BASE_DIR, VECTORSTORE_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the local Chroma vector store from the markdown knowledge base."
    )
    parser.add_argument(
        "--knowledge-dir",
        default=str(KNOWLEDGE_BASE_DIR),
        help="Path to the markdown knowledge base directory.",
    )
    parser.add_argument(
        "--persist-dir",
        default=str(VECTORSTORE_DIR / "chroma_db"),
        help="Path where the persistent Chroma database should be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_vectorstore(
        kb_dir=Path(args.knowledge_dir),
        persist_directory=Path(args.persist_dir),
    )
    print("Vector store rebuilt successfully.")
    print(f"Backend: {result['backend']}")
    print(f"Collection: {result['collection_name']}")
    print(f"Persist directory: {result['persist_directory']}")
    print(f"Documents indexed: {result['documents_indexed']}")
    print("Sources:")
    for source in result["source_files"]:
        print(f"- {source}")


if __name__ == "__main__":
    main()
