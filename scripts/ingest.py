"""CLI helper to ingest all documents in a folder into the vector store.

Usage:
    python scripts/ingest.py data/sample_docs
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.ingestion import ingest_directory  # noqa: E402


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python scripts/ingest.py <directory>")
        sys.exit(1)

    directory = sys.argv[1]
    names, chunk_count = ingest_directory(directory)

    if not names:
        print(f"No supported files (.pdf, .txt, .md) found in '{directory}'.")
        return

    print(f"Indexed {len(names)} file(s), {chunk_count} chunk(s) total:")
    for name in names:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
