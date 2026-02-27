#!/usr/bin/env python3
"""Script to reset the database and Qdrant vector store.

Usage:
  python scripts/reset_db.py [--force]
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to sys.path so we can import backend packages
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

from backend.core.database import Message, _engine, init_db
from backend.core.qdrant_manager import COLLECTIONS, QdrantManager


def reset_sqlite(force: bool):
    """Drop and recreate SQLite tables."""
    print("🧊 Resetting SQLite database...")
    if not force:
        confirm = input("  This will delete all chat history. Continue? [y/N]: ")
        if confirm.lower() != 'y':
            print("  Skipping SQLite reset.")
            return

    db_path = os.environ.get("DATABASE_URL", "sqlite:///backend/data/icberg.sqlite")
    # Initialize engine
    init_db(db_path)
    
    # Drop all tables and recreate
    from backend.core.database import Base
    if _engine:
        Base.metadata.drop_all(_engine)
        Base.metadata.create_all(_engine)
        print("  ✅ SQLite tables dropped and recreated.")
    else:
        print("  ❌ Failed to initialize SQLite engine.")


def reset_qdrant(force: bool):
    """Drop and recreate Qdrant collections."""
    print("🧊 Resetting Qdrant vector store...")
    qdrant = QdrantManager()
    
    if not qdrant.is_healthy():
        print("  ❌ Could not connect to Qdrant. Check URL/API Key.")
        return

    if not force:
        confirm = input(f"  This will delete all collections ({', '.join(COLLECTIONS)}). Continue? [y/N]: ")
        if confirm.lower() != 'y':
            print("  Skipping Qdrant reset.")
            return

    client = qdrant._client
    for col in COLLECTIONS:
        if client.collection_exists(col):
            client.delete_collection(col)
            print(f"  🗑️ Deleted collection: {col}")
        else:
            print(f"  ℹ️ Collection {col} did not exist.")
            
    # _ensure_collections creates missing ones
    qdrant._ensure_collections()
    print("  ✅ Qdrant collections recreated.")


def main():
    parser = argparse.ArgumentParser(description="Reset IcBerg databases.")
    parser.add_argument("--force", "-f", action="store_true", help="Skip confirmation prompts")
    parser.add_argument("--only", choices=["sqlite", "qdrant"], help="Only reset specific datastore")
    args = parser.parse_args()

    # Load environment variables
    load_dotenv(project_root / ".env")

    print("\n⚠️ WARNING: Database Reset ⚠️\n")
    
    if args.only in ["sqlite", None]:
        reset_sqlite(args.force)
        print("")
        
    if args.only in ["qdrant", None]:
        reset_qdrant(args.force)
        print("")

    print("✅ Done!")


if __name__ == "__main__":
    main()
