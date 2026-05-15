import argparse
import sqlite3
from pathlib import Path


def read_sql(sql_path: Path) -> str:
    return sql_path.read_text(encoding="utf-8")


def run_migration(db_path: Path, sql_path: Path) -> None:
    sql = read_sql(sql_path)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(sql)
        conn.commit()
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-click runner for sqlite_not_null_rebuild_v1.sql"
    )
    parser.add_argument(
        "--db-path",
        default="mcm_agent.db",
        help="SQLite database path",
    )
    parser.add_argument(
        "--sql-path",
        default="backend/db/migrations/sqlite_not_null_rebuild_v1.sql",
        help="SQL script path",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually execute migration. Without this flag, only preview.",
    )
    args = parser.parse_args()

    db_path = Path(args.db_path)
    sql_path = Path(args.sql_path)
    if not db_path.exists():
        raise SystemExit(f"Database not found: {db_path}")
    if not sql_path.exists():
        raise SystemExit(f"SQL script not found: {sql_path}")

    if not args.execute:
        print("Dry-run mode. Migration not executed.")
        print(f"Database: {db_path}")
        print(f"SQL script: {sql_path}")
        print("Use --execute to run migration.")
        raise SystemExit(0)

    run_migration(db_path=db_path, sql_path=sql_path)
    print("Migration completed.")
    print("Next checks:")
    print("  PRAGMA foreign_key_check;")
    print("  PRAGMA integrity_check;")


if __name__ == "__main__":
    main()
