import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple


AUTO_FIX_QUERIES: Dict[str, str] = {
    "tasks.status": "UPDATE tasks SET status='active' WHERE status IS NULL OR TRIM(status)=''",
    "tasks.created_at": "UPDATE tasks SET created_at=CURRENT_TIMESTAMP WHERE created_at IS NULL",
    "tasks.title": "UPDATE tasks SET title='untitled_task' WHERE title IS NULL OR TRIM(title)=''",
    "messages.role": "UPDATE messages SET role='system' WHERE role IS NULL OR TRIM(role)=''",
    "messages.content": "UPDATE messages SET content='[]' WHERE content IS NULL OR TRIM(content)=''",
    "messages.created_at": "UPDATE messages SET created_at=CURRENT_TIMESTAMP WHERE created_at IS NULL",
}


MANUAL_BLOCKER_CHECKS: Dict[str, str] = {
    "tasks.user_id": "SELECT COUNT(*) FROM tasks WHERE user_id IS NULL OR TRIM(user_id)=''",
    "messages.task_id": "SELECT COUNT(*) FROM messages WHERE task_id IS NULL OR TRIM(task_id)=''",
}


NULL_CHECK_QUERIES: Dict[str, str] = {
    "tasks.user_id": "SELECT COUNT(*) FROM tasks WHERE user_id IS NULL OR TRIM(user_id)=''",
    "tasks.title": "SELECT COUNT(*) FROM tasks WHERE title IS NULL OR TRIM(title)=''",
    "tasks.status": "SELECT COUNT(*) FROM tasks WHERE status IS NULL OR TRIM(status)=''",
    "tasks.created_at": "SELECT COUNT(*) FROM tasks WHERE created_at IS NULL",
    "messages.task_id": "SELECT COUNT(*) FROM messages WHERE task_id IS NULL OR TRIM(task_id)=''",
    "messages.role": "SELECT COUNT(*) FROM messages WHERE role IS NULL OR TRIM(role)=''",
    "messages.content": "SELECT COUNT(*) FROM messages WHERE content IS NULL OR TRIM(content)=''",
    "messages.created_at": "SELECT COUNT(*) FROM messages WHERE created_at IS NULL",
}


ORPHAN_CHECK_QUERIES: Dict[str, str] = {
    "orphan_tasks": (
        "SELECT COUNT(*) FROM tasks t "
        "LEFT JOIN users u ON t.user_id = u.id "
        "WHERE t.user_id IS NOT NULL AND u.id IS NULL"
    ),
    "orphan_messages": (
        "SELECT COUNT(*) FROM messages m "
        "LEFT JOIN tasks t ON m.task_id = t.id "
        "WHERE m.task_id IS NOT NULL AND t.id IS NULL"
    ),
}


def _fetch_count(conn: sqlite3.Connection, query: str) -> int:
    return int(conn.execute(query).fetchone()[0])


def audit(conn: sqlite3.Connection) -> Dict[str, Dict[str, int]]:
    null_counts = {key: _fetch_count(conn, query) for key, query in NULL_CHECK_QUERIES.items()}
    orphan_counts = {key: _fetch_count(conn, query) for key, query in ORPHAN_CHECK_QUERIES.items()}
    blocker_counts = {key: _fetch_count(conn, query) for key, query in MANUAL_BLOCKER_CHECKS.items()}
    return {
        "null_counts": null_counts,
        "orphan_counts": orphan_counts,
        "manual_blockers": blocker_counts,
    }


def apply_safe_backfill(conn: sqlite3.Connection) -> Dict[str, int]:
    affected: Dict[str, int] = {}
    for key, query in AUTO_FIX_QUERIES.items():
        cur = conn.execute(query)
        affected[key] = cur.rowcount if cur.rowcount is not None else 0
    conn.commit()
    return affected


def _sqlite_rebuild_template() -> List[str]:
    # SQLite 不支持直接修改列的 NOT NULL，采用重建表模板。
    return [
        "-- Phase A: 先完成数据清洗（本脚本可自动处理部分字段）",
        "-- Phase B: 在低峰窗口执行表重建迁移（示例模板如下）",
        "BEGIN TRANSACTION;",
        "",
        "-- 1) tasks 表重建（为 user_id/title/status/created_at 加 NOT NULL）",
        "CREATE TABLE tasks_new (",
        "  id VARCHAR PRIMARY KEY,",
        "  user_id VARCHAR NOT NULL REFERENCES users(id),",
        "  title VARCHAR NOT NULL,",
        "  api_key VARCHAR,",
        "  model_id VARCHAR,",
        "  status VARCHAR NOT NULL DEFAULT 'active',",
        "  created_at DATETIME NOT NULL,",
        "  updated_at DATETIME,",
        "  is_deleted BOOLEAN DEFAULT 0,",
        "  deleted_at DATETIME",
        ");",
        "INSERT INTO tasks_new (id,user_id,title,api_key,model_id,status,created_at,updated_at,is_deleted,deleted_at)",
        "SELECT id,user_id,title,api_key,model_id,status,created_at,updated_at,is_deleted,deleted_at FROM tasks;",
        "DROP TABLE tasks;",
        "ALTER TABLE tasks_new RENAME TO tasks;",
        "",
        "-- 2) messages 表重建（为 task_id/role/content/created_at 加 NOT NULL）",
        "CREATE TABLE messages_new (",
        "  id VARCHAR PRIMARY KEY,",
        "  task_id VARCHAR NOT NULL REFERENCES tasks(id),",
        "  role VARCHAR NOT NULL,",
        "  content TEXT NOT NULL,",
        "  node VARCHAR,",
        "  created_at DATETIME NOT NULL,",
        "  updated_at DATETIME,",
        "  is_deleted BOOLEAN DEFAULT 0,",
        "  deleted_at DATETIME",
        ");",
        "INSERT INTO messages_new (id,task_id,role,content,node,created_at,updated_at,is_deleted,deleted_at)",
        "SELECT id,task_id,role,content,node,created_at,updated_at,is_deleted,deleted_at FROM messages;",
        "DROP TABLE messages;",
        "ALTER TABLE messages_new RENAME TO messages;",
        "",
        "-- 3) 迁移后重建索引",
        "CREATE INDEX IF NOT EXISTS ix_tasks_created_at ON tasks(created_at);",
        "CREATE INDEX IF NOT EXISTS ix_tasks_user_id ON tasks(user_id);",
        "CREATE INDEX IF NOT EXISTS ix_messages_created_at ON messages(created_at);",
        "CREATE INDEX IF NOT EXISTS ix_messages_task_id ON messages(task_id);",
        "",
        "COMMIT;",
    ]


def _build_report(
    db_path: str,
    before: Dict[str, Dict[str, int]],
    after: Dict[str, Dict[str, int]],
    auto_fix_rows: Dict[str, int],
) -> Dict[str, object]:
    blockers = {
        key: value for key, value in after["manual_blockers"].items() if value > 0
    }
    orphan_blockers = {
        key: value for key, value in after["orphan_counts"].items() if value > 0
    }
    ready_for_not_null = not blockers and not orphan_blockers
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "database_path": db_path,
        "before": before,
        "auto_fix_rows": auto_fix_rows,
        "after": after,
        "blocking_items": {
            "manual_blockers": blockers,
            "orphan_blockers": orphan_blockers,
        },
        "ready_for_not_null_rebuild": ready_for_not_null,
        "notes": [
            "本脚本默认仅做审计（dry-run）。",
            "使用 --apply-auto-fix 可以先执行低风险字段回填，但不会修改表结构。",
            "最终 NOT NULL 迁移建议按模板执行重建表（低峰窗口 + 备份）。",
        ],
    }


def run(db_path: str, enable_auto_fix: bool, report_path: str, sql_template_path: str) -> int:
    conn = sqlite3.connect(db_path)
    try:
        before = audit(conn)
        fixed_rows: Dict[str, int] = {}
        if enable_auto_fix:
            fixed_rows = apply_safe_backfill(conn)
        after = audit(conn)
    finally:
        conn.close()

    report = _build_report(db_path, before, after, fixed_rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if report_path:
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if sql_template_path:
        sql_path = Path(sql_template_path)
        sql_path.parent.mkdir(parents=True, exist_ok=True)
        sql_path.write_text("\n".join(_sqlite_rebuild_template()) + "\n", encoding="utf-8")

    return 0 if report["ready_for_not_null_rebuild"] else 2


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare safe NOT NULL migration for SQLite (audit first, schema unchanged)."
    )
    parser.add_argument("--db-path", default="mcm_agent.db", help="SQLite database file path")
    parser.add_argument(
        "--apply-auto-fix",
        action="store_true",
        help="Apply safe backfill updates before reporting (no schema alteration)",
    )
    parser.add_argument(
        "--report-path",
        default="logs/not_null_migration_report.json",
        help="Optional JSON report output path",
    )
    parser.add_argument(
        "--sql-template-path",
        default="logs/not_null_rebuild_template.sql",
        help="Optional SQL template output path for table rebuild migration",
    )
    args = parser.parse_args()
    code = run(
        db_path=args.db_path,
        enable_auto_fix=args.apply_auto_fix,
        report_path=args.report_path,
        sql_template_path=args.sql_template_path,
    )
    raise SystemExit(code)


if __name__ == "__main__":
    main()
