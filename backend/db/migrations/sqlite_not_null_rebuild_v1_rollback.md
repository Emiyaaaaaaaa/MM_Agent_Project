# sqlite_not_null_rebuild_v1 rollback

## Scope

This rollback guide is for `sqlite_not_null_rebuild_v1.sql`.
It assumes migration completed and backup tables still exist:

- `tasks_backup_v1`
- `messages_backup_v1`

## Preconditions

- Stop backend write traffic first.
- Keep one file-level backup of `mcm_agent.db` before rollback.
- Run rollback in the same database file that executed v1 migration.

## Rollback SQL

```sql
PRAGMA foreign_keys = OFF;
BEGIN IMMEDIATE TRANSACTION;

-- Safety: require backup tables.
SELECT CASE
  WHEN NOT EXISTS (SELECT 1 FROM sqlite_master WHERE type='table' AND name='tasks_backup_v1')
  THEN RAISE(ABORT, 'rollback blocked: missing tasks_backup_v1')
END;
SELECT CASE
  WHEN NOT EXISTS (SELECT 1 FROM sqlite_master WHERE type='table' AND name='messages_backup_v1')
  THEN RAISE(ABORT, 'rollback blocked: missing messages_backup_v1')
END;

DROP TABLE IF EXISTS messages;
DROP TABLE IF EXISTS tasks;

ALTER TABLE tasks_backup_v1 RENAME TO tasks;
ALTER TABLE messages_backup_v1 RENAME TO messages;

-- Rebuild expected indexes.
CREATE INDEX IF NOT EXISTS ix_tasks_created_at ON tasks(created_at);
CREATE INDEX IF NOT EXISTS ix_tasks_user_id ON tasks(user_id);
CREATE INDEX IF NOT EXISTS ix_messages_created_at ON messages(created_at);
CREATE INDEX IF NOT EXISTS ix_messages_task_id ON messages(task_id);
CREATE INDEX IF NOT EXISTS ix_messages_id ON messages(id);

COMMIT;
PRAGMA foreign_keys = ON;
```

## Verification after rollback

Run these checks:

```sql
PRAGMA foreign_key_check;
PRAGMA integrity_check;
PRAGMA index_list(tasks);
PRAGMA index_list(messages);
```

Expected:

- `foreign_key_check` returns no rows.
- `integrity_check` returns `ok`.
- required indexes exist on `tasks` and `messages`.

## Cleanup policy

- If rollback is successful and confirmed, keep a copy of the DB backup.
- Do not immediately drop any extra backup artifacts until service is stable.
