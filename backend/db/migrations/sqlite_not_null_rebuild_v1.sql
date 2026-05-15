-- sqlite_not_null_rebuild_v1.sql
-- Goal:
--   Enforce NOT NULL on selected columns without changing primary key strategy.
--   Keep backup tables for rollback.
--
-- Target constraints:
--   tasks.user_id      NOT NULL
--   tasks.title        NOT NULL
--   tasks.status       NOT NULL DEFAULT 'active'
--   tasks.created_at   NOT NULL
--   messages.task_id   NOT NULL
--   messages.role      NOT NULL
--   messages.content   NOT NULL
--   messages.created_at NOT NULL
--
-- Preconditions:
--   1) Run a full backup of mcm_agent.db before execution.
--   2) Ensure application writes are paused during migration window.
--   3) Ideally run not_null_migration_plan.py first and confirm blockers = 0.

PRAGMA foreign_keys = OFF;
BEGIN IMMEDIATE TRANSACTION;

-- 0) Safety checks: abort if required fields still contain NULL/blank values.
CREATE TEMP TABLE _guard_tasks_user_id(x INTEGER CHECK (x = 1));
INSERT INTO _guard_tasks_user_id VALUES (
  CASE WHEN EXISTS (SELECT 1 FROM tasks WHERE user_id IS NULL OR TRIM(user_id) = '') THEN 0 ELSE 1 END
);
DROP TABLE _guard_tasks_user_id;

CREATE TEMP TABLE _guard_tasks_title(x INTEGER CHECK (x = 1));
INSERT INTO _guard_tasks_title VALUES (
  CASE WHEN EXISTS (SELECT 1 FROM tasks WHERE title IS NULL OR TRIM(title) = '') THEN 0 ELSE 1 END
);
DROP TABLE _guard_tasks_title;

CREATE TEMP TABLE _guard_tasks_status(x INTEGER CHECK (x = 1));
INSERT INTO _guard_tasks_status VALUES (
  CASE WHEN EXISTS (SELECT 1 FROM tasks WHERE status IS NULL OR TRIM(status) = '') THEN 0 ELSE 1 END
);
DROP TABLE _guard_tasks_status;

CREATE TEMP TABLE _guard_tasks_created_at(x INTEGER CHECK (x = 1));
INSERT INTO _guard_tasks_created_at VALUES (
  CASE WHEN EXISTS (SELECT 1 FROM tasks WHERE created_at IS NULL) THEN 0 ELSE 1 END
);
DROP TABLE _guard_tasks_created_at;

CREATE TEMP TABLE _guard_messages_task_id(x INTEGER CHECK (x = 1));
INSERT INTO _guard_messages_task_id VALUES (
  CASE WHEN EXISTS (SELECT 1 FROM messages WHERE task_id IS NULL OR TRIM(task_id) = '') THEN 0 ELSE 1 END
);
DROP TABLE _guard_messages_task_id;

CREATE TEMP TABLE _guard_messages_role(x INTEGER CHECK (x = 1));
INSERT INTO _guard_messages_role VALUES (
  CASE WHEN EXISTS (SELECT 1 FROM messages WHERE role IS NULL OR TRIM(role) = '') THEN 0 ELSE 1 END
);
DROP TABLE _guard_messages_role;

CREATE TEMP TABLE _guard_messages_content(x INTEGER CHECK (x = 1));
INSERT INTO _guard_messages_content VALUES (
  CASE WHEN EXISTS (SELECT 1 FROM messages WHERE content IS NULL OR TRIM(content) = '') THEN 0 ELSE 1 END
);
DROP TABLE _guard_messages_content;

CREATE TEMP TABLE _guard_messages_created_at(x INTEGER CHECK (x = 1));
INSERT INTO _guard_messages_created_at VALUES (
  CASE WHEN EXISTS (SELECT 1 FROM messages WHERE created_at IS NULL) THEN 0 ELSE 1 END
);
DROP TABLE _guard_messages_created_at;

-- 1) Build new tables with stronger NOT NULL constraints.
CREATE TABLE tasks_new (
  id VARCHAR PRIMARY KEY,
  user_id VARCHAR NOT NULL REFERENCES users(id),
  title VARCHAR NOT NULL,
  api_key VARCHAR,
  model_id VARCHAR,
  status VARCHAR NOT NULL DEFAULT 'active',
  created_at DATETIME NOT NULL,
  updated_at DATETIME,
  is_deleted BOOLEAN DEFAULT 0,
  deleted_at DATETIME
);

CREATE TABLE messages_new (
  id VARCHAR PRIMARY KEY,
  task_id VARCHAR NOT NULL REFERENCES tasks_new(id),
  role VARCHAR NOT NULL,
  content TEXT NOT NULL,
  node VARCHAR,
  created_at DATETIME NOT NULL,
  updated_at DATETIME,
  is_deleted BOOLEAN DEFAULT 0,
  deleted_at DATETIME
);

-- 2) Copy data into new tables.
INSERT INTO tasks_new (id, user_id, title, api_key, model_id, status, created_at, updated_at, is_deleted, deleted_at)
SELECT id, user_id, title, api_key, model_id, status, created_at, updated_at, is_deleted, deleted_at
FROM tasks;

INSERT INTO messages_new (id, task_id, role, content, node, created_at, updated_at, is_deleted, deleted_at)
SELECT id, task_id, role, content, node, created_at, updated_at, is_deleted, deleted_at
FROM messages;

-- 3) Sanity check row counts before swap.
CREATE TEMP TABLE _guard_tasks_count(x INTEGER CHECK (x = 1));
INSERT INTO _guard_tasks_count VALUES (
  CASE WHEN (SELECT COUNT(*) FROM tasks_new) = (SELECT COUNT(*) FROM tasks) THEN 1 ELSE 0 END
);
DROP TABLE _guard_tasks_count;

CREATE TEMP TABLE _guard_messages_count(x INTEGER CHECK (x = 1));
INSERT INTO _guard_messages_count VALUES (
  CASE WHEN (SELECT COUNT(*) FROM messages_new) = (SELECT COUNT(*) FROM messages) THEN 1 ELSE 0 END
);
DROP TABLE _guard_messages_count;

-- 4) Swap tables; keep backups for rollback.
DROP TABLE IF EXISTS tasks_backup_v1;
DROP TABLE IF EXISTS messages_backup_v1;

ALTER TABLE tasks RENAME TO tasks_backup_v1;
ALTER TABLE messages RENAME TO messages_backup_v1;

ALTER TABLE tasks_new RENAME TO tasks;
ALTER TABLE messages_new RENAME TO messages;

-- 5) Recreate indexes.
-- Note: index names are globally unique in SQLite. Backup tables may still hold old names.
DROP INDEX IF EXISTS ix_tasks_created_at;
DROP INDEX IF EXISTS ix_tasks_user_id;
DROP INDEX IF EXISTS ix_messages_created_at;
DROP INDEX IF EXISTS ix_messages_task_id;
DROP INDEX IF EXISTS ix_messages_id;

CREATE INDEX ix_tasks_created_at ON tasks(created_at);
CREATE INDEX ix_tasks_user_id ON tasks(user_id);
CREATE INDEX ix_messages_created_at ON messages(created_at);
CREATE INDEX ix_messages_task_id ON messages(task_id);
CREATE INDEX ix_messages_id ON messages(id);

COMMIT;
PRAGMA foreign_keys = ON;

-- 6) Post-checks (run after COMMIT):
-- PRAGMA foreign_key_check;
-- PRAGMA integrity_check;
