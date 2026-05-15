import sqlite3


def clear_thread_checkpoints(db_path: str, thread_id: str) -> int:
    if not thread_id:
        return 0

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
        deleted_writes = cursor.rowcount if cursor.rowcount is not None else 0
        cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
        deleted_checkpoints = cursor.rowcount if cursor.rowcount is not None else 0
        conn.commit()
    return deleted_writes + deleted_checkpoints
