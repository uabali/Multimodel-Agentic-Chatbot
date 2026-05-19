"""
Minimal SQLite-backed Chainlit Data Layer for thread persistence.

Source: Final-Project/src/persistence/sqlite_data_layer.py (full port).
"""

import asyncio
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from chainlit.data import BaseDataLayer
from chainlit.element import ElementDict
from chainlit.step import StepDict
from chainlit.types import PaginatedResponse, Pagination, ThreadDict, ThreadFilter, PageInfo
from chainlit.user import PersistedUser, User


def _now_iso() -> str:
    """Kısa: `_now_iso` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_loads(value: Any, default: Any = None) -> Any:
    """Kısa: `_json_loads` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return value
    if value == "":
        return default
    try:
        return json.loads(value)
    except Exception:
        return value


def _as_text(value: Any) -> str:
    """Kısa: `_as_text` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
    loaded = _json_loads(value, "")
    if loaded is None:
        return ""
    if isinstance(loaded, str):
        return loaded
    try:
        return json.dumps(loaded, ensure_ascii=False)
    except Exception:
        return str(loaded)


class SQLiteDataLayer(BaseDataLayer):
    def __init__(self, db_path: str | Path):
        """Kısa: `__init__` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
        self.db_path = str(db_path)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        """Kısa: `_connect` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self) -> None:
        """Kısa: `_init_db` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY, identifier TEXT UNIQUE NOT NULL,
                    createdAt TEXT NOT NULL, metadata TEXT NOT NULL)""")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS threads (
                    id TEXT PRIMARY KEY, name TEXT, userId TEXT, userIdentifier TEXT,
                    createdAt TEXT NOT NULL, updatedAt TEXT NOT NULL,
                    tags TEXT NOT NULL, metadata TEXT NOT NULL)""")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS steps (
                    id TEXT PRIMARY KEY, threadId TEXT NOT NULL, parentId TEXT,
                    name TEXT, type TEXT, input TEXT, output TEXT,
                    metadata TEXT NOT NULL, createdAt TEXT NOT NULL,
                    startTime TEXT, endTime TEXT, tags TEXT NOT NULL,
                    FOREIGN KEY(threadId) REFERENCES threads(id))""")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS elements (
                    id TEXT PRIMARY KEY, threadId TEXT NOT NULL,
                    data TEXT NOT NULL, createdAt TEXT NOT NULL,
                    FOREIGN KEY(threadId) REFERENCES threads(id))""")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id TEXT PRIMARY KEY, data TEXT NOT NULL, createdAt TEXT NOT NULL)""")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS mcp_saved_configs (
                    id TEXT PRIMARY KEY, userIdentifier TEXT NOT NULL,
                    name TEXT NOT NULL, command TEXT NOT NULL,
                    args TEXT NOT NULL, env TEXT NOT NULL,
                    createdAt TEXT NOT NULL, updatedAt TEXT NOT NULL,
                    UNIQUE(userIdentifier, name))""")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS _migrations (
                    id TEXT PRIMARY KEY, appliedAt TEXT NOT NULL)""")
            # One-time backfill: populate userIdentifier from users table
            cur.execute("SELECT id FROM _migrations WHERE id='backfill_user_identifier'")
            if not cur.fetchone():
                cur.execute("""
                    UPDATE threads SET userIdentifier = (
                        SELECT identifier FROM users WHERE users.id = threads.userId)
                    WHERE (userIdentifier IS NULL OR userIdentifier = '')
                      AND userId IS NOT NULL
                      AND EXISTS (SELECT 1 FROM users WHERE users.id = threads.userId)""")
                cur.execute(
                    "INSERT INTO _migrations (id, appliedAt) VALUES ('backfill_user_identifier', ?)",
                    (_now_iso(),)
                )
            conn.commit()

    async def build_debug_url(self) -> str:
        """Kısa: `build_debug_url` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
        return ""

    async def close(self) -> None:
        """Kısa: `close` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
        return None

    async def _run(self, fn, *args, **kwargs):
        """Kısa: `_run` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
        return await asyncio.to_thread(fn, *args, **kwargs)

    def _ensure_thread(self, thread_id: str, user_id: Optional[str], user_identifier: Optional[str]):
        """Kısa: `_ensure_thread` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT OR IGNORE INTO threads (id,name,userId,userIdentifier,createdAt,updatedAt,tags,metadata) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (thread_id, None, user_id, user_identifier, _now_iso(), _now_iso(), "[]", "{}"))
            cur.execute("SELECT userId,userIdentifier FROM threads WHERE id=?", (thread_id,))
            row = cur.fetchone()
            new_uid = user_id or (row["userId"] if row else None)
            new_uident = user_identifier or (row["userIdentifier"] if row else None)
            cur.execute("UPDATE threads SET updatedAt=?,userId=?,userIdentifier=? WHERE id=?",
                        (_now_iso(), new_uid, new_uident, thread_id))
            conn.commit()

    def _get_user_identifier_by_id(self, user_id: str) -> Optional[str]:
        """Kısa: `_get_user_identifier_by_id` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT identifier FROM users WHERE id=?", (user_id,))
            row = cur.fetchone()
            return row["identifier"] if row else None

    async def get_user(self, identifier: str) -> Optional[PersistedUser]:
        """Kısa: `get_user` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
        def _get():
            """Kısa: `_get` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
            with self._connect() as conn:
                cur = conn.cursor()
                cur.execute("SELECT * FROM users WHERE identifier=?", (identifier,))
                row = cur.fetchone()
                if not row:
                    return None
                return PersistedUser(id=row["id"], identifier=row["identifier"],
                                     createdAt=row["createdAt"], metadata=json.loads(row["metadata"] or "{}"))
        return await self._run(_get)

    async def create_user(self, user: User) -> Optional[PersistedUser]:
        """Kısa: `create_user` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
        async def _create():
            """Kısa: `_create` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
            existing = await self.get_user(user.identifier)
            with self._connect() as conn:
                cur = conn.cursor()
                if not existing:
                    cur.execute("INSERT INTO users (id,identifier,createdAt,metadata) VALUES (?,?,?,?)",
                                (str(uuid.uuid4()), user.identifier, _now_iso(), json.dumps(user.metadata or {})))
                else:
                    cur.execute("UPDATE users SET metadata=? WHERE identifier=?",
                                (json.dumps(user.metadata or {}), user.identifier))
                conn.commit()
            return await self.get_user(user.identifier)
        return await _create()

    async def upsert_feedback(self, feedback) -> str:
        """Kısa: `upsert_feedback` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
        fid = getattr(feedback, "id", None) or str(uuid.uuid4())
        def _up():
            """Kısa: `_up` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
            from dataclasses import asdict
            with self._connect() as conn:
                conn.cursor().execute("INSERT OR REPLACE INTO feedback (id,data,createdAt) VALUES (?,?,?)",
                                      (fid, json.dumps(asdict(feedback)), _now_iso()))
                conn.commit()
        await self._run(_up)
        return fid

    async def delete_feedback(self, feedback_id: str) -> bool:
        """Kısa: `delete_feedback` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
        def _d():
            """Kısa: `_d` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
            with self._connect() as conn:
                cur = conn.cursor()
                cur.execute("DELETE FROM feedback WHERE id=?", (feedback_id,))
                conn.commit()
                return cur.rowcount > 0
        return await self._run(_d)

    async def create_element(self, element) -> None:
        """Kısa: `create_element` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
        ed: ElementDict = element.to_dict()
        eid = ed.get("id") or str(uuid.uuid4())
        tid = ed.get("threadId")
        if not tid:
            return
        def _c():
            """Kısa: `_c` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
            with self._connect() as conn:
                conn.cursor().execute("INSERT OR REPLACE INTO elements (id,threadId,data,createdAt) VALUES (?,?,?,?)",
                                      (eid, tid, json.dumps(ed), _now_iso()))
                conn.commit()
        await self._run(_c)

    async def get_element(self, thread_id: str, element_id: str) -> Optional[ElementDict]:
        """Kısa: `get_element` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
        def _g():
            """Kısa: `_g` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
            with self._connect() as conn:
                cur = conn.cursor()
                cur.execute("SELECT data FROM elements WHERE id=? AND threadId=?", (element_id, thread_id))
                row = cur.fetchone()
                return json.loads(row["data"]) if row else None
        return await self._run(_g)

    async def delete_element(self, element_id: str, thread_id: Optional[str] = None):
        """Kısa: `delete_element` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
        def _d():
            """Kısa: `_d` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
            with self._connect() as conn:
                cur = conn.cursor()
                if thread_id:
                    cur.execute("DELETE FROM elements WHERE id=? AND threadId=?", (element_id, thread_id))
                else:
                    cur.execute("DELETE FROM elements WHERE id=?", (element_id,))
                conn.commit()
        await self._run(_d)

    async def create_step(self, step_dict: StepDict):
        """Kısa: `create_step` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
        def _c():
            """Kısa: `_c` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
            tid = step_dict.get("threadId")
            if not tid:
                return
            uid = step_dict.get("userId")
            uident = step_dict.get("userIdentifier")
            if not uident and uid:
                uident = self._get_user_identifier_by_id(uid)
            self._ensure_thread(tid, uid, uident)
            with self._connect() as conn:
                conn.cursor().execute(
                    "INSERT OR REPLACE INTO steps "
                    "(id,threadId,parentId,name,type,input,output,metadata,createdAt,startTime,endTime,tags) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (step_dict.get("id") or str(uuid.uuid4()), tid, step_dict.get("parentId"),
                     step_dict.get("name"), step_dict.get("type"),
                     json.dumps(step_dict.get("input")) if step_dict.get("input") is not None else None,
                     json.dumps(step_dict.get("output")) if step_dict.get("output") is not None else None,
                     json.dumps(step_dict.get("metadata") or {}),
                     step_dict.get("createdAt") or _now_iso(),
                     step_dict.get("start") or step_dict.get("startTime"),
                     step_dict.get("end") or step_dict.get("endTime"),
                     json.dumps(step_dict.get("tags") or [])))
                conn.commit()
        await self._run(_c)

    async def update_step(self, step_dict: StepDict):
        """Kısa: `update_step` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
        await self.create_step(step_dict)

    async def delete_step(self, step_id: str):
        """Kısa: `delete_step` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
        def _d():
            """Kısa: `_d` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
            with self._connect() as conn:
                conn.cursor().execute("DELETE FROM steps WHERE id=?", (step_id,))
                conn.commit()
        await self._run(_d)

    async def get_favorite_steps(self, user_id: str) -> List[StepDict]:
        """Kısa: `get_favorite_steps` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
        return []

    async def get_thread_author(self, thread_id: str) -> str:
        """Kısa: `get_thread_author` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
        def _g():
            """Kısa: `_g` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
            with self._connect() as conn:
                cur = conn.cursor()
                cur.execute("SELECT userIdentifier,userId FROM threads WHERE id=?", (thread_id,))
                row = cur.fetchone()
                if not row:
                    return ""
                if row["userIdentifier"]:
                    return row["userIdentifier"]
                if row["userId"]:
                    return self._get_user_identifier_by_id(row["userId"]) or ""
                return ""
        return await self._run(_g)

    async def delete_thread(self, thread_id: str):
        """Kısa: `delete_thread` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
        def _d():
            """Kısa: `_d` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
            with self._connect() as conn:
                cur = conn.cursor()
                cur.execute("DELETE FROM steps WHERE threadId=?", (thread_id,))
                cur.execute("DELETE FROM elements WHERE threadId=?", (thread_id,))
                cur.execute("DELETE FROM threads WHERE id=?", (thread_id,))
                conn.commit()
        await self._run(_d)

    async def list_threads(self, pagination: Pagination, filters: ThreadFilter) -> PaginatedResponse[ThreadDict]:
        """Kısa: `list_threads` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
        def _list():
            """Kısa: `_list` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
            user_id = filters.get("userId") if isinstance(filters, dict) else getattr(filters, "userId", None)
            search = filters.get("search") if isinstance(filters, dict) else getattr(filters, "search", None)
            limit = int(getattr(pagination, "first", None) or 20)
            cursor = getattr(pagination, "cursor", None)
            try:
                offset = int(cursor) if cursor else 0
            except ValueError:
                offset = 0

            where, params = [], []
            if user_id:
                where.append("userId=?"); params.append(user_id)
            if search:
                where.append("(name LIKE ?)"); params.append(f"%{search}%")
            wsql = ("WHERE " + " AND ".join(where)) if where else ""

            with self._connect() as conn:
                cur = conn.cursor()
                cur.execute(f"SELECT COUNT(1) as c FROM threads {wsql}", tuple(params))
                total = int(cur.fetchone()["c"])
                cur.execute(f"SELECT * FROM threads {wsql} ORDER BY updatedAt DESC LIMIT ? OFFSET ?",
                            tuple(params + [limit, offset]))
                rows = cur.fetchall()

            threads = [{
                "id": r["id"], "name": r["name"], "userId": r["userId"],
                "userIdentifier": r["userIdentifier"], "createdAt": r["createdAt"],
                "updatedAt": r["updatedAt"], "tags": json.loads(r["tags"] or "[]"),
                "metadata": json.loads(r["metadata"] or "{}"),
            } for r in rows]

            has_next = (offset + limit) < total
            return PaginatedResponse(
                data=threads,
                pageInfo=PageInfo(hasNextPage=has_next, startCursor=str(offset), endCursor=str(offset + len(threads))))
        return await self._run(_list)

    async def get_thread(self, thread_id: str) -> Optional[ThreadDict]:
        """Kısa: `get_thread` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
        def _g():
            """Kısa: `_g` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
            with self._connect() as conn:
                cur = conn.cursor()
                cur.execute("SELECT * FROM threads WHERE id=?", (thread_id,))
                t = cur.fetchone()
                if not t:
                    return None
                cur.execute("SELECT * FROM steps WHERE threadId=? ORDER BY createdAt ASC", (thread_id,))
                steps = []
                for s in cur.fetchall():
                    input_text = _as_text(s["input"])
                    output_text = _as_text(s["output"])
                    # Chainlit stores user_message text in output for this data layer.
                    # Mirror it into input as well so resume rendering/history works.
                    if s["type"] == "user_message" and not input_text and output_text:
                        input_text = output_text
                    steps.append({
                        "id": s["id"],
                        "threadId": s["threadId"],
                        "parentId": s["parentId"],
                        "name": s["name"] or "",
                        "type": s["type"] or "undefined",
                        "input": input_text,
                        "output": output_text,
                        "metadata": _json_loads(s["metadata"], {}) or {},
                        "createdAt": s["createdAt"],
                        "start": s["startTime"] or s["createdAt"],
                        "end": s["endTime"] or s["createdAt"],
                        "tags": _json_loads(s["tags"], []) or [],
                        "streaming": False,
                        "waitForAnswer": None,
                        "isError": False,
                        "command": None,
                        "modes": None,
                        "generation": None,
                        "showInput": None,
                        "defaultOpen": None,
                        "autoCollapse": None,
                        "language": None,
                        "icon": None,
                        "feedback": None,
                    })
                cur.execute("SELECT data FROM elements WHERE threadId=? ORDER BY createdAt ASC", (thread_id,))
                elements = [_json_loads(row["data"], {}) for row in cur.fetchall()]
            return {
                "id": t["id"], "name": t["name"], "userId": t["userId"],
                "userIdentifier": t["userIdentifier"], "createdAt": t["createdAt"],
                "updatedAt": t["updatedAt"], "tags": json.loads(t["tags"] or "[]"),
                "metadata": _json_loads(t["metadata"], {}) or {},
                "steps": steps,
                "elements": elements,
            }
        return await self._run(_g)

    async def patch_thread_metadata(self, thread_id: str, patch: Dict) -> None:
        """Mevcut thread metadata'sını değiştirmeden patch uygular (JSON merge)."""
        def _p():
            """Kısa: `_p` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
            with self._connect() as conn:
                cur = conn.cursor()
                cur.execute("SELECT metadata FROM threads WHERE id=?", (thread_id,))
                row = cur.fetchone()
                existing = json.loads(row["metadata"] or "{}") if row else {}
                existing.update(patch)
                cur.execute(
                    "UPDATE threads SET metadata=?, updatedAt=? WHERE id=?",
                    (json.dumps(existing), _now_iso(), thread_id),
                )
                conn.commit()
        await self._run(_p)

    async def update_thread(self, thread_id: str, name: Optional[str] = None,
                            user_id: Optional[str] = None, metadata: Optional[Dict] = None,
                            tags: Optional[List[str]] = None):
        """Kısa: `update_thread` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
        def _u():
            """Kısa: `_u` işlevini yürütür. Bağlantı: modül akışıyla entegredir."""
            user_identifier = self._get_user_identifier_by_id(user_id) if user_id else None
            self._ensure_thread(thread_id, user_id, user_identifier)
            fields, params = [], []
            if name is not None:
                fields.append("name=?"); params.append(name)
            if user_id is not None:
                fields.append("userId=?"); params.append(user_id)
            if metadata is not None:
                fields.append("metadata=?"); params.append(json.dumps(metadata))
            if tags is not None:
                fields.append("tags=?"); params.append(json.dumps(tags))
            fields.append("updatedAt=?"); params.append(_now_iso())
            params.append(thread_id)
            with self._connect() as conn:
                conn.cursor().execute(f"UPDATE threads SET {', '.join(fields)} WHERE id=?", tuple(params))
                conn.commit()
        await self._run(_u)
