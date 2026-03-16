"""
Unified Repository — thin CRUD layer over all ORM models.

SL had 24 separate repository files.  We keep a single class that groups
related queries by docstring sections.  Each public method is a thin async
helper around an SQLAlchemy query.
"""
from __future__ import annotations

import datetime as _dt
from typing import Any, Optional, Sequence

from sqlalchemy import delete, select, update, func, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert as pg_insert

from .models import (
    ActiveThread,
    BotResponse,
    EmotionState,
    GroupProfile,
    JargonTerm,
    LearningJob,
    RawMessage,
    ThreadMessage,
    UserMemory,
)


class Repository:
    """All database operations in one place."""

    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    # ---- helpers ----

    async def commit(self) -> None:
        await self._s.commit()

    async def flush(self) -> None:
        await self._s.flush()

    # =====================================================================
    #  RawMessage
    # =====================================================================

    async def save_raw_message(
        self,
        group_id: str,
        sender_id: str,
        sender_name: str,
        text_content: str,
        platform: str = "",
        embedding: Any = None,
        timestamp: _dt.datetime | None = None,
    ) -> int:
        msg = RawMessage(
            group_id=group_id,
            sender_id=sender_id,
            sender_name=sender_name,
            text=text_content,
            platform=platform,
            embedding=embedding,
            timestamp=timestamp or _dt.datetime.now(_dt.timezone.utc),
        )
        self._s.add(msg)
        await self._s.flush()
        return msg.id

    async def get_recent_messages(
        self, group_id: str, limit: int = 50
    ) -> Sequence[RawMessage]:
        stmt = (
            select(RawMessage)
            .where(RawMessage.group_id == group_id)
            .order_by(RawMessage.timestamp.desc())
            .limit(limit)
        )
        result = await self._s.execute(stmt)
        return result.scalars().all()

    async def count_messages_since(
        self, group_id: str, since: _dt.datetime
    ) -> int:
        stmt = (
            select(func.count())
            .select_from(RawMessage)
            .where(RawMessage.group_id == group_id, RawMessage.timestamp >= since)
        )
        result = await self._s.execute(stmt)
        return result.scalar_one()

    async def update_message_embedding(self, message_id: int, embedding: Any) -> None:
        stmt = (
            update(RawMessage)
            .where(RawMessage.id == message_id)
            .values(embedding=embedding)
        )
        await self._s.execute(stmt)

    # =====================================================================
    #  GroupProfile
    # =====================================================================

    async def get_or_create_group_profile(self, group_id: str) -> GroupProfile:
        stmt = select(GroupProfile).where(GroupProfile.group_id == group_id)
        result = await self._s.execute(stmt)
        profile = result.scalar_one_or_none()
        if profile is None:
            profile = GroupProfile(group_id=group_id)
            self._s.add(profile)
            await self._s.flush()
        return profile

    async def update_group_profile(
        self,
        group_id: str,
        *,
        base_prompt: str | None = None,
        source_whitelist: dict | None = None,
        learned_prompt: str | None = None,
        learned_prompt_history: dict | None = None,
        message_count_since_learn: int | None = None,
    ) -> None:
        values: dict[str, Any] = {}
        if base_prompt is not None:
            values["base_prompt"] = base_prompt
        if source_whitelist is not None:
            values["source_whitelist"] = source_whitelist
        if learned_prompt is not None:
            values["learned_prompt"] = learned_prompt
        if learned_prompt_history is not None:
            values["learned_prompt_history"] = learned_prompt_history
        if message_count_since_learn is not None:
            values["message_count_since_learn"] = message_count_since_learn
        if values:
            stmt = (
                update(GroupProfile)
                .where(GroupProfile.group_id == group_id)
                .values(**values)
            )
            await self._s.execute(stmt)

    async def increment_group_message_count(self, group_id: str) -> int:
        """Increment and return the new message_count_since_learn."""
        profile = await self.get_or_create_group_profile(group_id)
        profile.message_count_since_learn += 1
        await self._s.flush()
        return profile.message_count_since_learn

    async def list_all_groups(self) -> Sequence[GroupProfile]:
        stmt = select(GroupProfile).order_by(GroupProfile.updated_at.desc())
        result = await self._s.execute(stmt)
        return result.scalars().all()

    # =====================================================================
    #  ActiveThread
    # =====================================================================

    async def get_active_threads(
        self, group_id: str, limit: int = 10
    ) -> Sequence[ActiveThread]:
        stmt = (
            select(ActiveThread)
            .where(
                ActiveThread.group_id == group_id,
                ActiveThread.is_archived == False,  # noqa: E712
            )
            .order_by(ActiveThread.last_activity.desc())
            .limit(limit)
        )
        result = await self._s.execute(stmt)
        return result.scalars().all()

    async def create_thread(
        self, group_id: str, topic_summary: str = "", centroid: Any = None
    ) -> ActiveThread:
        thread = ActiveThread(
            group_id=group_id,
            topic_summary=topic_summary,
            centroid=centroid,
            message_count=0,
        )
        self._s.add(thread)
        await self._s.flush()
        return thread

    async def update_thread(
        self,
        thread_id: int,
        *,
        topic_summary: str | None = None,
        centroid: Any = None,
        message_count: int | None = None,
        last_activity: _dt.datetime | None = None,
    ) -> None:
        values: dict[str, Any] = {}
        if topic_summary is not None:
            values["topic_summary"] = topic_summary
        if centroid is not None:
            values["centroid"] = centroid
        if message_count is not None:
            values["message_count"] = message_count
        if last_activity is not None:
            values["last_activity"] = last_activity
        if values:
            stmt = (
                update(ActiveThread)
                .where(ActiveThread.id == thread_id)
                .values(**values)
            )
            await self._s.execute(stmt)

    async def archive_stale_threads(
        self, group_id: str, before: _dt.datetime
    ) -> int:
        stmt = (
            update(ActiveThread)
            .where(
                ActiveThread.group_id == group_id,
                ActiveThread.is_archived == False,  # noqa: E712
                ActiveThread.last_activity < before,
            )
            .values(is_archived=True)
        )
        result = await self._s.execute(stmt)
        return result.rowcount  # type: ignore[return-value]

    async def add_message_to_thread(
        self, message_id: int, thread_id: int
    ) -> None:
        tm = ThreadMessage(message_id=message_id, thread_id=thread_id)
        self._s.add(tm)

    async def get_thread_messages(
        self, thread_id: int, limit: int = 20
    ) -> Sequence[RawMessage]:
        stmt = (
            select(RawMessage)
            .join(ThreadMessage, ThreadMessage.message_id == RawMessage.id)
            .where(ThreadMessage.thread_id == thread_id)
            .order_by(RawMessage.timestamp.desc())
            .limit(limit)
        )
        result = await self._s.execute(stmt)
        return result.scalars().all()

    # =====================================================================
    #  UserMemory
    # =====================================================================

    async def add_user_memory(
        self,
        group_id: str,
        user_id: str,
        fact: str,
        importance: float = 0.5,
        embedding: Any = None,
    ) -> int:
        mem = UserMemory(
            group_id=group_id,
            user_id=user_id,
            fact=fact,
            importance=importance,
            embedding=embedding,
        )
        self._s.add(mem)
        await self._s.flush()
        return mem.id

    async def search_user_memories(
        self,
        group_id: str,
        user_id: str,
        query_embedding: Any,
        top_k: int = 5,
    ) -> Sequence[UserMemory]:
        """pgvector nearest-neighbor search on user_memories."""
        stmt = (
            select(UserMemory)
            .where(
                UserMemory.group_id == group_id,
                UserMemory.user_id == user_id,
                UserMemory.embedding.isnot(None),
            )
            .order_by(UserMemory.embedding.cosine_distance(query_embedding))
            .limit(top_k)
        )
        result = await self._s.execute(stmt)
        return result.scalars().all()

    async def get_user_memories(
        self, group_id: str, user_id: str, limit: int = 20
    ) -> Sequence[UserMemory]:
        stmt = (
            select(UserMemory)
            .where(UserMemory.group_id == group_id, UserMemory.user_id == user_id)
            .order_by(UserMemory.created_at.desc())
            .limit(limit)
        )
        result = await self._s.execute(stmt)
        return result.scalars().all()

    # =====================================================================
    #  EmotionState
    # =====================================================================

    async def get_or_create_emotion(self, group_id: str, default_mood: str = "neutral") -> EmotionState:
        stmt = select(EmotionState).where(EmotionState.group_id == group_id)
        result = await self._s.execute(stmt)
        state = result.scalar_one_or_none()
        if state is None:
            state = EmotionState(group_id=group_id, mood=default_mood)
            self._s.add(state)
            await self._s.flush()
        return state

    async def update_emotion(
        self,
        group_id: str,
        mood: str,
        valence: float = 0.0,
        arousal: float = 0.0,
    ) -> None:
        stmt = (
            update(EmotionState)
            .where(EmotionState.group_id == group_id)
            .values(mood=mood, valence=valence, arousal=arousal)
        )
        await self._s.execute(stmt)

    # =====================================================================
    #  JargonTerm
    # =====================================================================

    async def upsert_jargon(
        self,
        group_id: str,
        term: str,
        frequency: int = 1,
        meaning: str = "",
        is_custom: bool = False,
    ) -> None:
        """Insert or increment frequency (PostgreSQL upsert)."""
        stmt = pg_insert(JargonTerm).values(
            group_id=group_id,
            term=term,
            frequency=frequency,
            meaning=meaning,
            is_custom=is_custom,
        )
        # On conflict: increment frequency, only update meaning if provided
        update_dict: dict[str, Any] = {
            "frequency": JargonTerm.frequency + frequency,
        }
        if meaning:
            update_dict["meaning"] = meaning
        stmt = stmt.on_conflict_do_update(
            index_elements=["group_id", "term"],
            set_=update_dict,
        )
        await self._s.execute(stmt)

    async def get_group_jargon(
        self, group_id: str, min_frequency: int = 1
    ) -> Sequence[JargonTerm]:
        stmt = (
            select(JargonTerm)
            .where(
                JargonTerm.group_id == group_id,
                JargonTerm.frequency >= min_frequency,
            )
            .order_by(JargonTerm.frequency.desc())
        )
        result = await self._s.execute(stmt)
        return result.scalars().all()

    async def get_jargon_needing_meaning(
        self, group_id: str, min_frequency: int = 5, limit: int = 20
    ) -> Sequence[JargonTerm]:
        """Get terms above frequency threshold that have no meaning yet."""
        stmt = (
            select(JargonTerm)
            .where(
                JargonTerm.group_id == group_id,
                JargonTerm.frequency >= min_frequency,
                JargonTerm.meaning == "",
                JargonTerm.is_custom == False,  # noqa: E712
            )
            .order_by(JargonTerm.frequency.desc())
            .limit(limit)
        )
        result = await self._s.execute(stmt)
        return result.scalars().all()

    async def update_jargon_meaning(self, jargon_id: int, meaning: str) -> None:
        stmt = (
            update(JargonTerm)
            .where(JargonTerm.id == jargon_id)
            .values(meaning=meaning)
        )
        await self._s.execute(stmt)

    async def delete_jargon(self, jargon_id: int) -> None:
        stmt = delete(JargonTerm).where(JargonTerm.id == jargon_id)
        await self._s.execute(stmt)

    async def add_custom_jargon(
        self, group_id: str, term: str, meaning: str
    ) -> int:
        j = JargonTerm(
            group_id=group_id,
            term=term,
            meaning=meaning,
            frequency=0,
            is_custom=True,
        )
        self._s.add(j)
        await self._s.flush()
        return j.id

    # =====================================================================
    #  BotResponse
    # =====================================================================

    async def save_bot_response(
        self,
        group_id: str,
        response_text: str,
        thread_id: int | None = None,
    ) -> int:
        r = BotResponse(
            group_id=group_id,
            thread_id=thread_id,
            response_text=response_text,
        )
        self._s.add(r)
        await self._s.flush()
        return r.id

    # =====================================================================
    #  LearningJob
    # =====================================================================

    async def create_learning_job(
        self, group_id: str, job_type: str
    ) -> int:
        job = LearningJob(group_id=group_id, job_type=job_type, status="pending")
        self._s.add(job)
        await self._s.flush()
        return job.id

    async def complete_learning_job(
        self, job_id: int, result: dict | None = None
    ) -> None:
        stmt = (
            update(LearningJob)
            .where(LearningJob.id == job_id)
            .values(
                status="completed",
                result=result,
                completed_at=_dt.datetime.now(_dt.timezone.utc),
            )
        )
        await self._s.execute(stmt)

    async def fail_learning_job(self, job_id: int, error: str) -> None:
        stmt = (
            update(LearningJob)
            .where(LearningJob.id == job_id)
            .values(status="failed", result={"error": error})
        )
        await self._s.execute(stmt)
