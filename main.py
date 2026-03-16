"""
astrbot_plugin_sb_qunyou — 群聊智能体插件

主插件类，负责：
  - 消息监听 (on_message) — 事件驱动防抖
  - 防抖 → 存储 → 话题路由 → 后台学习触发
  - LLM 请求注入 (on_llm_request)
"""
from __future__ import annotations

import asyncio
from typing import Any, Optional

from astrbot.api.event import AstrMessageEvent
from astrbot.api.event import filter
from astrbot.api.event.filter import PermissionType
import astrbot.api.star as star
from astrbot.api.star import Context
from astrbot.api import logger, AstrBotConfig

from .config import PluginConfig
from .lifecycle import Lifecycle


class QunyouPlugin(star.Star):
    """群聊智能体 · 千群千面"""

    def __init__(self, context: Context, config: AstrBotConfig = None) -> None:
        super().__init__(context)
        self.context = context

        # Pre-initialize all attributes to avoid AttributeError
        self.db = None
        self.llm = None
        self.group_persona = None
        self.speaker_memory = None
        self.emotion = None
        self.jargon = None
        self.hook_handler = None
        self.debounce = None
        self.topic_router = None
        self.knowledge = None      # LightRAGKnowledgeManager (lazy-init)
        self.reranker = None       # IRerankProvider (lazy-init)
        self.background_tasks: set[asyncio.Task] = set()
        self._ingestion_buffer: dict[str, list[str]] = {}  # knowledge ingestion

        # Load config
        raw = config if isinstance(config, dict) else (config or {})
        try:
            self.plugin_config = PluginConfig.from_astrbot_config(raw)
        except Exception as e:
            logger.warning(f"[Qunyou] Config parse error, using defaults: {e}")
            self.plugin_config = PluginConfig()

        # Lifecycle orchestration
        self._lifecycle = Lifecycle(self)
        try:
            self._lifecycle.bootstrap(self.plugin_config, context)
        except Exception as e:
            logger.error(f"[Qunyou] Bootstrap failed: {e}", exc_info=True)

        logger.info("[Qunyou] Plugin initialized")

    # ================================================================== #
    #  Lifecycle
    # ================================================================== #

    async def initialize(self):
        """Called by AstrBot after handler binding."""
        await self._lifecycle.on_load()

    async def terminate(self):
        """Called on plugin unload/disable."""
        # Flush debounce buffer
        if self.debounce:
            await self.debounce.flush_all()
        # Flush jargon counters
        if self.jargon and self.db:
            try:
                await self.jargon.flush_all(self.db)
            except Exception:
                pass
        # Flush knowledge ingestion buffer
        if self._ingestion_buffer and self.knowledge:
            for gid, texts in self._ingestion_buffer.items():
                for txt in texts:
                    try:
                        await self.knowledge.insert(gid, txt)
                    except Exception:
                        pass
            self._ingestion_buffer.clear()
        await self._lifecycle.shutdown()

    # ================================================================== #
    #  Message listener — event-driven debounce
    # ================================================================== #

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_message(self, event: AstrMessageEvent):
        """Listen to all messages → debounce (event-driven) → process.

        Event-driven debounce pattern (from continuous_message):
          Msg1 → create session, block (await flush_event.wait())
          Msg2 → append, event.stop_event(), reset timer
          Timer → flush_event.set() → Msg1 handler wakes up
          → event rewritten with merged text → continues to LLM
        """
        if not self.db or not self.db.engine:
            return

        if self.debounce and self.plugin_config.debounce.mode != "off":
            # Event-driven debounce: may block on first msg, stop subsequent
            result = await self.debounce.handle_event(event)
            if not result:
                return  # event was intercepted or empty

            # After debounce settlement, event.message_str is updated
            # with merged text. Continue to process.
            text = event.get_message_str()
        else:
            text = event.get_message_str()

        if not text or not text.strip():
            return

        group_id = event.get_group_id() or event.get_sender_id()
        sender_id = event.get_sender_id()

        # Fire background processing (non-blocking)
        self._fire_and_forget(
            self._process_message(group_id, sender_id, text, event)
        )

    async def _process_message(
        self,
        group_id: str,
        sender_id: str,
        text: str,
        event: Optional[AstrMessageEvent] = None,
    ) -> None:
        """Core message processing: store → route → background tasks."""
        if not self.db:
            return

        try:
            sender_name = ""
            platform = ""
            if event:
                sender_name = event.get_sender_name() or ""
                platform = getattr(event, "platform_name", "") or ""

            # 1. Save raw message
            async with self.db.session() as session:
                from .db.repo import Repository
                repo = Repository(session)
                msg_id = await repo.save_raw_message(
                    group_id=group_id,
                    sender_id=sender_id,
                    sender_name=sender_name,
                    text_content=text,
                    platform=platform,
                )
                await session.commit()

            # 2. Topic routing (async, non-blocking)
            if self.topic_router:
                self._fire_and_forget(
                    self.topic_router.route_message(group_id, msg_id, text, self.db)
                )

            # 3. Jargon word counting (in-memory, fast)
            if self.jargon:
                self.jargon.count_words(group_id, text)

            # 4. Group persona: check if batch threshold reached
            if self.group_persona:
                self._fire_and_forget(
                    self._check_persona_learning(group_id)
                )

            # 5. Speaker memory extraction (async)
            if self.speaker_memory:
                self._fire_and_forget(
                    self.speaker_memory.extract_and_store(
                        group_id, sender_id, sender_name, text, self.db
                    )
                )

            # 6. Emotion update (async, probabilistic)
            if self.emotion and event and event.is_at_or_wake_command:
                self._fire_and_forget(
                    self.emotion.maybe_update(group_id, text, self.db)
                )

            # 7. Knowledge ingestion buffer (LightRAG)
            if (
                self.knowledge
                and len(text.strip()) >= self.plugin_config.knowledge.min_ingestion_length
            ):
                buf = self._ingestion_buffer.setdefault(group_id, [])
                buf.append(text)
                if len(buf) >= self.plugin_config.knowledge.ingestion_buffer_max:
                    self._fire_and_forget(
                        self._flush_knowledge_buffer(group_id)
                    )

        except Exception as e:
            logger.error(f"[Qunyou] Message processing error: {e}", exc_info=True)

    async def _check_persona_learning(self, group_id: str) -> None:
        """Check if batch learning threshold is reached."""
        if self.group_persona and self.db:
            should_learn = await self.group_persona.increment_and_check(
                group_id, self.db
            )
            if should_learn:
                self._fire_and_forget(
                    self.group_persona.run_batch_learning(group_id, self.db)
                )

    # ================================================================== #
    #  LLM Hook
    # ================================================================== #

    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req=None):
        """Inject context into LLM requests."""
        if self.hook_handler:
            await self.hook_handler.handle(event, req)

    # ================================================================== #
    #  Commands
    # ================================================================== #

    @filter.command("qunyou_status")
    @filter.permission_type(PermissionType.ADMIN)
    async def status_command(self, event: AstrMessageEvent):
        """查看群聊智能体状态"""
        group_id = event.get_group_id() or event.get_sender_id()
        parts = [f"🤖 群聊智能体状态 [{group_id}]"]

        if self.db:
            parts.append("✅ 数据库已连接")
        else:
            parts.append("❌ 数据库未连接")

        if self.emotion and self.db:
            try:
                mood = await self.emotion.get_mood(group_id, self.db)
                parts.append(f"😊 当前情绪: {mood}")
            except Exception:
                parts.append("😊 情绪: 获取失败")

        mode = self.plugin_config.debounce.mode
        parts.append(f"🔧 防抖模式: {mode}")
        if mode != "off":
            parts.append(f"   ⏱ 窗口: {self.plugin_config.debounce.time_window_seconds}s")
            parts.append(f"   ✍ 输入感知: {'开' if self.plugin_config.debounce.enable_typing_detection else '关'}")
        parts.append(f"📋 话题路由: {'开' if self.plugin_config.topic.enabled else '关'}")
        parts.append(f"📝 黑话统计: {'开' if self.plugin_config.jargon.enabled else '关'}")
        parts.append(
            f"📚 知识引擎: {self.plugin_config.knowledge.engine}"
            f" ({'ready' if self.knowledge else 'off'})"
        )
        parts.append(
            f"🔀 Reranker: {'开' if self.reranker else '关'}"
        )

        # Cache stats
        try:
            from .utils.cache import get_cache_manager
            cm = get_cache_manager()
            stats = cm.get_stats()
            if stats:
                hit_info = ", ".join(
                    f"{k}:{v['hit_rate']:.0%}" for k, v in stats.items()
                )
                parts.append(f"💾 缓存命中率: {hit_info}")
        except Exception:
            pass

        yield event.plain_result("\n".join(parts))

    @filter.command("set_mood")
    @filter.permission_type(PermissionType.ADMIN)
    async def set_mood_command(self, event: AstrMessageEvent):
        """设置情绪: /set_mood happy"""
        text = event.message_str.strip()
        parts = text.split(maxsplit=1)
        mood = parts[0] if parts else ""

        if not mood:
            yield event.plain_result("用法: /set_mood happy|neutral|angry|sad|excited|bored")
            return

        if self.emotion and self.db:
            group_id = event.get_group_id() or event.get_sender_id()
            await self.emotion.set_mood(group_id, mood, self.db)
            yield event.plain_result(f"情绪已设置为: {mood}")
        else:
            yield event.plain_result("情绪引擎未启用")

    # ================================================================== #
    #  Utility
    # ================================================================== #

    def _fire_and_forget(self, coro) -> None:
        """Launch a background task with proper tracking."""
        task = asyncio.create_task(coro)
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)

    async def _flush_knowledge_buffer(self, group_id: str) -> None:
        """Flush buffered messages for a group through LightRAG."""
        texts = self._ingestion_buffer.pop(group_id, [])
        if not texts or not self.knowledge:
            return
        combined = "\n\n".join(texts)
        try:
            await self.knowledge.insert(group_id, combined)
            logger.debug(
                f"[Qunyou] Knowledge ingestion: {group_id}, "
                f"{len(texts)} messages"
            )
        except Exception as e:
            logger.debug(f"[Qunyou] Knowledge ingestion failed: {e}")
