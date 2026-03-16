# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AstrBot plugin for group chat intelligence ("群聊智能体"). Provides per-group adaptive persona, topic routing, message debouncing, speaker memory, emotion engine, jargon tracking, and persona binding with tone learning. Runs on AstrBot 4.11.4+ with PostgreSQL + pgvector.

## Development Commands

```bash
# Install dependencies
pip install -r requirements.txt
pip install pytest pytest-asyncio  # test dependencies

# Run all tests
pytest tests/ -v

# Run a single test file
pytest tests/test_debounce.py -v

# Run a single test class or method
pytest tests/test_debounce.py::TestIsCommand -v
pytest tests/test_debounce.py::TestIsCommand::test_basic_command -v
```

No build step required — this is a pure Python AstrBot plugin loaded at runtime.

## Architecture

### Plugin Lifecycle (3 phases)

1. **Bootstrap** (`lifecycle.py:bootstrap`) — Synchronous, called from `__init__`. Creates all service instances, no I/O.
2. **on_load** (`lifecycle.py:on_load`) — Async. Starts DB, resolves lazy providers (Reranker, LightRAG), starts WebUI.
3. **shutdown** (`lifecycle.py:shutdown`) — Ordered teardown: LightRAG → WebUI → background tasks → DB.

Lazy provider resolution (Issue #78): Reranker and LightRAG use a background retry loop because AstrBot's `provider_manager.inst_map` isn't populated until all plugins are loaded.

### Message Flow

```
on_message → DebounceManager.handle_event → _process_message
                                              ├── repo.save_raw_message
                                              ├── TopicThreadRouter.route_message  (async)
                                              ├── JargonService.count_words        (in-memory)
                                              ├── GroupPersonaService.increment     (async)
                                              ├── SpeakerMemory.extract_and_store   (async)
                                              ├── EmotionEngine.maybe_update        (async, probabilistic)
                                              ├── PersonaBinding.increment_and_check(async)
                                              └── Knowledge ingestion buffer        (async)
```

All async operations after `save_raw_message` are fire-and-forget via `_fire_and_forget()`.

### LLM Hook Injection (on_llm_request)

`HookHandler.handle` gathers 7 context sources concurrently, then injects into the LLM request:

| Priority | Source | Target |
|----------|--------|--------|
| HIGHEST | Persona binding (bound persona + learned tone) | `system_prompt` |
| HIGH | Group persona (base + learned) | `system_prompt` |
| HIGH | Emotion state (if non-neutral) | `system_prompt` |
| MEDIUM | Thread context (topic + recent messages) | `extra_user_content_parts` |
| MEDIUM | User memories (pgvector kNN) | `extra_user_content_parts` |
| MEDIUM | Knowledge graph (LightRAG) | `extra_user_content_parts` |
| LOW | Jargon hints | `extra_user_content_parts` |

Extra parts are optionally reranked before injection.

### Key Abstractions

- **`LLMAdapter`** (`services/llm_adapter.py`) — Bridges AstrBot's Provider API. All LLM/embedding calls go through this. Exposes `fast_chat()`, `main_chat()`, `get_embedding()`. Never access framework providers directly.
- **`Repository`** (`db/repo.py`) — Single unified CRUD class for all 11 tables. All DB access goes through this. Uses `async with db.session() as session` pattern.
- **`Database`** (`db/engine.py`) — SQLAlchemy async engine wrapper. Auto-creates pgvector extension and all tables on `start()`.
- **`CacheManager`** (`utils/cache.py`) — Global singleton with TTL/LRU caches for context, embedding, emotion, knowledge. Uses `cachetools` with fallback to plain dicts.

### Event-Driven Debounce (`pipeline/debounce.py`)

Adapted from `astrbot_plugin_continuous_message`. Uses `asyncio.Event` + `Task.cancel()`:
- Msg1 → create session, block on `flush_event.wait()`
- Msg2+ → append to buffer, `event.stop_event()`, reset timer
- Timer fires → `flush_event.set()` → Msg1 wakes, reconstructs event with merged text

Three modes: `off`, `time` (L1 time window only), `time_bert` (L1 + L2 semantic completeness via LLM).

### Topic Router (`pipeline/topic_router.py`)

Embedding cosine similarity + sliding centroid average. Messages are routed to existing threads or spawn new ones. Stale threads are auto-archived based on TTL.

### Configuration

- **`config.py`** — Pydantic v2 models. `PluginConfig` is the root, with sub-configs for each module.
- **`_conf_schema.json`** — AstrBot config panel schema. Maps flat groups (e.g., `Debounce_Settings`) to nested config.
- **`PluginConfig.from_astrbot_config(raw)`** — Converts AstrBot's flat config dict into nested Pydantic models.

### Database Models (`db/models.py`)

11 tables: `raw_messages`, `group_profiles`, `active_threads`, `thread_messages`, `user_memories`, `emotion_states`, `jargon_terms`, `bot_responses`, `learning_jobs`, `group_persona_bindings`, `persona_tone_versions`. All vector columns use pgvector's `Vector(None)` (dimension set at runtime).

### Prompt Templates (`prompts/templates.py`)

All LLM prompts are centralized here. Injection templates use XML-style tags (`<group_persona>`, `<emotion_state>`, etc.) with trust levels.

## Code Conventions

- All imports from AstrBot use `astrbot.api` namespace (`logger`, `AstrBotConfig`, `star`, `event`).
- Plugin entry point: `QunyouPlugin(star.Star)` in `main.py`, exported via `__init__.py`.
- Services use `TYPE_CHECKING` guards for circular import avoidance.
- Repository methods follow get-or-create pattern (`get_or_create_group_profile`, etc.).
- Background tasks tracked via `self.background_tasks: set[asyncio.Task]` with `add_done_callback(discard)`.
- Log prefix convention: `[ModuleName]` (e.g., `[Qunyou]`, `[Lifecycle]`, `[Emotion]`, `[TopicRouter]`).
- Tests mock AstrBot internals; only pure-logic subcomponents are unit-testable without the framework.
