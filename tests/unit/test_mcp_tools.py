"""MCP tools through the in-process client, backed by the real services over port mocks."""

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from mcp import Client

from app.domain.assets import StorageBackend
from app.mcp_server.server import build_mcp_server
from tests.conftest import write_tone
from tests.harness import OTHER, OWNER, Harness, build_harness
from tests.mocks.storage import MockAssetStore

EXPECTED_TOOLS = {
    "speech_to_text",
    "text_to_speech",
    "transcribe_batch",
    "list_models",
    "list_voices",
    "register_voice",
    "get_job",
    "list_jobs",
    "health_check",
}


@contextlib.asynccontextmanager
async def running(harness: Harness) -> AsyncIterator[None]:
    """Run the worker in the background for the duration of the block."""
    task = asyncio.create_task(harness.worker.run())
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


async def call(harness: Harness, name: str, arguments: dict[str, Any], *, caller: Any = OWNER) -> tuple[bool, Any]:
    """Call one tool and return (is_error, structured payload)."""
    server = build_mcp_server(harness.settings, harness.tool_services(caller), None)
    async with Client(server) as client:
        result = await client.call_tool(name, arguments)
    payload = result.structured_content
    if payload is None and result.content:
        payload = json.loads(result.content[0].text)
    return result.is_error, payload


async def test_exactly_nine_tools_with_schemas(harness: Harness) -> None:
    """Expose exactly the nine documented tools with object input schemas."""
    server = build_mcp_server(harness.settings, harness.tool_services(), None)
    async with Client(server) as client:
        tools = (await client.list_tools()).tools
    assert {tool.name for tool in tools} == EXPECTED_TOOLS
    stt = next(tool for tool in tools if tool.name == "speech_to_text")
    assert stt.input_schema["required"] == ["file_path"]
    register = next(tool for tool in tools if tool.name == "register_voice")
    assert set(register.input_schema["required"]) == {"name", "reference_audio_path", "consent_confirmed"}


async def test_nine_tools_registered_without_tts(tmp_path: Path) -> None:
    """Register all nine tools even when no TTS runtime is installed."""
    harness = build_harness(tmp_path, tts=None)
    server = build_mcp_server(harness.settings, harness.tool_services(), None)
    async with Client(server) as client:
        assert {tool.name for tool in (await client.list_tools()).tools} == EXPECTED_TOOLS
    is_error, payload = await call(harness, "text_to_speech", {"text": "hi", "voice_id": "preset:vivian"})
    assert is_error
    assert payload["error"]["code"] == "capability_unavailable"
    assert "'tts' extra" in payload["error"]["message"]
    is_error, payload = await call(harness, "list_voices", {})
    assert is_error
    assert payload["error"]["code"] == "capability_unavailable"
    harness.executor.shutdown()


async def test_speech_to_text_waits_and_returns_minimal_envelope(harness: Harness) -> None:
    """Return the completed envelope with files and text, omitting inapplicable fields."""
    path = write_tone(harness.inputs / "a.wav")
    async with running(harness):
        is_error, payload = await call(harness, "speech_to_text", {"file_path": str(path)})
    assert not is_error
    assert payload["status"] == "completed"
    assert payload["text"] == "hello world"
    assert set(payload["files"]) == {"txt", "json", "srt", "vtt"}
    assert "progress" not in payload
    assert "error" not in payload
    assert "phase" not in payload


async def test_speech_to_text_without_wait_returns_queued(harness: Harness) -> None:
    """Return immediately with the queued status when wait is false."""
    path = write_tone(harness.inputs / "a.wav")
    is_error, payload = await call(harness, "speech_to_text", {"file_path": str(path), "wait": False})
    assert not is_error
    assert payload["status"] == "queued"
    assert payload["progress"] == 0


@pytest.mark.parametrize(
    ("arguments", "code"),
    [
        ({"file_path": "/etc/passwd"}, "invalid_request"),
        ({"file_path": "/nonexistent/x.wav"}, "invalid_request"),
        ({"file_path": "{inputs}/a.wav", "model": "medium"}, "model_not_installed"),
        ({"file_path": "{inputs}/a.wav", "preset": "ultra"}, "invalid_request"),
        ({"file_path": "{inputs}/a.wav", "options": {"beam_size": 99}}, None),
    ],
)
async def test_speech_to_text_errors_are_structured(
    harness: Harness, arguments: dict[str, Any], code: str | None
) -> None:
    """Return {error: {code, message}} without tracebacks; schema violations are rejected by the SDK."""
    write_tone(harness.inputs / "a.wav")
    resolved = {k: v.format(inputs=harness.inputs) if isinstance(v, str) else v for k, v in arguments.items()}
    server = build_mcp_server(harness.settings, harness.tool_services(), None)
    async with Client(server) as client:
        result = await client.call_tool("speech_to_text", resolved)
    assert result.is_error
    text = result.content[0].text
    assert "Traceback" not in text
    if code is not None:
        assert result.structured_content == json.loads(text)
        assert json.loads(text)["error"]["code"] == code


async def test_get_job_and_list_jobs_are_owner_scoped(harness: Harness) -> None:
    """Hide jobs from other callers and return not_found for them."""
    path = write_tone(harness.inputs / "a.wav")
    _, submitted = await call(harness, "speech_to_text", {"file_path": str(path), "wait": False})
    is_error, payload = await call(harness, "get_job", {"job_id": submitted["job_id"]}, caller=OTHER)
    assert is_error
    assert payload["error"]["code"] == "not_found"
    is_error, payload = await call(harness, "get_job", {"job_id": "not-a-uuid"})
    assert is_error
    assert payload["error"]["code"] == "invalid_request"
    _, own = await call(harness, "list_jobs", {})
    _, other = await call(harness, "list_jobs", {}, caller=OTHER)
    assert [item["job_id"] for item in own["items"]] == [submitted["job_id"]]
    assert other["items"] == []
    assert own["next_cursor"] is None
    assert "files" not in own["items"][0]


async def test_get_job_download_urls_only_in_s3_mode(harness: Harness) -> None:
    """Reject include_download_urls in local mode."""
    path = write_tone(harness.inputs / "a.wav")
    _, submitted = await call(harness, "speech_to_text", {"file_path": str(path), "wait": False})
    is_error, payload = await call(harness, "get_job", {"job_id": submitted["job_id"], "include_download_urls": True})
    assert is_error
    assert payload["error"]["code"] == "invalid_request"


async def test_list_models_and_health_check(harness: Harness) -> None:
    """Report installed models and health without touching inference."""
    _, models = await call(harness, "list_models", {"kind": "stt"})
    small = next(m for m in models["models"] if m["name"] == "small" and m["backend"] == "ctranslate2")
    assert small["installed"]
    assert small["active"]
    _, health = await call(harness, "health_check", {})
    assert health["status"] == "ok"
    assert health["stt"]["backend"] == "ctranslate2"
    assert harness.stt.load_count == 0


async def test_health_check_degraded_when_dependency_fails(harness: Harness) -> None:
    """Report degraded status when a dependency check fails."""
    harness.store.healthy = False
    _, health = await call(harness, "health_check", {})
    assert health["status"] == "degraded"
    assert health["dependencies"]["storage"].startswith("error")


async def test_download_urls_only_on_request_in_s3_mode(tmp_path: Path) -> None:
    """Return short-lived URLs only when asked, only to the owner, and not in list_jobs."""
    harness = build_harness(tmp_path, store=MockAssetStore(backend=StorageBackend.S3))
    path = write_tone(harness.inputs / "a.wav")
    async with running(harness):
        _, done = await call(harness, "speech_to_text", {"file_path": str(path)})
    assert done["files"]["txt"].startswith("s3://")
    _, plain = await call(harness, "get_job", {"job_id": done["job_id"]})
    assert "download_urls" not in plain
    _, signed = await call(harness, "get_job", {"job_id": done["job_id"], "include_download_urls": True})
    assert set(signed["download_urls"]) == set(done["files"])
    assert "ttl=300" in signed["download_urls"]["txt"]
    is_error, _ = await call(
        harness, "get_job", {"job_id": done["job_id"], "include_download_urls": True}, caller=OTHER
    )
    assert is_error
    _, listed = await call(harness, "list_jobs", {})
    assert "download_urls" not in json.dumps(listed)
    harness.executor.shutdown()
