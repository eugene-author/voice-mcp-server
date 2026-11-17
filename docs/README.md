# Voice MCP Server: flow diagrams

Mermaid diagrams of how a request travels through the code, from an MCP tool call to a transcript or an audio file.
They render natively on GitHub and in PyCharm's Markdown preview.

| Document | What it shows |
|---|---|
| [architecture.md](architecture.md) | Components, layers and what the composition root wires together |
| [speech-to-text.md](speech-to-text.md) | `speech_to_text`: MCP call, job submission, worker, STT pipeline, published transcripts |
| [text-to-speech.md](text-to-speech.md) | `text_to_speech`: MCP call, voice resolution, chunked synthesis, mastering, published audio |
| [voice-registration.md](voice-registration.md) | `register_voice`: how a cloned voice becomes usable by `text_to_speech` |
| [job-transitions.md](job-transitions.md) | Job state machine, worker loop, retries and crash recovery |
| [cli.md](cli.md) | Every `python -m app` command and what it touches |
| [gpu-worker.md](gpu-worker.md) | Process roles, the separate GPU worker, on-demand start, drain and audio across machines |

All job-based tools (`speech_to_text`, `text_to_speech`, `register_voice`, `transcribe_batch`) share one shape:

1. The tool handler validates input and **commits** queued job rows.
2. Only after the commit is the worker woken up.
3. The call waits up to `jobs.inline_wait_seconds`, then returns the job envelope. If the job is still running,
   the client polls `get_job`.

Inference doesn't run inside a database transaction or on the event loop. It runs on the single inference thread
through `ModelHost`.
