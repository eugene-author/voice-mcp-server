"""FFmpeg and ffprobe through async subprocesses with fixed executables, argument lists and timeouts."""

import asyncio
import json
import shutil
from pathlib import Path

from app.domain.errors import ConfigurationError, MediaError
from app.domain.media import MediaInfo

_ENCODER_ARGS: dict[str, tuple[str, ...]] = {
    "wav": ("-c:a", "pcm_s16le"),
    "mp3": ("-c:a", "libmp3lame", "-q:a", "2"),
    "flac": ("-c:a", "flac"),
    "ogg": ("-c:a", "libvorbis", "-q:a", "5"),
}
LOUDNESS_FILTER = "loudnorm=I=-16:TP=-1.5:LRA=11"
_STDERR_TAIL_CHARS = 400


class FfmpegMediaTool:
    """MediaProbe and MediaConverter backed by the FFmpeg command-line tools."""

    def __init__(
        self, *, ffmpeg_path: Path, ffprobe_path: Path, probe_timeout_seconds: float, convert_timeout_seconds: float
    ) -> None:
        """Store fixed executable paths and timeouts."""
        self._ffmpeg = str(ffmpeg_path)
        self._ffprobe = str(ffprobe_path)
        self._probe_timeout = probe_timeout_seconds
        self._convert_timeout = convert_timeout_seconds

    @classmethod
    def from_path(cls, *, probe_timeout_seconds: float = 30, convert_timeout_seconds: float = 3600) -> FfmpegMediaTool:
        """Locate ffmpeg and ffprobe once on PATH, failing startup if either is missing."""
        ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
        if ffmpeg is None or ffprobe is None:
            raise ConfigurationError("ffmpeg and ffprobe must be installed and on PATH")
        return cls(
            ffmpeg_path=Path(ffmpeg),
            ffprobe_path=Path(ffprobe),
            probe_timeout_seconds=probe_timeout_seconds,
            convert_timeout_seconds=convert_timeout_seconds,
        )

    async def probe(self, path: Path) -> MediaInfo:
        """Return duration, codec and size facts of the first audio stream."""
        output = await self._run(
            [
                self._ffprobe,
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "format=duration,format_name,size:stream=codec_name,sample_rate,channels",
                "-of",
                "json",
                str(path),
            ],
            limit_seconds=self._probe_timeout,
        )
        try:
            data = json.loads(output)
            fmt = data["format"]
            streams = data.get("streams") or []
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise MediaError("could not read media information") from exc
        if not streams:
            raise MediaError("the file contains no audio stream")
        stream = streams[0]
        duration = float(fmt.get("duration") or 0.0)
        if duration <= 0:
            raise MediaError("the audio has no measurable duration")
        return MediaInfo(
            duration_seconds=duration,
            format_name=str(fmt.get("format_name", "")),
            codec=stream.get("codec_name"),
            sample_rate=int(stream["sample_rate"]) if stream.get("sample_rate") else None,
            channels=int(stream["channels"]) if stream.get("channels") else None,
            size_bytes=int(fmt.get("size") or 0),
        )

    async def to_wav(self, source: Path, destination: Path, *, sample_rate: int, channels: int = 1) -> Path:
        """Decode any supported input to 16-bit PCM WAV at the given rate and channel count."""
        await self._convert(
            ["-i", str(source), "-vn", "-ac", str(channels), "-ar", str(sample_rate), "-c:a", "pcm_s16le"], destination
        )
        return destination

    async def encode(self, source_wav: Path, destination: Path, *, output_format: str) -> Path:
        """Encode a WAV file to wav, mp3, flac or ogg."""
        encoder = _ENCODER_ARGS.get(output_format)
        if encoder is None:
            raise MediaError(f"unsupported output format {output_format!r}")
        await self._convert(["-i", str(source_wav), "-vn", *encoder, "-f", output_format], destination)
        return destination

    async def normalize_loudness(self, source_wav: Path, destination: Path) -> Path:
        """Write an EBU R128 loudness-normalized, true-peak-limited copy at the source sample rate."""
        info = await self.probe(source_wav)
        rate = str(info.sample_rate or 24000)
        await self._convert(
            ["-i", str(source_wav), "-af", LOUDNESS_FILTER, "-ar", rate, "-c:a", "pcm_s16le", "-f", "wav"], destination
        )
        return destination

    async def _convert(self, arguments: list[str], destination: Path) -> None:
        """Run ffmpeg writing to destination, an existing directory, removing partial output on failure."""
        command = [self._ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", *arguments, str(destination)]
        try:
            await self._run(command, limit_seconds=self._convert_timeout)
        except BaseException:
            await asyncio.to_thread(destination.unlink, missing_ok=True)
            raise

    @staticmethod
    async def _run(command: list[str], *, limit_seconds: float) -> str:
        """Run a command, returning stdout, or raise MediaError on failure or timeout."""
        process = await asyncio.create_subprocess_exec(
            *command, stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=limit_seconds)
        except TimeoutError as exc:
            process.kill()
            await process.wait()
            raise MediaError(f"{Path(command[0]).name} timed out after {limit_seconds:.0f}s") from exc
        except asyncio.CancelledError:
            process.kill()
            await process.wait()
            raise
        if process.returncode != 0:
            detail = stderr.decode(errors="replace").strip()[-_STDERR_TAIL_CHARS:]
            raise MediaError(f"{Path(command[0]).name} failed: {detail or 'unknown error'}")
        return stdout.decode(errors="replace")
