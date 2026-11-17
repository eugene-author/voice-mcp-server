"""Response builders producing the documented minimal shapes."""

from typing import Any

from app.domain.jobs import JobStatus, JobType, JobView

_ACTIVE = (JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.RETRY_WAIT)
_FINISHED = (JobStatus.COMPLETED, JobStatus.COMPLETED_WITH_ERRORS)


def job_envelope(view: JobView, *, download_urls: dict[str, str] | None = None) -> dict[str, Any]:
    """Return the job envelope for get_job and job-submitting tools."""
    envelope: dict[str, Any] = {"job_id": str(view.id), "type": str(view.type), "status": str(view.status)}
    if view.status in _ACTIVE:
        envelope["progress"] = view.progress
    if view.status is JobStatus.RUNNING and view.phase is not None:
        envelope["phase"] = str(view.phase)
    if view.files and (view.status in _FINISHED or view.type is JobType.TRANSCRIBE_BATCH):
        envelope["files"] = view.files
    if view.status in _FINISHED:
        text = view.result.get("text")
        if text:
            envelope["text"] = text
        if view.warnings:
            envelope["warnings"] = list(view.warnings)
    if view.error is not None and view.status in (JobStatus.FAILED, JobStatus.RETRY_WAIT):
        envelope["error"] = {"code": view.error.code, "message": view.error.message}
    if view.counts is not None:
        envelope["counts"] = view.counts.to_dict()
    if view.voice_id is not None:
        envelope["voice_id"] = view.voice_id
    if view.parent_job_id is not None:
        envelope["parent_job_id"] = str(view.parent_job_id)
    if download_urls:
        envelope["download_urls"] = download_urls
    return envelope


def job_summary(view: JobView) -> dict[str, Any]:
    """Return the compact list_jobs item: no transcripts, files only for finished jobs."""
    summary: dict[str, Any] = {
        "job_id": str(view.id),
        "type": str(view.type),
        "status": str(view.status),
        "created_at": view.created_at.isoformat().replace("+00:00", "Z"),
    }
    if view.files:
        summary["files"] = view.files
    if view.counts is not None:
        summary["counts"] = view.counts.to_dict()
    if view.error is not None and view.status is JobStatus.FAILED:
        summary["error"] = {"code": view.error.code, "message": view.error.message}
    if view.parent_job_id is not None:
        summary["parent_job_id"] = str(view.parent_job_id)
    return summary
