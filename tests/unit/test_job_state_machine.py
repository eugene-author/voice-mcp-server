"""Job state machine transitions."""

import itertools

import pytest

from app.domain.jobs import BatchCounts, JobStatus

S = JobStatus
ALLOWED = {
    (S.QUEUED, S.RUNNING),
    (S.QUEUED, S.CANCELLED),
    (S.RUNNING, S.COMPLETED),
    (S.RUNNING, S.COMPLETED_WITH_ERRORS),
    (S.RUNNING, S.FAILED),
    (S.RUNNING, S.RETRY_WAIT),
    (S.RUNNING, S.INTERRUPTED),
    (S.RETRY_WAIT, S.QUEUED),
    (S.RETRY_WAIT, S.CANCELLED),
    (S.INTERRUPTED, S.QUEUED),
    (S.INTERRUPTED, S.FAILED),
    (S.FAILED, S.QUEUED),
}


@pytest.mark.parametrize(("source", "target"), list(itertools.product(JobStatus, JobStatus)))
def test_transition_table_is_exhaustive(source: JobStatus, target: JobStatus) -> None:
    """Allow exactly the documented transitions."""
    assert source.can_transition_to(target) is ((source, target) in ALLOWED)


def test_terminal_statuses() -> None:
    """Treat completed, completed_with_errors, failed and cancelled as terminal."""
    assert {s for s in JobStatus if s.is_terminal} == {S.COMPLETED, S.COMPLETED_WITH_ERRORS, S.FAILED, S.CANCELLED}


@pytest.mark.parametrize(
    ("counts", "expected"),
    [
        (BatchCounts(3, 3, 0, 0), S.COMPLETED),
        (BatchCounts(3, 2, 1, 0), S.COMPLETED_WITH_ERRORS),
        (BatchCounts(3, 0, 3, 0), S.FAILED),
        (BatchCounts(3, 1, 1, 1), None),
    ],
)
def test_batch_final_status(counts: BatchCounts, expected: JobStatus | None) -> None:
    """Derive the parent status only once every child finished."""
    assert counts.final_status() is expected
