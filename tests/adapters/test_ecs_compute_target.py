"""Amazon ECS compute target over a stubbed boto3 client: state mapping and desired-count updates."""

import pytest
from botocore.exceptions import ClientError

from app.adapters.compute.aws_ecs.target import EcsComputeTarget
from app.domain.errors import BackendUnavailableError
from app.domain.workers import TargetState
from tests.mocks.compute import MockEcsClient


def _target(client: MockEcsClient) -> EcsComputeTarget:
    """Return a target over the stub."""
    return EcsComputeTarget(region="eu-central-1", cluster="voice", service="gpu-worker", client=client)


@pytest.mark.parametrize(
    ("desired", "running", "pending", "state"),
    [
        (0, 0, 0, TargetState.STOPPED),
        (0, 1, 0, TargetState.STOPPING),
        (0, 0, 1, TargetState.STOPPING),
        (1, 0, 0, TargetState.STARTING),
        (1, 0, 1, TargetState.STARTING),
        (1, 1, 0, TargetState.RUNNING),
        (1, 1, 1, TargetState.RUNNING),
    ],
)
async def test_counts_map_to_target_states(desired: int, running: int, pending: int, state: TargetState) -> None:
    """Map desired, running and pending counts to the documented target state."""
    target = _target(MockEcsClient(desired=desired, running=running, pending=pending))
    assert await target.status() is state


async def test_errors_and_missing_service_read_as_unknown() -> None:
    """Report unknown when the API fails or the service does not exist."""
    failing = MockEcsClient()
    failing.error = ClientError({"Error": {"Code": "AccessDeniedException", "Message": "no"}}, "DescribeServices")
    missing = MockEcsClient()
    missing.missing = True
    assert await _target(failing).status() is TargetState.UNKNOWN
    assert await _target(missing).status() is TargetState.UNKNOWN


async def test_start_and_stop_set_the_desired_count() -> None:
    """Scale the one service to 1 and back to 0; force is the same call."""
    client = MockEcsClient()
    target = _target(client)
    await target.start()
    await target.stop(force=False)
    await target.stop(force=True)
    assert [update["desiredCount"] for update in client.updates] == [1, 0, 0]
    assert {(u["cluster"], u["service"]) for u in client.updates} == {("voice", "gpu-worker")}


async def test_refused_update_raises() -> None:
    """Raise a retryable error when update_service is refused."""
    client = MockEcsClient()
    client.error = ClientError({"Error": {"Code": "AccessDeniedException", "Message": "no"}}, "UpdateService")
    with pytest.raises(BackendUnavailableError, match="update_service"):
        await _target(client).start()
