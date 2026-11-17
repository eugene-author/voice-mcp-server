"""Builds the configured compute target, importing its adapter only when it is selected."""

from app.core.settings import Settings
from app.domain.errors import ConfigurationError
from app.ports.compute import ComputeTarget


def build_compute_target(settings: Settings) -> ComputeTarget:
    """Return the docker or aws_ecs target from gpu_worker.on_demand; called only when a target is configured."""
    on_demand = settings.gpu_worker.on_demand
    match on_demand.target:
        case "docker":
            from app.adapters.compute.docker.target import DockerComputeTarget

            docker = on_demand.docker
            if docker.container is None:
                raise ConfigurationError("gpu_worker.on_demand.docker.container is required")
            return DockerComputeTarget(
                endpoint=docker.endpoint,
                container=docker.container,
                stop_timeout_seconds=round(settings.gpu_worker.drain_timeout_seconds),
                request_timeout_seconds=docker.request_timeout_seconds,
            )
        case "aws_ecs":
            from app.adapters.compute.aws_ecs.target import EcsComputeTarget

            ecs = on_demand.aws_ecs
            if not (ecs.region and ecs.cluster and ecs.service):
                raise ConfigurationError("gpu_worker.on_demand.aws_ecs needs region, cluster and service")
            return EcsComputeTarget(region=ecs.region, cluster=ecs.cluster, service=ecs.service)
        case _:
            raise ConfigurationError("gpu_worker.on_demand.target is none; set docker or aws_ecs")
