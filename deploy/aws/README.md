# Deploying on AWS

Target shape: one GPU EC2 instance (or ECS on EC2 with the ECS GPU-optimized AMI) running the `cuda` image,
Amazon RDS for PostgreSQL 16, a private S3 bucket for assets, an ALB with TLS in front of the MCP endpoint,
and Keycloak (or another verified OIDC provider) for tokens. Section 8 describes the alternative shape: an API on
CPU and a GPU worker that starts on demand.

## 1. Network and TLS

- Put the instance in private subnets; only the ALB is public.
- ALB listener on 443 with an ACM certificate, forwarding to the container on 8000.
- The MCP resource URL becomes `https://voice.example.com/mcp`. Set it in three places:
  `VOICE_MCP_AUTH__AUDIENCE`, `VOICE_MCP_AUTH__RESOURCE_SERVER_URL`, and the Keycloak audience mapper
  (`included.custom.audience` in `deploy/keycloak/realm-voice.json`).
- Do not expose port 8000 directly. The container listens on `0.0.0.0`, so the server refuses to start until
  `VOICE_MCP_SERVER__ALLOWED_HOSTS` lists the Host headers it accepts. Behind the ALB that is the public name
  (`["voice.example.com"]`). The ALB forwards the client's Host header unchanged, and requests with any other
  Host get `421`. If ALB health checks hit the MCP port, target the ALB check at a path that does not need
  `/mcp`, or add the target's private address to the list.
- Set `VOICE_MCP_SERVER__ALLOWED_ORIGINS` only for browser-based clients (for example
  `["https://voice.example.com"]`). Agents that send no `Origin` header are always accepted, while a browser
  origin that is not listed gets `403`. Together this is the DNS-rebinding protection the MCP transport requires;
  OAuth still applies to every request.

## 2. Database

- Amazon RDS for PostgreSQL 16, encryption at rest, in the same VPC.
- Store `DATABASE_URL` in Secrets Manager and inject it into the task or instance environment.
- Run migrations once per release: `python -m app migrate` (the `migrate` Compose service, or a one-off ECS task).

## 3. Asset storage (S3)

Set:

```bash
STORAGE_BACKEND=s3
VOICE_MCP_STORAGE__S3__BUCKET=my-voice-mcp-assets
VOICE_MCP_STORAGE__S3__REGION=eu-west-1
VOICE_MCP_STORAGE__S3__PREFIX=voice-mcp
# optional
VOICE_MCP_STORAGE__S3__ENCRYPTION=sse-kms
VOICE_MCP_STORAGE__S3__KMS_KEY_ID=arn:aws:kms:...
VOICE_MCP_STORAGE__S3__PRESIGN_DOWNLOADS=true
```

Bucket requirements (verify in IaC, not at runtime):

- Block Public Access on, bucket policy denying non-TLS requests (`aws:SecureTransport = false`).
- SSE-S3 by default or SSE-KMS.
- Lifecycle rule aborting incomplete multipart uploads after 1 day.

Least-privilege task/instance role:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {"Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
     "Resource": "arn:aws:s3:::my-voice-mcp-assets/voice-mcp/*"},
    {"Effect": "Allow", "Action": ["s3:ListBucket"], "Resource": "arn:aws:s3:::my-voice-mcp-assets",
     "Condition": {"StringLike": {"s3:prefix": ["voice-mcp/*"]}}}
  ]
}
```

Credentials come only from the default boto3 chain (instance or task role). The server checks the bucket with
`HeadBucket` at startup and refuses to start if it is not accessible; it doesn't fall back to local storage.

Orphans (objects uploaded before a failed database commit) are removed by the hourly reconciler once they are
older than `storage.orphan_retention_hours` (default 24) and no `media_assets` row references them.

## 4. Models

Models aren't downloaded at runtime. Either:

- attach an EBS volume (or EFS) mounted at `/srv/voice-mcp/models` and populate it once with
  `python -m app models install ...` from an instance with internet access, or
- bake them into a separate private image layer.

Mount it read-only in the serving container.

## 5. Inputs

Tools take host-visible file paths. With one process (or an API and a worker sharing a filesystem) that means a
shared mounted directory (EFS) listed in `VOICE_MCP_SERVER__ALLOWED_INPUT_DIRECTORIES`, at the same path everywhere.
With `VOICE_MCP_STORAGE__INPUT_TRANSFER=ingest` the API copies each input to S3 at submission and the worker reads
it from there, so the worker needs no access to the input directory (section 8).

## 6. Identity provider

Run the same Keycloak container on ECS with RDS for its database, importing `deploy/keycloak/realm-voice.json`
after changing the audience to the public MCP URL. `VOICE_MCP_AUTH__ISSUER_URL` must be the issuer exactly as it
appears in tokens (the public Keycloak URL). If the server cannot reach that URL from its subnet, set
`VOICE_MCP_AUTH__JWKS_URL` to the internal JWKS endpoint (for example through the internal ALB or service discovery);
validation still uses the public issuer, and the `identity_provider` health check probes the internal endpoint. Amazon Cognito is not supported: its access tokens do not carry
the `aud` claim and resource indicators the MCP authorization flow expects.

## 8. API on CPU, GPU worker on demand

See [docs/gpu-worker.md](../../docs/gpu-worker.md) for the roles, the controller's behavior table and the drain
protocol.

- **API service:** ECS on EC2 or Fargate, CPU only, the same image with `SERVICE_ROLE=api`, command
  `serve --role api --transport streamable_http`, behind the ALB. It needs no models volume and no GPU. Install the
  `aws` extra (and `s3`) in its image.
- **GPU worker service:** a separate ECS service, command `worker`, `SERVICE_ROLE=worker`, on a GPU capacity
  provider (EC2 Auto Scaling group with the ECS GPU-optimized AMI) with **managed scaling and a minimum of 0**,
  desired count 0. Mount the models volume read-only. Run `python -m app worker --register-only` once as a one-off
  task per deployment and after every `models install`, so the API can validate requests before the first start.
- **Optional warm pool:** an EC2 Auto Scaling warm pool in the `Stopped` state keeps the CUDA image cached on
  stopped instances and shortens cold starts.
- **Storage:** `STORAGE_BACKEND=s3` with `VOICE_MCP_STORAGE__INPUT_TRANSFER=ingest`; no shared filesystem is needed.
- **Controller settings on the API:**

  ```bash
  SERVICE_ROLE=api
  GPU_ON_DEMAND=true
  VOICE_MCP_GPU_WORKER__ON_DEMAND__TARGET=aws_ecs
  VOICE_MCP_GPU_WORKER__ON_DEMAND__AWS_ECS__REGION=eu-west-1
  VOICE_MCP_GPU_WORKER__ON_DEMAND__AWS_ECS__CLUSTER=voice
  VOICE_MCP_GPU_WORKER__ON_DEMAND__AWS_ECS__SERVICE=voice-gpu-worker
  ```

  Leave `GPU_ON_DEMAND` unset on the worker. Start with `VOICE_MCP_GPU_WORKER__ON_DEMAND__DRY_RUN=true` to watch the
  decisions in the logs and `health_check` before letting it act.
- **IAM for the API task role:** only `ecs:DescribeServices` and `ecs:UpdateService` on the worker service's ARN:

  ```json
  {
    "Effect": "Allow",
    "Action": ["ecs:DescribeServices", "ecs:UpdateService"],
    "Resource": "arn:aws:ecs:eu-west-1:123456789012:service/voice/voice-gpu-worker"
  }
  ```

- **Stop timeout:** set the worker container's `stopTimeout` (EC2 launch type) or `ECS_CONTAINER_STOP_TIMEOUT` on
  the agent as high as allowed. The controller always drains through the registry first, so this only matters for
  crashes and manual stops. Fargate caps `stopTimeout` at 120 s and has no GPUs, so it cannot run the worker.
- **Scale-in protection:** the controller sets the desired count to 0 only after the worker reports `stopped` (or on
  a forced stop), so managed scaling doesn't remove an instance in the middle of a job.

## 9. Checklist

- [ ] `diagnose` passes on the instance (`docker compose run --rm admin diagnose`).
- [ ] `health_check` reports `linux-cuda`, `ctranslate2` on `cuda`, `torch` on `cuda`.
- [ ] Unauthenticated requests get `401` with `WWW-Authenticate`; tokens without `mcp:access` get `403`
  (`deploy/smoke/oauth_smoke.sh` with `KEYCLOAK_URL` and `MCP_URL` set checks all of these).
- [ ] `get_job` returns `s3://` URIs; presigned URLs only with `include_download_urls=true`.
- [ ] Split deployment: `python -m app gpu status` shows the target and the worker row; a submission starts the
  worker service, and it drains and scales back to 0 after `idle_timeout_seconds` without work.
