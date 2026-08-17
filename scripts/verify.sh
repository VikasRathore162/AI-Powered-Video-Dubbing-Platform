#!/usr/bin/env bash
# The one command that proves the whole thing works. Everything runs in Docker.
#
#   ./scripts/verify.sh
#
# Tests on both databases, then the real models, then the same black-box E2E
# against each swappable backend — the queue and the storage layer are
# configuration, so each gets exercised rather than asserted.
set -euo pipefail
cd "$(dirname "$0")/.."
export APP_UID="$(id -u)"          # so the container can write ./data

TOOLS="docker compose run --rm -T tools"
PG="postgresql+psycopg2://dub:dub@postgres:5432/dubbing"
SQS_OPTS='{"region":"elasticmq","is_secure":false,"visibility_timeout":3600,"polling_interval":1,"wait_time_seconds":5}'

echo "== 1/8 build =="
docker compose build

echo "== 2/8 infrastructure =="
docker compose up -d --wait postgres redis

echo "== 3/8 tests (SQLite) =="
$TOOLS pytest

echo "== 4/8 tests (Postgres — column widths and FK order only bite here) =="
docker compose run --rm -T -e TEST_DB_URL="$PG" tools pytest

echo "== 5/8 models, fixture, and the real-model integration test =="
$TOOLS python scripts/setup.py models
[ -f tests/two_speakers_en.mp4 ] || $TOOLS python scripts/setup.py fixture
$TOOLS env RUN_FULL_PIPELINE=1 pytest -m integration

echo "== 6/8 end to end on the default stack (Redis, local storage) =="
docker compose up -d --wait api worker
$TOOLS python scripts/e2e.py http://api:8000
docker compose down --remove-orphans

echo "== 7/8 the same end to end on SQS, then on RabbitMQ =="
export BROKER_URL='sqs://local:local@sqs:9324' BROKER_TRANSPORT_OPTIONS="$SQS_OPTS"
docker compose --profile sqs up -d --wait sqs api worker
docker compose --profile sqs run --rm -T tools python scripts/e2e.py http://api:8000
docker compose --profile sqs down --remove-orphans

export BROKER_URL='amqp://guest:guest@rabbitmq:5672//' BROKER_TRANSPORT_OPTIONS='{}'
docker compose --profile rabbitmq up -d --wait rabbitmq api worker
docker compose --profile rabbitmq run --rm -T tools python scripts/e2e.py http://api:8000
docker compose --profile rabbitmq down --remove-orphans
unset BROKER_URL BROKER_TRANSPORT_OPTIONS

echo "== 8/8 the same end to end on S3 storage, then with cloud configuration =="
export STORAGE=s3 AWS_ENDPOINT_URL=http://minio:9000 \
       STORAGE_OPTIONS='{"bucket":"dubbing","endpoint_url":"http://minio:9000"}'
docker compose --profile minio up -d --wait minio
docker compose --profile minio run --rm -T tools python -c "
import boto3, os
boto3.client('s3', endpoint_url=os.environ['AWS_ENDPOINT_URL']).create_bucket(Bucket='dubbing')"
docker compose --profile minio up -d --wait api worker
docker compose --profile minio run --rm -T tools python scripts/e2e.py http://api:8000
docker compose --profile minio down --remove-orphans
unset STORAGE STORAGE_OPTIONS AWS_ENDPOINT_URL

export CLOUD_CONFIG='aws-ssm://dubbing/' AWS_ENDPOINT_URL=http://localstack:4566
docker compose --profile localstack up -d --wait localstack
docker compose --profile localstack run --rm -T tools python scripts/setup.py cloud
docker compose --profile localstack up -d --wait api worker
docker compose --profile localstack run --rm -T tools python -c "
from app.config import get_settings
s = get_settings()
assert s.limits.max_duration_sec == 300, s.limits.max_duration_sec
assert s.elevenlabs_api_key.get_secret_value() == 'seeded-from-parameter-store'
print('  settings and a secret loaded from Parameter Store')"
docker compose --profile localstack run --rm -T tools python scripts/e2e.py http://api:8000
docker compose --profile localstack down --remove-orphans
unset CLOUD_CONFIG AWS_ENDPOINT_URL

echo
echo "ALL VERIFICATIONS PASSED — start the stack again with: docker compose up -d"
