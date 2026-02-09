#!/bin/bash
set -e

docker-compose run --rm before
docker-compose run --rm after
docker-compose run --rm test
