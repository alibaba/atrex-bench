#!/usr/bin/env bash
# Build the Atrex-Bench ROCm image.
#
#   ./docker/build.sh                                  # -> atrex-bench:rocm7.2  (local only)
#   REGISTRY=docker.io/<you> ./docker/build.sh         # also adds a remote tag + push hint
#   IMAGE=foo TAG=bar ./docker/build.sh                # override image name / tag
#   BASE_IMAGE=... AITER_REF=<sha> ./docker/build.sh   # override build args (see Dockerfile.rocm)
set -euo pipefail

IMAGE="${IMAGE:-atrex-bench}"
TAG="${TAG:-rocm7.2}"
REGISTRY="${REGISTRY:-}"          # empty => local-only build, no remote tag

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

LOCAL_TAG="${IMAGE}:${TAG}"

build_args=(-f "${SCRIPT_DIR}/Dockerfile.rocm" -t "${LOCAL_TAG}")
[ -n "${BASE_IMAGE:-}" ] && build_args+=(--build-arg "BASE_IMAGE=${BASE_IMAGE}")
[ -n "${AITER_REF:-}" ]  && build_args+=(--build-arg "AITER_REF=${AITER_REF}")

if [ -n "${REGISTRY}" ]; then
    REMOTE_TAG="${REGISTRY}/${IMAGE}:${TAG}"
    build_args+=(-t "${REMOTE_TAG}")
fi

docker build "${build_args[@]}" "${REPO_ROOT}"

echo
echo "Built: ${LOCAL_TAG}"
if [ -n "${REGISTRY}" ]; then
    echo "Built: ${REMOTE_TAG}"
    echo
    echo "To push: docker push ${REMOTE_TAG}"
fi
