"""Atrex-Bench: End-to-end benchmark for kernel generation."""

from atrex_bench.sdk import (
    AtrexConfigError,
    AtrexEvaluationError,
    AtrexSDKError,
    evaluate,
)
from atrex_bench.shards import SHAPE_SHARD_SCHEMA_VERSION, merge_evaluation_shards

__version__ = "0.1.0"

__all__ = [
    "AtrexConfigError",
    "AtrexEvaluationError",
    "AtrexSDKError",
    "evaluate",
    "merge_evaluation_shards",
    "SHAPE_SHARD_SCHEMA_VERSION",
]
