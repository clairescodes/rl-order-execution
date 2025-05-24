"""
env.py

Initializes QLib and constructs a SingleAssetOrderExecutionSimple
environment for RL-based order execution from a user-specified config.
"""
import os
from pathlib import Path

import qlib
from qlib.backtest.decision import Order, OrderDir
from qlib.rl.order_execution.simulator_simple import SingleAssetOrderExecutionSimple

def make_env(config: dict) -> SingleAssetOrderExecutionSimple:
    """
    Initializes QLib and returns a SingleAssetOrderExecutionSimple 
    (RL order-execution env).
    config must include:
      - provider_uri: path to qlib_data ("~/.qlib/qlib_data/cn_data")
      - region: "cn" or "us"
      - instrument: a single ticker string (e.g. "SZ000001")
      - start_time, end_time: ISO datetimes for the order (e.g. "2025-05-22 09:31:00")
      - amount: total shares to execute
      - direction: "buy" or "sell"
      - data_granularity: (optional, default=1)
      - ticks_per_step: (optional, how many ticks in each RL step)
      - vol_threshold: (optional, max fraction of market volume tradable)
      - feature_columns_today / feature_columns_yesterday: (optional lists of extra columns)
    """
    # init QLib
    provider_uri = os.path.expanduser(config["provider_uri"])
    qlib.init(provider_uri=provider_uri, region=config["region"])

    # build an Order for a single‐asset execution
    dir_str = config.get("direction", "buy").lower()
    direction = OrderDir.BUY if dir_str == "buy" else OrderDir.SELL

    order = Order(
        stock_id=config["instrument"],
        start_time=config["start_time"],
        end_time=config["end_time"],
        amount=float(config["amount"]),
        direction=direction,
    )

    # construct the Simple SAOE simulator
    env = SingleAssetOrderExecutionSimple(
        order=order,
        data_dir=Path(provider_uri),
        feature_columns_today=config.get("feature_columns_today", []),
        feature_columns_yesterday=config.get("feature_columns_yesterday", []),
        data_granularity=int(config.get("data_granularity", 1)),
        ticks_per_step=int(config.get("ticks_per_step", 1)),
        vol_threshold=config.get("vol_threshold", None),
    )

    return env
