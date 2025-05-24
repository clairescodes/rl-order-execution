# trainer.py
"""
Train a PPO agent to execute large orders via Qlib’s SingleAssetOrderExecutionSimple env.
"""

import os
import yaml
import numpy as np
import torch
from torch import nn
from tianshou.policy import PPOPolicy
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.trainer import onpolicy_trainer
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger

from env import make_env       # env factory
from agent import build_policy # builds actor & critic nets

def main():
    # Load config
    cfg_path = "configs/config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    env_cfg = cfg["env"]
    policy_cfg = cfg["policy"]
    train_cfg = cfg["training"]

    # Create vectorized envs for train & test
    train_envs = [make_env(env_cfg) for _ in range(train_cfg["n_train_envs"])]
    test_envs  = [make_env(env_cfg) for _ in range(train_cfg["n_test_envs"])]

    # Build networks and wrap into PPOPolicy
    # build_policy returns (policy, optimizer)
    policy, optimizer = build_policy(
        obs_shape=train_envs[0].observation_space.shape,
        action_space=train_envs[0].action_space,
        **policy_cfg
    )

    # Prepare collectors & buffer
    buffer = VectorReplayBuffer(train_cfg["buffer_size"], len(train_envs))
    train_collector = Collector(policy, train_envs, buffer)
    test_collector  = Collector(policy, test_envs)

    # Warm up (fill buffer)
    train_collector.collect(n_step=policy_cfg["step_per_epoch"] * len(train_envs))

    # Create TensorBoard writer + Tianshou logger
    log_dir   = os.path.expanduser(train_cfg.get("log_dir", "~/runs/rl_order_exec"))
    tb_writer = SummaryWriter(log_dir)            # writes to disk for tensorboard
    tb_logger = TensorboardLogger(tb_writer)      # wraps it for onpolicy_trainer

    # Run training loop
    result = onpolicy_trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=train_cfg["max_epoch"],
        step_per_epoch=policy_cfg["step_per_epoch"],
        repeat_per_collect=policy_cfg["repeat_per_collect"],
        test_num=train_cfg["n_test_envs"],
        batch_size=policy_cfg["batch_size"],
        logger=tb_logger, 
        save_best_fn=train_cfg.get("save_best_fn"),
    )

    # Clean up & report
    tb_writer.close()
    print("Training finished! Best reward:", result["best_reward"])

if __name__ == "__main__":
    main()