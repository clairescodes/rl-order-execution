# trainer.py
import os, yaml
import torch
from torch.utils.tensorboard import SummaryWriter
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.trainer.base import OnpolicyTrainer
from tianshou.utils import TensorboardLogger

from env import make_env
from agent import build_policy

def main():
    # load YAML config 
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    env_cfg, policy_cfg, train_cfg = cfg["env"], cfg["policy"], cfg["training"]

    # make vectorized envs
    train_envs = [make_env(env_cfg) for _ in range(train_cfg["n_train_envs"])]
    test_envs  = [make_env(env_cfg) for _ in range(train_cfg["n_test_envs"])]

    # build network + optimizer + policy
    net = build_policy(
        obs_shape=train_envs[0].observation_space.shape,
        action_space=train_envs[0].action_space,
        **policy_cfg
    )
    policy, optimizer = net  # if build_policy returns both

    # collectors and buffer
    buffer = VectorReplayBuffer(train_cfg["buffer_size"], len(train_envs))
    train_collector = Collector(policy, train_envs, buffer)
    test_collector  = Collector(policy, test_envs)

    # warm-up
    train_collector.collect(n_step=policy_cfg["step_per_epoch"] * len(train_envs))

    # tensorboard
    tb_writer = SummaryWriter(os.path.expanduser(train_cfg.get("log_dir", "~/runs")))
    logger    = TensorboardLogger(tb_writer)

    # instantiate the on-policy trainer
    trainer = OnpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=train_cfg["max_epoch"],
        step_per_epoch=policy_cfg["step_per_epoch"],
        repeat_per_collect=policy_cfg["repeat_per_collect"],
        batch_size=policy_cfg["batch_size"],
        episode_per_test=train_cfg["n_test_envs"],
        step_per_collect=policy_cfg["step_per_epoch"],  # same as step_per_epoch * envs
        logger=logger,
    )

    # run training
    for epoch, train_stat, test_stat, info in trainer:
        pass

    print("Best reward:", trainer.best_reward)
    tb_writer.close()


if __name__ == "__main__":
    main()
