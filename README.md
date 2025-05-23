# rl-order-execution

This project builds and evaluates a Reinforcement Learning agent that learns how to split a large order over time to minimize market impact and execution cost, comparing its performance to classic strategies like TWAP (time-weighted average price) or VWAP (volume-weighted average price).

Built on top of Qlib (https://github.com/microsoft/qlib), this project uses historical market data and a gym-compatible environment to train agents like PPO in a realistic execution scenario.

## Motivation 

Order execution is a critical part of algorithmic trading, where poor execution can result in slippage, higher transaction costs, or missed opportunities. Traditional strategies like TWAP and VWAP are static and don't adapt to real time market conditions.

## Goals 

- Simulate a limit order book or use historical market data.

- Train an RL agent to decide how much to trade at each timestep.

- Backtest the strategy using historical order book snapshots or price series.

- Compare performance against rule-based baselines.

## Academic Context 

This project is inspired by recent advances in the use of reinforcement learning (RL) for optimal order execution, a key challenge in algorithmic trading that involves minimizing transaction costs while executing large trades.

Traditional approaches, such as:
- Almgren-Chriss (AC) model and
- Bertsimas and Lo (1998)
offer elegant closed-form solutions, but rely on strong assumptions: continuous trading, constant volatility, and no dynamic market impact.

Recent RL-based research relaxes these assumptions and adapts execution strategies to real-time microstructure signals:
- Nevmyvaka et al. (2006): Q-learning with millisecond NASDAQ LOB data.
- Cartea et al. (2015): Trade-off between price risk and market impact.
- Wang et al. (2021): Hierarchical RL with explicit execution modules.
- Ning et al. (2020) & Fang et al. (2021): Rich state features (e.g., bid-ask spread, order imbalance).
- Bao & Liu (2019): Multi-agent execution with fairness-adjusted rewards.

These studies show RL’s potential to model dynamic environments where traditional models break down.
