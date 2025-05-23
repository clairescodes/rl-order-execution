import gym
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    """
    trading environment wrapping QLib data for RL.
    Observation: cleaned historical feature vectors for a window of days.
    Action: a scalar in [-1, 1] representing position (short to long).
    Reward: daily PnL.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        data: pd.DataFrame,
        window_size: int = 10,
        transaction_cost: float = 1e-4,
    ):
        super().__init__()
        # DataFrame indexed by (instrument, datetime), columns=features
        self.data = data
        self.window = window_size
        self.tc = transaction_cost

        # Build index of dates and instruments
        self.dates = sorted(data.index.get_level_values("datetime").unique())
        self.tickers = sorted(data.index.get_level_values("instrument").unique())

        # Action and observation spaces
        # Single continuous action: position [-1,1]
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        # Observation: window x feature dims
        feat_dim = data.shape[1]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.window, feat_dim), dtype=np.float32
        )

        # Internal pointers
        self.current_step = self.window
        self.position = 0.0

    def reset(self):
        """Resets to the first possible state."""
        self.current_step = self.window
        self.position = 0.0
        return self._get_observation()

    def _get_observation(self):
        date = self.dates[self.current_step]
        # extract last `window` days for all tickers and flatten
        start = self.current_step - self.window
        window_dates = self.dates[start : self.current_step]
        obs = []
        for d in window_dates:
            X = self.data.xs(d, level="datetime").values  # shape: (n_tickers, feat_dim)
            obs.append(X)
        obs = np.stack(obs, axis=0)
        return obs.astype(np.float32)

    def step(self, action):
        """
        Takes action, computes reward (PnL), and advances one day.
        """
        action = float(action)
        prev_date = self.dates[self.current_step - 1]
        curr_date = self.dates[self.current_step]

        # Price change vector for all tickers
        open_prev = self.data.xs(prev_date, level="datetime")["$open"].values
        close_curr = self.data.xs(curr_date, level="datetime")["$close"].values
        returns = (close_curr - open_prev) / open_prev

        # PnL = position * return, averaged across universe
        pnl = np.mean(action * returns) - self.tc * abs(action - self.position)
        self.position = action
        reward = pnl

        self.current_step += 1
        done = self.current_step >= len(self.dates)

        obs = self._get_observation() if not done else None
        info = {"pnl": pnl}
        return obs, reward, done, info

    def render(self, mode="human"):
        pass

    def close(self):
        pass
