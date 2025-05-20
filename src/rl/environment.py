import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

class PairsEnv(gym.Env):
    """
    Gym environment for pairs trading.
    Obs: flattened window of z-scores for each pair.
    Act: MultiDiscrete [0,1,2] per pair (hold, long spread, short spread).
    Reward: mark-to-market PnL minus transaction costs.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        prices_csv: str,
        pairs_csv: str,
        window_length: int = 270,
        trading_cost: float = 0.0005,
    ):
        super().__init__()
        prices_df = pd.read_csv(prices_csv, index_col=0, parse_dates=True)
        pairs_df  = pd.read_csv(pairs_csv)
        self.pairs = list(zip(pairs_df.TickerA, pairs_df.TickerB))

        # precompute z-score series
        self.zscores = {}
        for a, b in self.pairs:
            s1, s2 = prices_df[a], prices_df[b]
            spread  = s1 - s2
            self.zscores[(a,b)] = (spread - spread.mean()) / spread.std()

        self.dates  = prices_df.index
        self.window = window_length
        self.cost   = trading_cost

        # obs = window_length * num_pairs
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(len(self.pairs)*self.window,),
            dtype=np.float32
        )
        # actions per pair: hold / long / short
        self.action_space = spaces.MultiDiscrete([3] * len(self.pairs))
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # pick a random start so there's enough future data
        self.t = np.random.randint(self.window, len(self.dates)-1)
        self.prev_pos = np.zeros(len(self.pairs), dtype=int)
        obs = self._get_obs()
        return obs, {}  # Return observation and empty info dict

    def _get_obs(self):
        obs = []
        for (a,b) in self.pairs:
            zs = self.zscores[(a,b)].iloc[self.t-self.window:self.t].values
            obs.append(zs)
        # Return observation with shape (n_env, obs_dim)
        return np.concatenate(obs).astype(np.float32).reshape(1, -1)

    def step(self, actions):
        # map {0,1,2}→{0,+1,-1}
        pos = np.where(actions==1, 1, np.where(actions==2, -1, 0))
        reward = 0.0
        for i, (a,b) in enumerate(self.pairs):
            zs = self.zscores[(a,b)]
            dz = zs.iloc[self.t+1] - zs.iloc[self.t]
            reward += pos[i] * dz
            reward -= self.cost * abs(pos[i] - self.prev_pos[i])
        self.prev_pos = pos
        self.t += 1
        terminated = self.t >= len(self.dates)-1
        truncated = False
        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        # optional: implement a plot or print
        pass
