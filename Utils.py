import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Independent, OneHotCategoricalStraightThrough, Bernoulli
from torch.distributions.kl import kl_divergence
import warnings
import numpy as np
from typing import Dict, Optional, Sequence
from math import isnan

import labmaze
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Wall, Ball
from minigrid.minigrid_env import MiniGridEnv



def compute_stochastic_state(logits, discrete=32, sample=True):
    logits = logits.view(*logits.shape[:-1], -1, discrete)
    dist = Independent(OneHotCategoricalStraightThrough(logits=logits), 1) # to creat joint distribution of categorical variables
    stochastic_state = dist.rsample() if sample else dist.mode # to get random samples during training and mode during evaluation
    return stochastic_state


def symlog(x):
    # Symmetric logarithm function: symlog(x) = sign(x) * log(1 + |x|)
    return torch.sign(x) * torch.log1p(torch.abs(x))

def symexp(x):
    # Inverse of symlog: symexp(x) = sign(x) * (exp(|x|) - 1)
    return torch.sign(x) * (torch.expm1(torch.abs(x)))


def create_bins(v_min=-20, v_max=20, num_bins=250, device='cpu'):
    # Cearte a exponentially spaced bins (exponentiality in the orginal space instead of symlog space)
    l_bins = torch.linspace(v_min, v_max, num_bins, device=device)
    return l_bins



def two_hot_encoding(x, bins):
    """
    x: [seq_len, batch_size, reward_dim]
    bins: [num_bins]
    Returns: [*x_shape, num_bins]
    """
    original_shape = x.shape[:-1]
    num_bins = bins.shape[0]
    device = x.device
    
    # N = seq_len * batch_size * reward_dim
    x_flat = x.reshape(-1) 
    num_elements = x_flat.shape[0]

    # [1, num_bins]
    bins_expanded = bins.unsqueeze(0) 

    # [N, 1]
    indices = torch.sum(bins_expanded <= x_flat.unsqueeze(1), dim=1) - 1
    indices = torch.clamp(indices, 0, num_bins - 2)
    
    bin_left = bins[indices]
    bin_right = bins[indices + 1]
    
    weight_right = (x_flat - bin_left) / (bin_right - bin_left)
    weight_right = torch.clamp(weight_right, 0, 1)
    weight_left = 1.0 - weight_right

    # [N, num_bins]
    two_hot = torch.zeros(num_elements, num_bins, device=device)
    two_hot.scatter_(1, indices.unsqueeze(1), weight_left.unsqueeze(1))
    two_hot.scatter_(1, (indices + 1).unsqueeze(1), weight_right.unsqueeze(1))

    return two_hot.view(*original_shape, num_bins)


def prepare_obs(obs, cnn_keys, num_envs, device):
    # 1. Ensure obs is a dictionary (handles ImgObsWrapper returning raw arrays)
    if not isinstance(obs, dict):
        obs = {cnn_keys[0]: obs}

    torch_obs = {}
    for k, v in obs.items():
        # 2. Convert to Tensor and move to GPU/CPU
        t = torch.as_tensor(v, device=device).float()
        
        if k in cnn_keys:
            # Handle [num_envs, H, W, C] -> [num_envs, C, H, W]
            if t.shape[-1] == 3: 
                t = t.permute(0, 3, 1, 2)
            
            # Normalize to [-0.5, 0.5] and add sequence dimension if needed
            # Shape becomes [num_envs, C, H, W]
            torch_obs[k] = t / 255.0 - 0.5
        else:
            # Standard vector observation
            torch_obs[k] = t.view(num_envs, -1)

    return torch_obs

class SymLogDistribution:
    def __init__(self, mode, dims, tol=1e-8):
        self._mode = mode
        self._dims = tuple([-x for x in range(1, dims + 1)])
        self._tol = tol
    
    @property
    def mode(self):
        return symexp(self._mode)
    
    @property
    def mean(self):
        return symexp(self._mode)
    
    def log_prob(self, value):
        assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
        
        distance = (self._mode - symlog(value)) ** 2
        distance = torch.where(distance < self._tol, 0, distance)

        loss = distance.mean(self._dims)
        return -loss
    
class TwoHotEncodingDistribution:
    def __init__(self, logits, dims, low=-20.0, high=20.0):
        self.logits = logits
        self.probs = F.softmax(logits, dim=-1)
        self.bins = create_bins(v_min=low, v_max=high, num_bins=logits.shape[-1], device=logits.device)
        self.dims = tuple([-x for x in range(1, dims + 1)])

    @property
    def mean(self):
        return symexp((self.probs * self.bins).sum(dim=-1, keepdim=True))

    @property
    def mode(self):
        return self.mean

    def log_prob(self, x):
        x = symlog(x)
        # print("size of x:", x.shape)
        
        target = two_hot_encoding(x, self.bins)
        log_pred = F.log_softmax(self.logits, dim=-1)
        
        # print("size of target:", target.shape)
        # print("size of log_pred:", log_pred.shape)
        return (target * log_pred).sum(dim=-1)

# Safe mode ensures that the return of the mode is accurate
class BernoulliSafeMode(Bernoulli):
    def __init__(self, probs=None, logits=None, validate_args=None):
        super().__init__(probs, logits, validate_args)

    @property
    def mode(self):
        mode = (self.probs > 0.5).to(self.probs)
        return mode
    
    
def world_model_loss(po, obs, pr, rewards, pc, continue_targets, priors_logits, posteriors_logits, cfg=None,
                     beta_pred=1, beta_dyn=1, beta_rep=0.1, kl_free_nats=1, kl_regularizer=1):

    # Get parameters
    if cfg:
        w_cfg = cfg.algo.world_model
        beta_pred = w_cfg.beta_pred
        beta_dyn = w_cfg.beta_dyn
        beta_rep = w_cfg.beta_rep
        kl_free_nats = w_cfg.kl_free_nats
        kl_regularizer = w_cfg.kl_regularizer
    
    obs_loss = -po.log_prob(obs)
    reward_loss = -pr.log_prob(rewards)
    continue_loss = -pc.log_prob(continue_targets)
    
    pred_loss = obs_loss + reward_loss + continue_loss
    
    # KL diveregence loss
    ## Stop-gradeint operator to force prior to approach the posterior, not vice-versa 
    dyn_loss = kl = kl_divergence(
        Independent(OneHotCategoricalStraightThrough(logits=posteriors_logits.detach()), 1),
        Independent(OneHotCategoricalStraightThrough(logits=priors_logits), 1),
    )
    free_nats = torch.full_like(dyn_loss, kl_free_nats)
    ## Free bits employment: in loss, clipping means the maximum
    dyn_loss = torch.maximum(dyn_loss, free_nats)
    
    
    rep_loss = kl_divergence(
        Independent(OneHotCategoricalStraightThrough(logits=posteriors_logits), 1),
        Independent(OneHotCategoricalStraightThrough(logits=priors_logits.detach()), 1),
    )
    rep_loss = torch.maximum(rep_loss, free_nats)

    kl_loss = beta_dyn * dyn_loss + beta_rep * rep_loss
    reconstruction_loss = (beta_pred * pred_loss + kl_regularizer * kl_loss).mean()
    
    return (
        reconstruction_loss,
        kl.mean(),
        kl_loss.mean(),
        reward_loss.mean(),
        obs_loss.mean(),
        continue_loss.mean(),
    )
    
    
# The bootstrapped lambda returns computation
def compute_lambda_values(rewards, values, continues, lmbda=0.95):
    
    ## Last time step
    vals = [values[-1:]]
    ## continues already multiply with discount factor gamma
    interm = rewards + continues * (1 - lmbda) * values
    for t in reversed(range(len(continues))):
        vals.append(interm[t] + continues[t] * lmbda * vals[-1])
    ## Remove the last value which is not return
    ret = torch.cat(list(reversed(vals))[:-1])
    return ret


class Moments(nn.Module):
    def __init__(self, decay=0.99, max_=1e8, percentile_low=0.05, percentile_high=0.95):
        super().__init__()
        self._decay = decay
        self._max_ = torch.tensor(max_)
        self.percentile_low = percentile_low
        self.percentile_high = percentile_high
        
        self.register_buffer('low', torch.zeros((), dtype=torch.float32))
        self.register_buffer('high', torch.zeros((), dtype=torch.float32))
        
    ## Update the moments using a batch of data
    def forward(self, x):
        x = x.detach().float()
        
        low = torch.quantile(x, self.percentile_low)
        high = torch.quantile(x, self.percentile_high)
        
        # 1-decay proportion from current data
        self.low = self._decay * self.low + (1 - self._decay) * low
        self.high = self._decay * self.high + (1 - self._decay) * high
        # _max to avoid numerical issues
        # invscale = torch.max(1 / self._max_, self.high - self.low)
        invscale = torch.max(torch.tensor(1.0, device=x.device), self.high - self.low)
        return self.low.detach(), invscale.detach()
    

class MetricAggregatorException(Exception):
    pass
class MetricAggregator:
    # Aggrgate metrics over time
    disabled = False

    def __init__(self, metrics=None, raise_on_missing=False):
        self.metrics = {} if metrics is None else metrics
        self._raise_on_missing = raise_on_missing

    def __iter__(self):
        return iter(self.metric.keys())
    
    def add(self, name, metric):
        # Add a new metric with default value
        if not self.disabled:
            if name not in self.metrics:
                self.metrics.setdefault(name, metric)
            else:
                if self._raise_on_missing:
                    raise MetricAggregatorException(f"Metric {name} already exists")
                else:
                    warnings.warn(
                        f"The key '{name}' is already in the metric aggregator. Nothing will be added.", UserWarning)
    

    @torch.no_grad()
    def update(self, name, value):
        # Update the metric with new value
        if not self.disabled:
            if name not in self.metrics:
                if self._raise_on_missing:
                    raise MetricAggregatorException(f"Metric {name} does not exist")
                else:
                    warnings.warn(
                        f"The key '{name}' is missing from the metric aggregator. Nothing will be added.", UserWarning
                    )
            else:
                self.metrics[name].update(value)
   
    def pop(self, name):
        # Remove a metric
        if not self.disabled:
            if name not in self.metrics:
                if self._raise_on_missing:
                    raise MetricAggregatorException(f"Metric {name} does not exist")
                else:
                    warnings.warn(
                        f"The key '{name}' is missing from the metric aggregator. Nothing will be popped.", UserWarning
                    )
            self.metrics.pop(name, None)

    def reset(self):
        # Resest all metrics to initial state
        if not self.disabled:
            for metric in self.metrics.values():
                metric.reset()

    def to(self, device="cpu"):
        # Move all metrics to the given device
        if not self.disabled:
            if self.metrics:
                for k, v in self.metrics.items():
                    self.metrics[k] = v.to(device)
        return self
    

    @torch.no_grad()
    def compute(self):
        """Reduce the metrics to a single value
        Returns:
            Reduced metrics
        """
        reduced_metrics = {}
        if not self.disabled:
            if self.metrics:
                for k, v in self.metrics.items():
                    reduced = v.compute()
                    is_tensor = torch.is_tensor(reduced)
                    if is_tensor and reduced.numel() == 1:
                        reduced_metrics[k] = reduced.item()
                    else:
                        if not is_tensor:
                            warnings.warn(
                                f"The reduced metric {k} is not a scalar tensor: type={type(reduced)}. "
                                "This may create problems during the logging phase.",
                                category=RuntimeWarning,
                            )
                        else:
                            warnings.warn(
                                f"The reduced metric {k} is not a scalar: size={v.size()}. "
                                "This may create problems during the logging phase.",
                                category=RuntimeWarning,
                            )
                        reduced_metrics[k] = reduced

                    is_tensor = torch.is_tensor(reduced_metrics[k])
                    if (is_tensor and torch.isnan(reduced_metrics[k]).any()) or (
                        not is_tensor and isnan(reduced_metrics[k])
                    ):
                        reduced_metrics.pop(k, None)
        return reduced_metrics


class LocalBuffer:
    """Stores data for a single environment to allow individual access."""
    def __init__(self, buffer_size: int):
        self.buffer_size = int(buffer_size)
        self._pos = 0
        self._full = False
        self._buf: Dict[str, np.ndarray] = {}

    def __getitem__(self, key: str):
        return self._buf[key]

    def add(self, data: Dict[str, np.ndarray]):
        # data shape: [Steps, 1, Features...]
        seq_len = data[next(iter(data.keys()))].shape[0]
        if not self._buf:
            for k, v in data.items():
                # Store as [Capacity, Features...]
                self._buf[k] = np.zeros((self.buffer_size, *v.shape[2:]), dtype=v.dtype)

        idxes = np.arange(self._pos, self._pos + seq_len) % self.buffer_size
        for k, v in data.items():
            self._buf[k][idxes] = v.squeeze(1) # Remove env dimension

        if self._pos + seq_len >= self.buffer_size:
            self._full = True
        self._pos = (self._pos + seq_len) % self.buffer_size

class ReplayBuffer:
    def __init__(self, buffer_size: int, n_envs: int):
        self.n_envs = n_envs
        self.buffer_size = buffer_size
        # This creates the 'rb.buffer[i]' access your code needs
        self.buffer = [LocalBuffer(buffer_size) for _ in range(n_envs)]

    def __len__(self):
        # Returns the length of the first buffer as a proxy
        return len(self.buffer[0]._buf[next(iter(self.buffer[0]._buf.keys()))]) if self.buffer[0]._buf else 0

    def add(self, data: Dict[str, np.ndarray], indices: Optional[Sequence[int]] = None, **kwargs):
        if indices is None:
            indices = list(range(self.n_envs))
        
        for i, env_idx in enumerate(indices):
            # Slice data for this specific environment: [Steps, 1, Features...]
            env_data = {k: v[:, i:i+1] for k, v in data.items()}
            self.buffer[env_idx].add(env_data)

    def sample_tensors(self, batch_size: int, sequence_length: int, n_samples: int = 1, device="cpu"):
        # Ensure we have enough data
        if not self.buffer[0]._full and self.buffer[0]._pos <= sequence_length:
            return None

        total_sequences = batch_size * n_samples
        samples = {k: [] for k in self.buffer[0]._buf.keys()}

        for _ in range(total_sequences):
            # 1. Pick a random environment buffer
            env_idx = np.random.randint(0, self.n_envs)
            buf = self.buffer[env_idx]
            
            # 2. Pick a valid starting index
            max_idx = self.buffer_size if buf._full else buf._pos
            start_idx = np.random.randint(0, max_idx - sequence_length)
            
            # 3. Collect sequence
            for k in samples.keys():
                # samples[k].append(buf._buf[k][start_idx : start_idx + sequence_length])
                indices = np.arange(start_idx, start_idx + sequence_length) % self.buffer_size
                samples[k].append(buf._buf[k][indices])

    	# 4. Format into [n_samples, sequence_length, batch_size, Features...]
        out = {}
        for k, v in samples.items():
            # Stack into [Total_Seqs, Seq_Len, Features...]
            tensor = np.stack(v, axis=0) 
            # Reshape to [n_samples, batch_size, seq_len, Features...]
            feat_shape = tensor.shape[2:]
            tensor = tensor.reshape(n_samples, batch_size, sequence_length, *feat_shape)
            # Permute to [n_samples, seq_len, batch_size, Features...]
            out[k] = torch.as_tensor(tensor, device=device).transpose(1, 2).float()
        
        return out
class Ratio:
    def __init__(self, ratio: float, pretrain_steps: int = 0):
        if pretrain_steps < 0:
            raise ValueError(f"'pretrain_steps' must be non-negative, got {pretrain_steps}")
        if ratio < 0:
            raise ValueError(f"'ratio' must be non-negative, got {ratio}")
        self._pretrain_steps = pretrain_steps
        self._ratio = ratio
        self._prev = None

    def __call__(self, step: int):
        if self._ratio == 0:
            return 0
        if self._prev is None:
            self._prev = step
            repeats = int(step * self._ratio)
            if self._pretrain_steps > 0:
                if step < self._pretrain_steps:
                    warnings.warn(
                        "The number of pretrain steps is greater than the number of current steps. This could lead to "
                        f"a higher ratio than the one specified ({self._ratio}). Setting the 'pretrain_steps' equal to "
                        "the number of current steps."
                    )
                    self._pretrain_steps = step
                repeats = int(self._pretrain_steps * self._ratio)
            return repeats
        repeats = int((step - self._prev) * self._ratio)
        self._prev += repeats / self._ratio
        return repeats

    def state_dict(self):
        return {"_ratio": self._ratio, "_prev": self._prev, "_pretrain_steps": self._pretrain_steps}

    def load_state_dict(self, state_dict):
        self._ratio = state_dict["_ratio"]
        self._prev = state_dict["_prev"]
        self._pretrain_steps = state_dict["_pretrain_steps"]
        return self



import time
from contextlib import ContextDecorator
from typing import Dict, Optional, Type, Union

import torch
from torchmetrics import Metric, SumMetric

class TimerError(Exception):
    """A custom exception used to report errors in use of timer class"""
class timer(ContextDecorator):
    """A timer class to measure the time of a code block."""

    disabled: bool = False
    timers: Dict[str, Metric] = {}
    _start_time: Optional[float] = None

    def __init__(self, name: str, metric: Optional[Type[Metric]] = None, **kwargs) -> None:
        """Add timer to dict of timers after initialization"""
        self.name = name
        if not timer.disabled and self.name is not None and self.name not in self.timers.keys():
            self.timers.setdefault(self.name, metric(**kwargs) if metric is not None else SumMetric(**kwargs))

    def start(self) -> None:
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError("timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self) -> float:
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError("timer is not running. Use .start() to start it")

        # Calculate elapsed time
        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None

        # Report elapsed time
        if self.name:
            self.timers[self.name].update(elapsed_time)

        return elapsed_time

    @classmethod
    def to(cls, device: Union[str, torch.device] = "cpu") -> None:
        """Create a new timer on a different device"""
        if cls.timers:
            for k, v in cls.timers.items():
                cls.timers[k] = v.to(device)

    @classmethod
    def reset(cls) -> None:
        """Reset all timers"""
        for timer in cls.timers.values():
            timer.reset()
        cls._start_time = None

    @classmethod
    def compute(cls) -> Dict[str, torch.Tensor]:
        """Reduce the timers to a single value"""
        reduced_timers = {}
        if cls.timers:
            for k, v in cls.timers.items():
                reduced_timers[k] = v.compute().item()
        return reduced_timers

    def __enter__(self):
        """Start a new timer as a context manager"""
        if not timer.disabled:
            self.start()
        return self

    def __exit__(self, *exc_info):
        """Stop the context manager timer"""
        if not timer.disabled:
            self.stop()



class LabMazePlaygroundEnv(MiniGridEnv):
    """
    MiniGrid environment using LabMaze for maze generation.
    No explicit goal or reward.
    """

    def __init__(self, size = 15, max_steps=100, **kwargs):
        mission_space = MissionSpace(mission_func=self._gen_mission)
        self.size = size + 2  # full MiniGrid size INCLUDING borders
        super().__init__(
            mission_space=mission_space,
            width=self.size,
            height=self.size,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return ""

    def _gen_grid(self, width, height):
        # Create empty MiniGrid grid
        self.grid = Grid(width, height)

        # -------------------------
        # Generate LabMaze layout
        # -------------------------
        # We generate inside the border
        maze_w = width
        maze_h = height

        maze = labmaze.RandomMaze(
            height = height,  # with outer walls
            width = width,
            room_max_size=3,
            objects_per_room=1,
            max_rooms=9,
        )

        # Convert entity layer to numpy array
        # entity_layer is shape (H, W)
        maze_grid = np.array(maze.entity_layer)

        # -------------------------
        # Translate LabMaze → MiniGrid
        # -------------------------
        for y in range(maze_h):
            for x in range(maze_w):
                cell = maze_grid[y, x]

                # Wall symbols vary slightly; these are the common ones
                if cell in ("#", "*"):
                    self.grid.set(x, y, Wall())

        # -------------------------
        # Optional: random objects
        # -------------------------
        for y, x in np.argwhere(maze_grid == 'G'):
            # LabMaze uses (row, col) == (y, x)
            # y, x = obj.position

            # Skip if cell is not empty in MiniGrid
            if self.grid.get(x, y) is not None:
                continue

            color = self._rand_elem(COLOR_NAMES)
            ball = Ball(color)

            self.grid.set(x, y, ball)
    
        # -------------------------
        # Place agent
        # -------------------------
        self.place_agent()

        self.mission = ""