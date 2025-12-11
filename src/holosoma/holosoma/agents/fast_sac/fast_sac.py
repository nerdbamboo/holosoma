from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class Actor(nn.Module):
    def __init__(
        self,
        obs_indices: dict[str, dict[str, int]],
        obs_keys: list[str],
        n_act: int,
        num_envs: int,
        hidden_dim: int,
        log_std_max: float,
        log_std_min: float,
        use_tanh: bool = True,
        use_layer_norm: bool = True,
        device: torch.device | str | None = None,
        action_scale: torch.Tensor | None = None,
        action_bias: torch.Tensor | None = None,
        encoder_obs_key: str | None = None,
        encoder_obs_shape: tuple[int, int, int] | None = None,
    ):
        super().__init__()
        self.obs_indices = obs_indices
        self.obs_keys = obs_keys
        self.n_act = n_act
        self.log_std_max = log_std_max
        self.log_std_min = log_std_min
        self.use_tanh = use_tanh
        self.n_envs = num_envs
        self.device = device
        self.hidden_dim = hidden_dim
        self.use_layer_norm = use_layer_norm
        self.encoder_obs_key = encoder_obs_key
        self.encoder_obs_shape = encoder_obs_shape

        # Setup the network - this will be overridden in subclasses if needed
        self.setup_network()

        # Register action scaling parameters as buffers
        if action_scale is not None:
            self.register_buffer("action_scale", action_scale.to(device))
        else:
            self.register_buffer("action_scale", torch.ones(n_act, device=device))

        if action_bias is not None:
            self.register_buffer("action_bias", action_bias.to(device))
        else:
            self.register_buffer("action_bias", torch.zeros(n_act, device=device))

    def setup_network(self) -> None:
        """Setup the network architecture. Can be overridden by subclasses."""
        n_obs = sum(self.obs_indices[obs_key]["size"] for obs_key in self.obs_keys)
        self._setup_network_with_input_dim(n_obs)

    def _setup_network_with_input_dim(self, input_dim: int) -> None:
        """Setup network with specific input dimension."""
        self.net = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim, device=self.device),
            nn.LayerNorm(self.hidden_dim, device=self.device) if self.use_layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2, device=self.device),
            nn.LayerNorm(self.hidden_dim // 2, device=self.device) if self.use_layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4, device=self.device),
            nn.LayerNorm(self.hidden_dim // 4, device=self.device) if self.use_layer_norm else nn.Identity(),
            nn.SiLU(),
        )
        self.fc_mu = nn.Sequential(
            nn.Linear(self.hidden_dim // 4, self.n_act, device=self.device),
        )
        self.fc_logstd = nn.Linear(self.hidden_dim // 4, self.n_act, device=self.device)
        nn.init.constant_(self.fc_mu[0].weight, 0.0)
        nn.init.constant_(self.fc_mu[0].bias, 0.0)
        nn.init.constant_(self.fc_logstd.weight, 0.0)
        nn.init.constant_(self.fc_logstd.bias, 0.0)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.process_obs(obs)
        x = self.net(x)
        mean = self.fc_mu(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
            log_std + 1
        )  # From SpinUp / Denis Yarats

        if self.use_tanh:
            tanh_mean = torch.tanh(mean)
            action = tanh_mean * self.action_scale + self.action_bias
        else:
            action = mean

        return action, mean, log_std

    def get_actions_and_log_probs(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, mean, log_std = self(obs)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        raw_action = dist.rsample()

        if self.use_tanh:
            # Apply tanh to get bounded actions in [-1, 1]
            tanh_action = torch.tanh(raw_action)
            # Scale and bias to get final actions
            action = tanh_action * self.action_scale + self.action_bias

            # Compute log probability with proper Jacobian correction
            log_prob = dist.log_prob(raw_action)
            # Jacobian correction for tanh transformation
            log_prob -= torch.log(1 - tanh_action.pow(2) + 1e-6)
            # Jacobian correction for scaling transformation
            log_prob -= torch.log(self.action_scale + 1e-6)
        else:
            # Non-tanh case
            action = raw_action
            log_prob = dist.log_prob(raw_action)

        log_prob = log_prob.sum(1)
        return action, log_prob

    @torch.no_grad()
    def explore(
        self, obs: torch.Tensor, dones: torch.Tensor | None = None, deterministic: bool = False
    ) -> torch.Tensor:
        _, mean, log_std = self(obs)
        if deterministic:
            if self.use_tanh:
                tanh_mean = torch.tanh(mean)
                return tanh_mean * self.action_scale + self.action_bias
            return mean

        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        raw_action = dist.rsample()

        if self.use_tanh:
            tanh_action = torch.tanh(raw_action)
            action = tanh_action * self.action_scale + self.action_bias
        else:
            action = raw_action

        return action

    def process_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                obs[..., self.obs_indices[obs_key]["start"] : self.obs_indices[obs_key]["end"]]
                for obs_key in self.obs_keys
            ],
            -1,
        )


class CNNActor(Actor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup_network(self) -> None:
        """Setup CNN encoder and network with correct input dimensions."""
        if self.encoder_obs_shape is None:
            raise ValueError("encoder_obs_shape must be provided for CNNActor")

        # Create the CNN encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(self.encoder_obs_shape[0], 16, kernel_size=4, stride=2, padding=1, device=self.device),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=4, stride=2, padding=1, device=self.device),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate CNN output dimension using mathematical calculation
        cnn_output_dim = calculate_cnn_output_dim(self.encoder_obs_shape)

        # Calculate total input dimension: CNN features + state observations
        state_obs_dim = sum(self.obs_indices[obs_key]["size"] for obs_key in self.obs_keys)
        total_input_dim = cnn_output_dim + state_obs_dim

        # Setup the main network with the correct input dimension
        self._setup_network_with_input_dim(total_input_dim)

    def process_obs(self, obs: torch.Tensor) -> torch.Tensor:
        if self.encoder_obs_key is None or self.encoder_obs_shape is None:
            raise ValueError("encoder_obs_key and encoder_obs_shape must be provided for CNNActor")

        # Handle encoder observation
        encoder_obs = torch.cat(
            [obs[..., self.obs_indices[self.encoder_obs_key]["start"] : self.obs_indices[self.encoder_obs_key]["end"]]],
            -1,
        )
        encoder_obs = encoder_obs.view(encoder_obs.shape[0], *self.encoder_obs_shape)
        encoder_x = self.encoder(encoder_obs)

        # Handle state observations. This could include encoder obs if the user wants
        state_x = torch.cat(
            [
                obs[..., self.obs_indices[obs_key]["start"] : self.obs_indices[obs_key]["end"]]
                for obs_key in self.obs_keys
            ],
            -1,
        )

        # Concatenate CNN features with state observations
        return torch.cat([encoder_x, state_x], -1)


class QuantileQNetwork(nn.Module):
    def __init__(
        self,
        n_obs: int,
        n_act: int,
        num_quantiles: int,
        hidden_dim: int,
        use_layer_norm: bool = True,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs + n_act, hidden_dim, device=device),
            nn.LayerNorm(hidden_dim, device=device) if use_layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2, device=device),
            nn.LayerNorm(hidden_dim // 2, device=device) if use_layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4, device=device),
            nn.LayerNorm(hidden_dim // 4, device=device) if use_layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, num_quantiles, device=device),
        )
        self.num_quantiles = num_quantiles

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, actions], 1)
        x = self.net(x)
        return x  # noqa: RET504



class Critic(nn.Module):
    def __init__(
        self,
        obs_indices: dict[str, dict[str, int]],
        obs_keys: list[str],
        n_act: int,
        num_quantiles: int,
        hidden_dim: int,
        use_layer_norm: bool = True,
        num_q_networks: int = 2,
        encoder_obs_key: str | None = None,
        encoder_obs_shape: tuple[int, int, int] | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.obs_indices = obs_indices
        self.obs_keys = obs_keys
        self.n_act = n_act
        self.num_quantiles = num_quantiles
        self.hidden_dim = hidden_dim
        self.use_layer_norm = use_layer_norm
        if num_q_networks < 1:
            raise ValueError("num_q_networks must be at least 1")
        self.num_q_networks = num_q_networks
        self.encoder_obs_key = encoder_obs_key
        self.encoder_obs_shape = encoder_obs_shape
        self.device = device

        # Setup Q-networks - this will be overridden in subclasses if needed
        self.setup_qnetworks()

    def setup_qnetworks(self) -> None:
        """Setup Q-networks. Can be overridden by subclasses."""
        n_obs = sum(self.obs_indices[obs_key]["size"] for obs_key in self.obs_keys)
        self._setup_qnetworks_with_obs_dim(n_obs)

    def _setup_qnetworks_with_obs_dim(self, n_obs: int) -> None:
        """Setup Q-networks with specific observation dimension."""
        self.qnets = nn.ModuleList(
            [
                QuantileQNetwork(
                    n_obs=n_obs,
                    n_act=self.n_act,
                    num_quantiles=self.num_quantiles,
                    hidden_dim=self.hidden_dim,
                    use_layer_norm=self.use_layer_norm,
                    device=self.device,
                )
                for _ in range(self.num_q_networks)
            ]
        )

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = self.process_obs(obs)
        outputs = [qnet(x, actions) for qnet in self.qnets]
        return torch.stack(outputs, dim=0)

    def get_value(self, quantiles: torch.Tensor) -> torch.Tensor:
        """Calculate value from quantiles (mean)"""
        return torch.mean(quantiles, dim=-1)

    def process_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                obs[..., self.obs_indices[obs_key]["start"] : self.obs_indices[obs_key]["end"]]
                for obs_key in self.obs_keys
            ],
            -1,
        )


class CNNCritic(Critic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup_qnetworks(self) -> None:
        """Setup CNN encoder and Q-networks with correct input dimensions."""
        if self.encoder_obs_shape is None:
            raise ValueError("encoder_obs_shape must be provided for CNNCritic")

        # Create the CNN encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(self.encoder_obs_shape[0], 16, kernel_size=4, stride=2, padding=1, device=self.device),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=4, stride=2, padding=1, device=self.device),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate CNN output dimension using mathematical calculation
        cnn_output_dim = calculate_cnn_output_dim(self.encoder_obs_shape)

        # Calculate total input dimension: CNN features + state observations
        state_obs_dim = sum(self.obs_indices[obs_key]["size"] for obs_key in self.obs_keys)
        total_obs_dim = cnn_output_dim + state_obs_dim

        # Setup Q-networks with the correct observation dimension
        self._setup_qnetworks_with_obs_dim(total_obs_dim)

    def process_obs(self, obs: torch.Tensor) -> torch.Tensor:
        if self.encoder_obs_key is None or self.encoder_obs_shape is None:
            raise ValueError("encoder_obs_key and encoder_obs_shape must be provided for CNNCritic")

        encoder_obs = torch.cat(
            [obs[..., self.obs_indices[self.encoder_obs_key]["start"] : self.obs_indices[self.encoder_obs_key]["end"]]],
            -1,
        )
        encoder_obs = encoder_obs.view(encoder_obs.shape[0], *self.encoder_obs_shape)
        encoder_x = self.encoder(encoder_obs)

        # Handle state observations. This could include encoder obs if the user wants
        state_x = torch.cat(
            [
                obs[..., self.obs_indices[obs_key]["start"] : self.obs_indices[obs_key]["end"]]
                for obs_key in self.obs_keys
            ],
            -1,
        )

        # Concatenate CNN features with state observations
        return torch.cat([encoder_x, state_x], -1)


def calculate_cnn_output_dim(input_shape: tuple[int, int, int]) -> int:
    """
    Calculate CNN output dimension for the fixed CNN architecture.

    The CNN has the following architecture:
    1. Conv2d(channels, 16, kernel_size=4, stride=2, padding=1)
    2. Conv2d(16, 16, kernel_size=4, stride=2, padding=1)
    3. Flatten()

    Args:
        input_shape: (channels, height, width)

    Returns:
        Output dimension after flattening
    """
    channels, height, width = input_shape

    # First conv layer: Conv2d(channels, 16, kernel_size=4, stride=2, padding=1)
    h1 = (height + 2 * 1 - 4) // 2 + 1
    w1 = (width + 2 * 1 - 4) // 2 + 1

    # Second conv layer: Conv2d(16, 16, kernel_size=4, stride=2, padding=1)
    h2 = (h1 + 2 * 1 - 4) // 2 + 1
    w2 = (w1 + 2 * 1 - 4) // 2 + 1

    # Flatten: 16 channels * h2 * w2
    return 16 * h2 * w2
