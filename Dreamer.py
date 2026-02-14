import torch
import torch.nn as nn
from Networks import VisualEncoder, VisualDecoder, MLP, Two_Hot_MLP, RSSM, Actor, WorldModel
from EGRU import EGRU
import numpy as np
from Utils import compute_stochastic_state, create_bins
import copy



# Build Wolrd Model
## Learning compact representations and predict future states
###----------******-----------###
# Sequence model h_t = f(h_t-1, z_t-1, a_t-1)
# Encoder (posterior) Z_t ~ q(z_t|h_t, x_t)
# Dynamics predictor (prior) z_t_hat ~ p(z_t_hat|h_t)
# Reward predictor r_t_hat ~ p(r_t_hat|h_t, z_t)
# Continue flag predictor c_t_hat ~ p(c_t_hat|h_t, z_t)
# Decoder x_t_hat ~ p(x_t_hat|h_t, z_t)
###----------******-----------##


# The "body" to interact with environment
class Player(nn.Module):
    def __init__(
        self,
        encoder,
        recurrent_model,
        posterior_model,
        actor,
        actions_dim,
        num_envs,
        hidden_state_size,
        device,
        stoch_dim=32,
        discrete_dim=32
    ):
        super().__init__()
        self.encoder = encoder
        self.recurrent_model = recurrent_model
        self.posterior_model = posterior_model
        self.actor = actor
        self.actions_dim = actions_dim
        self.num_envs = num_envs
        self.hidden_state_size = hidden_state_size
        self.device = device
        self.stoch_dim = stoch_dim
        self.discrete_dim = discrete_dim
    
    def init_states(self, reset_envs=None):
        num_layers = self.recurrent_model.num_layers
        if reset_envs is None or len(reset_envs) == 0:
            self.actions = torch.zeros(1, self.num_envs, np.sum(self.actions_dim), device=self.device)
            self.stochastic_state = torch.zeros(1, self.num_envs, self.stoch_dim * self.discrete_dim, device=self.device)
            # Tuple initialization
            c_list = [torch.zeros(self.num_envs, self.hidden_state_size, device=self.device) for _ in range(num_layers)]
            y_list = [torch.zeros(self.num_envs, self.hidden_state_size, device=self.device) for _ in range(num_layers)]
            self.recurrent_state = (c_list, y_list)
        else:
            self.actions[:, reset_envs] = 0.0
            self.stochastic_state[:, reset_envs] = 0.0
            for i in range(num_layers):
                self.recurrent_state[0][i][reset_envs] = 0.0 # reset c
                self.recurrent_state[1][i][reset_envs] = 0.0 # reset y
        
    def get_actions(self, obs, greedy=False):
        embedded_obs = self.encoder(obs)
        # EGRU call
        top_y, self.recurrent_state = self.recurrent_model(
            torch.cat((self.stochastic_state.squeeze(0), self.actions.squeeze(0)), -1), 
            self.recurrent_state
        )
        posterior_logits = self.posterior_model(torch.cat((top_y, embedded_obs), -1))
        stochastic_state = compute_stochastic_state(posterior_logits, discrete=self.discrete_dim)
        
        self.stochastic_state = stochastic_state.view(1, self.num_envs, -1)
        actions, _ = self.actor(torch.cat((self.stochastic_state.squeeze(0), top_y), -1), greedy=greedy)
        self.actions = torch.cat(actions, -1).unsqueeze(0)
        return actions



# The 'brain' of the RL model
def build_agent(cfg, actions_dim, obs_space, world_model_state=None, actor_state=None, critic_state=None, target_critic_state=None):

    # Get the parameters from the config file
    world_model_cfg = cfg.algo.world_model
    actor_cfg = cfg.algo.actor
    critic_cfg = cfg.algo.critic

    # Sizes 
    recurrent_state_size = world_model_cfg.recurrent_model.recurrent_state_size
    stochastic_size_flat = world_model_cfg.stochastic_size * world_model_cfg.discrete_size
    latent_state_size = stochastic_size_flat + recurrent_state_size

    # Define models
    visual_encoder = VisualEncoder()

    recurrent_model = EGRU(
        input_size=int(sum(actions_dim) + stochastic_size_flat),
        hidden_size=world_model_cfg.recurrent_model.recurrent_state_size,
        num_layers=world_model_cfg.recurrent_model.num_layers
    )

    posterior_model_input_size = visual_encoder.output_size + recurrent_state_size
    posterior_model = MLP(
        input_dims=posterior_model_input_size,
        output_dim=stochastic_size_flat,
        hidden_sizes=world_model_cfg.posterior_model.hidden_sizes
    )

    prior_model = MLP(
        input_dims=recurrent_state_size,
        output_dim=stochastic_size_flat,
        hidden_sizes=world_model_cfg.prior_model.hidden_sizes
    )

    rssm = RSSM(
        recurrent_model=recurrent_model,
        posteriror_model=posterior_model,
        prior_model=prior_model,
        stoch_dim=world_model_cfg.stochastic_size
    )

    observation_model = VisualDecoder(input_size=latent_state_size)
    
    vmin, vmax, num_bins = world_model_cfg.reward_model.bins_param
    reward_bins = create_bins(vmin, vmax, num_bins)
    reward_model = Two_Hot_MLP(
        input_dims=latent_state_size,
        hidden_sizes=world_model_cfg.reward_model.hidden_sizes,
        bins=reward_bins
    )

    continue_model = MLP(input_dims=latent_state_size,output_dim=1, hidden_sizes=world_model_cfg.continue_model.hidden_sizes)

    world_model = WorldModel(
        encoder=visual_encoder,
        rssm=rssm,
        decoder=observation_model,
        reward_predictor=reward_model,
        continue_predictor=continue_model
    )
    
    actor = Actor(
        actions_dim=actions_dim,
        mlp_params=actor_cfg.mlp_params,
    )

    vmin, vmax, num_bins = critic_cfg.bins_param
    critic_bins = create_bins(vmin, vmax, num_bins)
    critic = Two_Hot_MLP(
        input_dims=latent_state_size,
        hidden_sizes=critic_cfg.hidden_sizes,
        bins=critic_bins
    )


    # Load the model parameters if provided
    if world_model_state:
        world_model.load_state_dict(world_model_state)
    if actor_state:
        actor.load_state_dict(actor_state)
    if critic_state:
        critic.load_state_dict(critic_state)

    # Create the Player
    player = Player(
        encoder=visual_encoder,
        recurrent_model=recurrent_model,
        posterior_model=posterior_model,
        actor=actor,
        actions_dim=actions_dim,
        num_envs=cfg.env.num_envs,
        hidden_state_size=recurrent_state_size,
        device=cfg.device,
        stoch_dim=world_model_cfg.stochastic_size,
        discrete_dim=world_model_cfg.discrete_size
    )


    # Setup target critic to stabilize training
    target_critic = copy.deepcopy(critic)
    if target_critic_state:
        target_critic.load_state_dict(target_critic_state)
    

    # Bind weights between world model and player
    for agent_p, p in zip(world_model.encoder.parameters(), player.encoder.parameters()):
        p.data = agent_p.data
    for agent_p, p in zip(world_model.rssm.recurrent_model.parameters(), player.recurrent_model.parameters()):
        p.data = agent_p.data
    for agent_p, p in zip(world_model.rssm.posteriror_model.parameters(), player.posterior_model.parameters()):
        p.data = agent_p.data
    for agent_p, p in zip(actor.parameters(), player.actor.parameters()):
        p.data = agent_p.data
    
    return world_model, player, actor, critic, target_critic