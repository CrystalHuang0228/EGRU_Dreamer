import torch
from torch.distributions import Independent, OneHotCategorical, Distribution
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torch.nn.functional as F
from Utils import SymLogDistribution, TwoHotEncodingDistribution, BernoulliSafeMode, Moments
from Utils import MetricAggregator, ReplayBuffer, Ratio, timer
from Utils import LabMazePlaygroundEnv
from Utils import world_model_loss, compute_lambda_values, prepare_obs
from Dreamer import build_agent
import numpy as np
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import os
# os.environ['MUJOCO_GL'] = 'glfw' # special for Windows
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
from pathlib import Path
from torchmetrics import Metric, SumMetric
import copy


import gymnasium as gym
import shimmy
import memory_maze # Still need to import to register the envs
from gymnasium.wrappers import ResizeObservation, RecordEpisodeStatistics, TimeLimit

def make_env(env_id, seed, screen_size):
    def thunk():
        # 1. Create the environment using Gymnasium
        # 'GymV21KeepStepLimit-v0' is a special Gymnasium shim that 
        # loads old gym environments and gives them the new API.
        # env = gym.make(f"Shimmy-{env_id}", render_mode="rgb_array")
        env = shimmy.GymV21CompatibilityV0(env_id=env_id, render_mode=None)
        # env = TimeLimit(env, max_episode_steps=10) # just for testing purposes
        env = RecordEpisodeStatistics(env)
        
        # 2. Resize the pixels
        env = ResizeObservation(env, (screen_size, screen_size))

        env.action_space.seed(seed)
        return env
    return thunk

# One-step update of the agent
def train(
    world_model, actor, critic, target_critic,
    world_optimizer, actor_optimizer, critic_optimizer,
    data,
    moments, actions_dim,
    cfg, aggregator=None
):
    # Get parameters
    batch_size = cfg.algo.batch_size
    # sequence_length = cfg.algo.sequence_length
    sequence_length = data["is_first"].shape[0] 
    recurrent_state_size = cfg.algo.world_model.recurrent_model.recurrent_state_size
    num_layers = cfg.algo.world_model.recurrent_model.num_layers
    stochastic_size = cfg.algo.world_model.stochastic_size
    discrete_size = cfg.algo.world_model.discrete_size
    device = cfg.device
    
    # 1. Image preparation
    # VisualEncoder now handles normalization, but we ensure data is floats
    batch_obs = data['image'] / 255.0 - 0.5
    # []

    # 2. Time Alignment
    data["is_first"][0, :] = torch.ones_like(data["is_first"][0, :])
    batch_actions = torch.cat((torch.zeros_like(data["actions"][:1]), data["actions"][:-1]), dim=0)

    # 3. Initialization of EGRU Tuple State
    stochastic_size_flat = stochastic_size * discrete_size
    
    # Recurrent state for EGRU is (List[Membrane], List[Spikes])
    c_list = [torch.zeros(batch_size, recurrent_state_size, device=device) for _ in range(num_layers)]
    y_list = [torch.zeros(batch_size, recurrent_state_size, device=device) for _ in range(num_layers)]
    recurrent_state = (c_list, y_list)
    
    # We need this to ensure imagination starts with the correct memory at each step t
    history_c = [torch.empty(sequence_length, batch_size, recurrent_state_size, device=device) for _ in range(num_layers)]
    history_y = [torch.empty(sequence_length, batch_size, recurrent_state_size, device=device) for _ in range(num_layers)]

    # Storage for the spiking outputs (top layer) and stochastic states
    recurrent_states_y = torch.empty(sequence_length, batch_size, recurrent_state_size, device=device)
    priors_logits = torch.empty(sequence_length, batch_size, stochastic_size_flat, device=device)
    posterior = torch.zeros(1, batch_size, stochastic_size, discrete_size, device=device)
    posteriors = torch.empty(sequence_length, batch_size, stochastic_size, discrete_size, device=device)
    posteriors_logits = torch.empty(sequence_length, batch_size, stochastic_size_flat, device=device)

    # 4. Perform the recurrent state updates over the sequence length
    with torch.amp.autocast(dtype=torch.bfloat16, device_type=device):
        embedded_obs = world_model.encoder(batch_obs)

        for i in range(0, sequence_length):
            # rssm.dynamic now returns (top_y_spike, new_tuple_state, ...)
            top_y, recurrent_state, posterior, _, posterior_logits, prior_logits = world_model.rssm.dynamic(
                posterior,
                recurrent_state,
                batch_actions[i : i + 1],
                embedded_obs[i : i + 1],
                data["is_first"][i : i + 1],
            )
            recurrent_states_y[i] = top_y
            priors_logits[i] = prior_logits
            posteriors[i] = posterior
            posteriors_logits[i] = posterior_logits
            
            for l in range(num_layers):
                history_c[l][i] = recurrent_state[0][l].detach()
                history_y[l][i] = recurrent_state[1][l].detach()
        
        # 5. Compute Latent states using the top-layer spikes
        latent_states = torch.cat((posteriors.view(*posteriors.shape[:-2], -1), recurrent_states_y), -1)

        # 6. Prediction and Loss (World Model)
        reconstructed_obs = world_model.decoder(latent_states)
        po = SymLogDistribution(reconstructed_obs, dims=len(reconstructed_obs.shape[2:]))
        pr = TwoHotEncodingDistribution(world_model.reward_predictor(latent_states)[0], dims=1)
        pc = Independent(BernoulliSafeMode(logits=world_model.continue_predictor(latent_states)), 1)
        continues_targets = 1 - data["terminated"]

        priors_logits = priors_logits.view(*priors_logits.shape[:-1], stochastic_size, discrete_size)
        posteriors_logits = posteriors_logits.view(*posteriors_logits.shape[:-1], stochastic_size, discrete_size)

        world_optimizer.zero_grad(set_to_none=True)
        world_loss, kl, state_loss, reward_loss, obs_loss, continue_loss = world_model_loss(
            po=po, obs=batch_obs, pr=pr, rewards=data["rewards"], pc=pc, continue_targets=continues_targets,
            priors_logits=priors_logits, posteriors_logits=posteriors_logits,
            cfg=cfg) # Assuming world_model_loss matches cfg params
        world_loss.backward()
        world_model_grads = None
        if cfg.algo.world_model.clip_gradients > 0:
            clip_grad_norm_(world_model.parameters(), cfg.algo.world_model.clip_gradients)
        world_optimizer.step()
        
        #### Recording of firing rate
        with torch.no_grad():
            firing_rate = (recurrent_states_y > 0).float().mean()
            if aggregator:
                aggregator.update("State/egru_firing_rate", firing_rate)
        
        # ---------------------------------------------------------------------
        # Actor-Critic Imagined Trajectories
        # ---------------------------------------------------------------------
        
        # Start from the posterior states
        imagined_prior = posteriors.detach().reshape(1, -1, stochastic_size_flat)
        # We must expand the EGRU tuple state to match the AC batch size (batch * seq)
        # recurrent_state currently contains lists of [batch, dim]
        # We need lists of [batch * seq, dim]
        ac_batch_size = batch_size * sequence_length
        c_ac = [c.reshape(ac_batch_size, recurrent_state_size) for c in history_c]
        y_ac = [y.reshape(ac_batch_size, recurrent_state_size) for y in history_y]
        ac_recurrent_state = (c_ac, y_ac)

        # Current top layer spikes for the AC latent state
        top_y_ac = recurrent_states_y.detach().reshape(ac_batch_size, recurrent_state_size)
        
        imagined_latent_state = torch.cat((imagined_prior.squeeze(0), top_y_ac), -1)
        actions_tuple, _ = actor(imagined_latent_state.detach())
        actions = torch.cat(actions_tuple, dim=-1)

        imagined_trajectories = torch.empty(cfg.algo.horizon + 1, ac_batch_size, stochastic_size_flat + recurrent_state_size, device=device)
        imagined_trajectories[0] = imagined_latent_state
        
        imagined_actions = torch.empty(cfg.algo.horizon + 1, ac_batch_size, actions.shape[-1], device=device)
        imagined_actions[0] = actions
        
        # Imagine trajectories in the latent space
        for i in range(1, cfg.algo.horizon + 1):
            # rssm.inference now returns (abstract_rep, new_tuple_state)
            # Note: abstract_rep is the stochastic prior z
            # We also need the top_y spike from the recurrent model inside inference
            # To keep Networks.py minimal, we modify inference to return top_y as well
            z_prior, ac_recurrent_state = world_model.rssm.inference(imagined_prior, ac_recurrent_state, actions)
            
            # In EGRU, the top_y is the spiking output (already updated in ac_recurrent_state[1][-1])
            top_y_ac = ac_recurrent_state[1][-1]
            
            imagined_prior = z_prior.view(1, -1, stochastic_size_flat)
            imagined_latent_state = torch.cat((imagined_prior.squeeze(0), top_y_ac), -1)
            imagined_trajectories[i] = imagined_latent_state
            
            actions_tuple, _ = actor(imagined_latent_state.detach())
            actions = torch.cat(actions_tuple, dim=-1)
            imagined_actions[i] = actions


        # Given imagined trajectories, predict values, rewards and continues
        predicted_values = TwoHotEncodingDistribution(critic(imagined_trajectories)[0], dims=1).mean
        predicted_rewards = TwoHotEncodingDistribution(world_model.reward_predictor(imagined_trajectories)[0], dims=1).mean
        continues = Independent(BernoulliSafeMode(logits=world_model.continue_predictor(imagined_trajectories)), 1).mode
        true_continue = (1 - data["terminated"]).flatten().reshape(1, -1, 1)
        continues = torch.cat((true_continue, continues[1:]))

        # Compute returns: estimated lambda-values
        lambda_values = compute_lambda_values(predicted_rewards[1:], predicted_values[1:], 
                                            continues[1:] * cfg.algo.gamma, lmbda=cfg.algo.lmbda)

        # Cumulative discounts along time axis(dim=0), gammma
        with torch.no_grad():
            discount = torch.cumprod(continues * cfg.algo.gamma, dim=0) / cfg.algo.gamma

        # Actor optimization step
        actor_optimizer.zero_grad(set_to_none=True)
        policies = actor(imagined_trajectories.detach())[1] #actor distriutions

        # Normalization before advantage computation
        baseline = predicted_values[:-1]
        offset, invscale = moments(lambda_values)
        normed_lambda_values = (lambda_values - offset) / invscale
        normed_baseline = (baseline - offset) / invscale
        
        advantage = normed_lambda_values - normed_baseline
        
        
        objective = (
            torch.stack(
                [
                    p.log_prob(imgnd_act.detach()).unsqueeze(-1)[:-1]
                    for p, imgnd_act in zip(policies, torch.split(imagined_actions, actions_dim, dim=-1))
                ],
                dim=-1,
            ).sum(dim=-1) # Joint log_probs of sub_action_spaces
            * advantage.detach()
        )
        try:
            entropy = cfg.algo.actor.ent_coef * torch.stack([p.entropy() for p in policies], -1).sum(dim=-1)
        except NotImplementedError:
            entropy = torch.zeros_like(objective)
        policy_loss = -torch.mean(discount[:-1].detach() * (objective + entropy.unsqueeze(dim=-1)[:-1]))
        
        policy_loss.backward()
        actor_grads = None
        if cfg.algo.actor.clip_gradients is not None and cfg.algo.actor.clip_gradients > 0:
            actor_grads = torch.nn.utils.clip_grad_norm_(
                parameters=actor.parameters(), 
                max_norm=cfg.algo.actor.clip_gradients, 
                error_if_nonfinite=False
            )
        actor_optimizer.step()
        
        #---------------------------***************----------------------------#
        # Predict the values
        qv = TwoHotEncodingDistribution(critic(imagined_trajectories.detach()[:-1])[0], dims=1)
        predicted_target_values = TwoHotEncodingDistribution(
            target_critic(imagined_trajectories.detach()[:-1])[0], dims=1).mean

        # Critic optimization
        critic_optimizer.zero_grad(set_to_none=True)
        ## "Teacher-student model": minimize the divergence between predicted values and lambda-returns
        value_loss = -qv.log_prob(lambda_values.detach())
        ## Regularization via a target network which updates slowly
        value_loss = value_loss - qv.log_prob(predicted_target_values.detach())
        value_loss = torch.mean(value_loss * discount[:-1].squeeze(-1))

        value_loss.backward()
        critic_grads = None
        if cfg.algo.critic.clip_gradients is not None and cfg.algo.critic.clip_gradients > 0:
            critic_grads = torch.nn.utils.clip_grad_norm_(
                parameters=critic.parameters(), 
                max_norm=cfg.algo.critic.clip_gradients,
                error_if_nonfinite=False
            )
        critic_optimizer.step()

        # Log metrics
        if aggregator and not aggregator.disabled:
            aggregator.update("Loss/world_model_loss", world_loss.detach())
            aggregator.update("Loss/observation_loss", obs_loss.detach())
            aggregator.update("Loss/reward_loss", reward_loss.detach())
            aggregator.update("Loss/state_loss", state_loss.detach())
            aggregator.update("Loss/continue_loss", continue_loss.detach())
            aggregator.update("State/kl", kl.mean().detach())
            aggregator.update(
                "State/post_entropy",
                Independent(OneHotCategorical(logits=posteriors_logits.detach()), 1).entropy().mean().detach(),
            )
            aggregator.update(
                "State/prior_entropy",
                Independent(OneHotCategorical(logits=priors_logits.detach()), 1).entropy().mean().detach(),
            )
            aggregator.update("Loss/policy_loss", policy_loss.detach())
            aggregator.update("Loss/value_loss", value_loss.detach())
            if world_model_grads:
                aggregator.update("Grads/world_model", world_model_grads.mean().detach())
            if actor_grads:
                aggregator.update("Grads/actor", actor_grads.mean().detach())
            if critic_grads:
                aggregator.update("Grads/critic", critic_grads.mean().detach())

        # Reset everything
        actor_optimizer.zero_grad(set_to_none=True)
        critic_optimizer.zero_grad(set_to_none=True)
        world_optimizer.zero_grad(set_to_none=True)
    
    ### Last set of images for visualization (detach to avoid memory issues)
    return reconstructed_obs.detach(), batch_obs.detach()
    
    


def clean_state_dict(state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            # 如果键以 _orig_mod. 开头，则截掉这部分
            name = k[10:] if k.startswith('_orig_mod.') else k
            new_state_dict[name] = v
        return new_state_dict

def main(cfg):

    # Single core
    device = cfg.device
    rank = 0 # gpu_id
    world_size = 1 # gpu_count
    
    state = None
    if cfg.checkpoint.resume_from:
        state = torch.load(cfg.checkpoint.resume_from, map_location=device, weights_only=False)
        state["world_model"] = clean_state_dict(state["world_model"])
        state["actor"] = clean_state_dict(state["actor"])
        state["critic"] = clean_state_dict(state["critic"])

    # These arguments cannot be changed
    ## Unnecessary stacking of frames as Dreamer uses a recurrent model
    cfg.env.frame_stack = -1
    ## Images have to be square and of size power of 2
    if 2 ** int(np.log2(cfg.env.screen_size)) != cfg.env.screen_size:
        raise ValueError(f"The screen size must be a power of 2, got: {cfg.env.screen_size}")


    # Environment setup
    print(f"Initializing Memory Maze: {cfg.env.env_id}")
    # Create Vectorized Environments
    envs = gym.vector.SyncVectorEnv([
        make_env(
            env_id=cfg.env.env_id, 
            seed=cfg.seed + i, 
            screen_size=cfg.env.screen_size
        ) for i in range(cfg.env.num_envs)
    ])
    
    action_space = envs.single_action_space
    observation_space = envs.single_observation_space

    actions_dim = tuple([action_space.n])
    obs_keys = cfg.algo.cnn_keys.encoder # Ensure this is ["image"]

    # Build the agent Dreamerv3
    world_model, player, actor, critic, target_critic = build_agent(
        cfg=cfg,
        actions_dim=actions_dim,
        obs_space=observation_space,
        world_model_state=state["world_model"] if cfg.checkpoint.resume_from else None,
        actor_state=state["actor"] if cfg.checkpoint.resume_from else None,
        critic_state=state["critic"] if cfg.checkpoint.resume_from else None,
        target_critic_state=state["target_critic"] if cfg.checkpoint.resume_from else None,
    )
    world_model.to(device)
    player.to(device)
    actor.to(device)
    critic.to(device)
    target_critic.to(device)
    
    if cfg.checkpoint.resume_from:
        print("Performing EGRU surgery...")
        for name, param in world_model.named_parameters():
            if 'rssm.recurrent_model' in name and 'theta' in name:
                print("Performed EGRU surgery...")
                param.data.fill_(0.0)
    
    # Speedup for cuda 5090
    if hasattr(torch, 'compile'):
        world_model = torch.compile(world_model)
        actor = torch.compile(actor)
        critic = torch.compile(critic)

    # Optimizers
    world_optimizer = hydra.utils.instantiate(
        cfg.algo.world_model.optimizer, params=world_model.parameters(), _convert_="all"
    )
    actor_optimizer = hydra.utils.instantiate(cfg.algo.actor.optimizer, params=actor.parameters(), _convert_="all")
    critic_optimizer = hydra.utils.instantiate(cfg.algo.critic.optimizer, params=critic.parameters(), _convert_="all")
    if cfg.checkpoint.resume_from:
        world_optimizer.load_state_dict(state["world_optimizer"])
        actor_optimizer.load_state_dict(state["actor_optimizer"])
        critic_optimizer.load_state_dict(state["critic_optimizer"])
    

    moments = Moments(
        cfg.algo.actor.moments.decay,
        cfg.algo.actor.moments.max,
        cfg.algo.actor.moments.percentile.low,
        cfg.algo.actor.moments.percentile.high,
    )
    moments.to(device)
    if cfg.checkpoint.resume_from:
        moments.load_state_dict(state["moments"])


    # Metrics
    aggregator = None
    if not MetricAggregator.disabled:
        aggregator = hydra.utils.instantiate(cfg.metric.aggregator, _convert_="all").to(device)
    writer = SummaryWriter('/root/tf-logs') # Creates a 'runs/' folder
    
    # Local data
    buffer_size = int(cfg.buffer.size // int(cfg.env.num_envs * world_size)) if not cfg.dry_run else 2
    rb = ReplayBuffer(buffer_size, n_envs=cfg.env.num_envs)
    # if cfg.checkpoint.resume_from and cfg.buffer.checkpoint:
    #     if "rb" in state:
    #         rb = state["rb"]
    #     else:
    #         raise RuntimeError(f"Checkpoint found at {cfg.checkpoint.resume_from}, but it contains no 'rb' key.")

    # Global variables
    train_step = 0
    last_train = 0
    start_iter = ((state["iter_num"] // world_size) + 1 if cfg.checkpoint.resume_from else 1)
    ## Total number of environment steps
    policy_step = state["iter_num"] * cfg.env.num_envs if cfg.checkpoint.resume_from else 0
    last_log = state["last_log"] if cfg.checkpoint.resume_from else 0
    last_checkpoint = state["last_checkpoint"] if cfg.checkpoint.resume_from else 0
    ## Number of environment steps per iteration
    policy_steps_per_iter = int(cfg.env.num_envs * world_size)
    ## Transformation from total steps to iterations
    total_iters = int(cfg.algo.total_steps // policy_steps_per_iter) if not cfg.dry_run else 1
    ## The actual iteration where learning starts
    learning_starts = cfg.algo.learning_starts // policy_steps_per_iter if not cfg.dry_run else 0
    ## Ensure there is data in the buffer before training starts
    prefill_steps = learning_starts - int(learning_starts > 0)
    if cfg.checkpoint.resume_from:
        cfg.algo.per_rank_batch_size = state["batch_size"] // world_size
        learning_starts += start_iter
        prefill_steps += start_iter


    # Create Ratio class
    ratio = Ratio(cfg.algo.replay_ratio, pretrain_steps=cfg.algo.per_rank_pretrain_steps)
    if cfg.checkpoint.resume_from:
        ratio.load_state_dict(state["ratio"])

    # Warning for log and checkpoint every (Using world_size in logic)
    if cfg.metric.log_level > 0 and cfg.metric.log_every % policy_steps_per_iter != 0:
        print(f"Warning: log_every is not a multiple of {policy_steps_per_iter}")
    
    # Initialization
    step_data = {}
    obs, info = envs.reset(seed=cfg.seed)

    # obs shape is [num_envs, 64, 64, 3]
    # step_data needs [seq_len, num_envs, 64, 64, 3] where seq_len is 1
    if isinstance(obs, dict):
        for k in obs_keys:
            step_data[k] = obs[k][np.newaxis]
    else:
        # Map the array observation to the first key in obs_keys (e.g., 'images')
        step_data[obs_keys[0]] = obs[np.newaxis]
        
    # Standard shapes: [seq_len, num_envs, feature_dim]
    step_data["rewards"] = np.zeros((1, cfg.env.num_envs, 1))
    step_data["truncated"] = np.zeros((1, cfg.env.num_envs, 1))
    step_data["terminated"] = np.zeros((1, cfg.env.num_envs, 1))
    step_data["is_first"] = np.ones_like(step_data["terminated"])
    player.init_states()

    cumulative_per_rank_gradient_steps = 0
    
    for iter_num in range(start_iter, total_iters + 1):
        policy_step += policy_steps_per_iter

        with torch.inference_mode():
            with timer("Time/env_interaction_time", SumMetric, sync_on_compute=False):
    
                # Exploration Phase
                if iter_num <= learning_starts and cfg.checkpoint.resume_from is None:
                    # Sample from env
                    real_actions = envs.action_space.sample() 
                    
                    # Convert to One-Hot for the Replay Buffer/Model
                    # Note: We use torch_obs here just to get the device/batch size logic if needed
                    actions = np.concatenate([
                        F.one_hot(torch.as_tensor(act), act_dim).numpy()
                        for act, act_dim in zip(real_actions.reshape(len(actions_dim), -1), actions_dim)], axis=-1)
                else:
                    # Logic for Player/Agent
                    torch_obs = prepare_obs(
                        obs, 
                        device=device, 
                        cnn_keys=cfg.algo.cnn_keys.encoder, 
                        num_envs=cfg.env.num_envs
                    )

                    real_actions = actions = player.get_actions(torch_obs)
                    actions = torch.cat(actions, -1).cpu().numpy()
                    
                    # Discrete argmax for environment
                    real_actions = torch.stack([real_act.argmax(dim=-1) for real_act in real_actions], dim=-1).cpu().numpy()

                step_data["actions"] = actions.reshape((1, cfg.env.num_envs, -1))
                rb.add(step_data)

                next_obs, rewards, terminated, truncated, infos = envs.step(
                    real_actions.reshape(envs.action_space.shape)
                )
                
                # Calculate dones for the Player reset logic
                dones = np.logical_or(terminated, truncated).astype(np.uint8)

            
            step_data["is_first"] = np.zeros_like(step_data["terminated"])
            if "restart_on_exception" in infos:
                for i, agent_roe in enumerate(infos["restart_on_exception"]):
                    if agent_roe and not dones[i]:
                        last_inserted_idx = (rb.buffer[i]._pos - 1) % rb.buffer[i].buffer_size
                        rb.buffer[i]["terminated"][last_inserted_idx] = np.zeros_like(
                            rb.buffer[i]["terminated"][last_inserted_idx]
                        )
                        rb.buffer[i]["truncated"][last_inserted_idx] = np.ones_like(
                            rb.buffer[i]["truncated"][last_inserted_idx]
                        )
                        rb.buffer[i]["is_first"][last_inserted_idx] = np.zeros_like(
                            rb.buffer[i]["is_first"][last_inserted_idx]
                        )
                        step_data["is_first"][i] = np.ones_like(step_data["is_first"][i])

                
            if any(dones): # Only look at infos if at least one env finished
                if isinstance(infos, dict):
                    # 1. Path for your current output (Shimmy/Direct style)
                    if "episode" in infos:
                        # SyncVectorEnv puts statistics in arrays inside the 'episode' dict
                        # info["episode"]["r"] is usually an array of shape [num_envs]
                        for i in range(cfg.env.num_envs):
                            if dones[i]: # Only log for the specific env that finished
                                ep_rew = infos["episode"]["r"][i]
                                ep_len = infos["episode"]["l"][i]
                                if aggregator and not aggregator.disabled:
                                    aggregator.update("Rewards/rew_avg", float(ep_rew))
                                    aggregator.update("Game/ep_len_avg", float(ep_len))
                                print(f">>> Env {i} Finished: Reward={ep_rew:.2f}, Steps={ep_len}")

            # if cfg.metric.log_level > 0:
            #     # Check for the standard Gymnasium VectorEnv "final_info" key
            #     if "final_info" in infos:
            #         for i, info_dict in enumerate(infos["final_info"]):
            #             # info_dict will be None for envs that haven't finished yet
            #             if info_dict is not None and "episode" in info_dict:
            #                 ep_rew = info_dict["episode"]["r"]
            #                 ep_len = info_dict["episode"]["l"]
                            
            #                 if aggregator and not aggregator.disabled:
            #                     # We use float() because rewards might be numpy scalars
            #                     aggregator.update("Rewards/rew_avg", float(ep_rew))
            #                     aggregator.update("Game/ep_len_avg", float(ep_len))
                                
            #                 # Optional: Print to console immediately so you know it's working
            #                 print(f"Env {i} finished: Reward={ep_rew:.2f}, Length={ep_len}")
            
            # Save the real next observation
            real_next_obs = copy.deepcopy(next_obs)
            if "final_observation" in infos:
                for idx, final_obs in enumerate(infos["final_observation"]):
                    if final_obs is not None:
                            real_next_obs[idx] = final_obs
            
            if isinstance(next_obs, dict):
                for k in obs_keys:
                    step_data[k] = next_obs[k][np.newaxis]
            else:
                step_data[cfg.algo.cnn_keys.encoder[0]] = next_obs[np.newaxis]

            # next_obs becomes the new obs
            obs = next_obs

            step_data["terminated"] = terminated.reshape((1, cfg.env.num_envs, -1))
            step_data["truncated"] = truncated.reshape((1, cfg.env.num_envs, -1))
            step_data["rewards"] = rewards.reshape((1, cfg.env.num_envs, -1))

            dones_idxes = dones.nonzero()[0].tolist()
            reset_envs = len(dones_idxes)
            if reset_envs > 0:
                reset_data = {}
                for k in obs_keys:
                    reset_data[k] = (real_next_obs[dones_idxes])[np.newaxis]
                reset_data["terminated"] = step_data["terminated"][:, dones_idxes]
                reset_data["truncated"] = step_data["truncated"][:, dones_idxes]
                reset_data["actions"] = np.zeros((1, reset_envs, np.sum(actions_dim)))
                reset_data["rewards"] = step_data["rewards"][:, dones_idxes]
                reset_data["is_first"] = np.zeros_like(reset_data["terminated"])
                rb.add(reset_data, dones_idxes)

                # Reset already inserted step data
                step_data["rewards"][:, dones_idxes] = np.zeros_like(reset_data["rewards"])
                step_data["terminated"][:, dones_idxes] = np.zeros_like(step_data["terminated"][:, dones_idxes])
                step_data["truncated"][:, dones_idxes] = np.zeros_like(step_data["truncated"][:, dones_idxes])
                step_data["is_first"][:, dones_idxes] = np.ones_like(step_data["is_first"][:, dones_idxes])
                player.init_states(dones_idxes)


        # Train the agent
        if iter_num >= learning_starts and len(rb) > cfg.algo.per_rank_sequence_length:
            ratio_steps = policy_step - prefill_steps * policy_steps_per_iter
            # world_size is kept here to maintain the ratio calculation
            per_rank_gradient_steps = ratio(ratio_steps / world_size)
            
            if per_rank_gradient_steps > 0:
                local_data = rb.sample_tensors(
                    batch_size=cfg.algo.per_rank_batch_size,
                    sequence_length=cfg.algo.per_rank_sequence_length,
                    n_samples=per_rank_gradient_steps,
                    device=device
                )
                
                with timer("Time/train_time", SumMetric, sync_on_compute=False):
                    for i in range(per_rank_gradient_steps):
                        # Target network EMA update
                        if cumulative_per_rank_gradient_steps % cfg.algo.critic.per_rank_target_network_update_freq == 0:
                            tau = 1 if cumulative_per_rank_gradient_steps == 0 else cfg.algo.critic.tau
                            for cp, tcp in zip(critic.parameters(), target_critic.parameters()):
                                tcp.data.copy_(tau * cp.data + (1 - tau) * tcp.data)
                        
                        # Process batch
                        batch = {k: v[i].float() for k, v in local_data.items()}
                        # batch = {k: v.float() for k, v in local_data.items()}
                        recon_imgs, target_imgs = train(world_model=world_model,actor=actor,critic=critic,target_critic=target_critic,
                              world_optimizer=world_optimizer,actor_optimizer=actor_optimizer, critic_optimizer=critic_optimizer, 
                              data=batch,moments=moments, actions_dim=actions_dim,
                              cfg=cfg, aggregator=aggregator)
                        
                        cumulative_per_rank_gradient_steps += 1
                    train_step += world_size # Increment by world_size (1)

        # Log metrics (Local only)
        if cfg.metric.log_level > 0 and (policy_step - last_log >= cfg.metric.log_every):
            if aggregator:
                metrics = aggregator.compute()
                print(f"Step {policy_step} | Metrics: {metrics}")
                for name, value in metrics.items():
                    writer.add_scalar(name, value, policy_step)
                  
                if iter_num >= learning_starts:
                    with torch.no_grad():
                        vis_recon = (recon_imgs[:8, 0] + 0.5).clamp(0, 1)
                        vis_target = (target_imgs[:8, 0] + 0.5).clamp(0, 1)
                        vis_recon = vis_recon.permute(0, 3, 1, 2)   # Now [8, 3, 64, 64]
                        vis_target = vis_target.permute(0, 3, 1, 2) # Now [8, 3, 64, 64]
                        comparison = torch.cat([vis_target, vis_recon], dim=0) # [16, 3, 64, 64]
                        grid = torchvision.utils.make_grid(comparison, nrow=8)
                        writer.add_image("Train/Reconstruction_Quality", grid.cpu(), policy_step)
                    
                aggregator.reset()
            last_log = policy_step

        # Local Checkpoint
        if cfg.checkpoint.every > 0 and policy_step - last_checkpoint >= cfg.checkpoint.every:
            last_checkpoint = policy_step
            # state_to_save = {
            #     "world_model": world_model.state_dict(),
            #     "actor": actor.state_dict(),
            #     "critic": critic.state_dict(),
            #     "iter_num": iter_num * world_size,
            #     "rb": rb if cfg.buffer.checkpoint else None
            # }
            state_to_save = {
                "world_model": world_model.state_dict(),
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "target_critic": target_critic.state_dict(),
                "world_optimizer": world_optimizer.state_dict(),
                "actor_optimizer": actor_optimizer.state_dict(),
                "critic_optimizer": critic_optimizer.state_dict(),
                "moments": moments.state_dict(),
                "ratio": ratio.state_dict(),
                "iter_num": iter_num * world_size,
                "batch_size": cfg.algo.per_rank_batch_size * world_size,
                "last_log": last_log,
                "last_checkpoint": last_checkpoint,
            }
            log_dir = HydraConfig.get().runtime.output_dir
            checkpoint_dir = Path(log_dir) / "checkpoint"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            save_path = checkpoint_dir / f"ckpt_latest.ckpt"
            torch.save(state_to_save, save_path)

    envs.close()


# This allows Hydra to evaluate the 'torch.linspace' string in the YAML
OmegaConf.register_new_resolver("eval", eval)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def entry_point(cfg: DictConfig):
    # Convert to primitive container so it's easier to work with
    # and ensures torch tensors in the cfg are handled correctly
    main(cfg)

if __name__ == "__main__":
    entry_point()