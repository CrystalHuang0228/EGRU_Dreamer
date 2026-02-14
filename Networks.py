import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Independent, OneHotCategoricalStraightThrough
from Utils import symlog, symexp, compute_stochastic_state


def cnn_forward(model, x, input_shape, output_shape):
    # x: [d1, ..., dn, C, H, W]
    # input_shape: (C, H, W)
    # output_shape: (-1,) or (OutputSize,)
    
    batch_dims = x.shape[:-len(input_shape)]
    # Flatten all leading dimensions into one batch dimension
    x = x.reshape(-1, *input_shape)
    # Apply CNN
    res = model(x)
    # Restore leading dimensions: [d1, ..., dn, *output_shape]
    return res.reshape(*batch_dims, *res.shape[1:])

class VisualEncoder(nn.Module):
    def __init__(self, keys=["image"], input_channels=[3], output_size=256):
        super().__init__()
        self.keys = keys
        self.input_channels = sum(input_channels)
        self.output_size = output_size

        # CNN stages assume 64x64 input
        self.cnn = nn.Sequential(
            nn.Conv2d(self.input_channels, 32, 4, stride=2, padding=1), # 32x32
            nn.ELU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), # 16x16
            nn.ELU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), # 8x8
            nn.ELU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), # 4x4
            nn.ELU(),
            nn.Flatten(-3, -1) # Flatten spatial dims
        )
        self.fc = nn.Linear(256 * 4 * 4, output_size)

    def forward(self, obs):
        # 1. Concatenate multiple image keys along the channel dimension
        # If obs is Tensor, we wrap it in a dict for consistency
        if not isinstance(obs, dict):
            obs = {self.keys[0]: obs}
        
        # 2. Check each image for channel ordering and concatenate
        imgs = []
        for k in self.keys:
            img = obs[k]
            # If [..., H, W, C], permute to [..., C, H, W]
            if img.shape[-1] == 3 or img.shape[-1] == 1: # Common channel counts
                img = img.permute(*range(img.ndim - 3), -1, -3, -2)
            imgs.append(img)
        
        x = torch.cat(imgs, dim=-3) # Concat on C

        # 4. Use squash-and-unsquash logic
        # Input shape to CNN is (C, H, W)
        spatial_shape = x.shape[-3:]
        
        # Define a lambda to run the whole model pipeline
        model_pipeline = lambda t: self.fc(self.cnn(t))
        
        return cnn_forward(model_pipeline, x, spatial_shape, (self.output_size,))


class VisualDecoder(nn.Module):
    def __init__(self, input_size=256, output_channels=3):
        super().__init__()
        self.input_size = input_size
        self.output_channels = output_channels

        self.fc = nn.Linear(input_size, 256 * 4 * 4)

        self.decoder = nn.Sequential(
            # Start from 256 channels at 4x4
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), # 8x8
            nn.ELU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 16x16
            nn.ELU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # 32x32
            nn.ELU(),
            nn.ConvTranspose2d(32, output_channels, 4, stride=2, padding=1), # 64x64
            nn.Tanh() # Pixels in [-1, 1] or use Sigmoid for [0, 1]
        )

    def forward(self, latent):
        # latent shape: [..., LatentSize]
        
        batch_dims = latent.shape[:-1]
        # print("Latent size:", latent.shape)
        x = latent.reshape(-1, self.input_size)
        
        # Map to spatial volume
        x = self.fc(x)
        x = x.reshape(-1, 256, 4, 4)
        
        # Apply Deconvs
        x = self.decoder(x) # [B_flattened, C, H, W]
        reconstruction = x * 0.5 # Align with the preprocessing of orignal obs [-0.5, 0.5]
        
        # Restore leading dimensions: [..., C, H, W]
        out_shape = (self.output_channels, 64, 64)
        out = reconstruction.reshape(*batch_dims, *out_shape)
        return torch.moveaxis(out, -3, -1).contiguous() 
    

# An encapsulated MLP for multiple networks
class MLP(nn.Module):
    def __init__(self, input_dims, output_dim, hidden_sizes):
        super().__init__()
        layers = []
        input_size = input_dims
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ELU())
            input_size = hidden_size
        layers.append(nn.Linear(input_size, output_dim))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
        
# Two-hot MLP for reward and Critic networks
# From a discrete distribution of categorical variables
class Two_Hot_MLP(nn.Module):
    def __init__(self, input_dims, hidden_sizes, bins):
        super().__init__()
        layers = []
        input_size = input_dims
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ELU())
            input_size = hidden_size

        num_bins = bins.shape[0]
        layers.append(nn.Linear(input_size, num_bins))
        self.model = nn.Sequential(*layers)

        ## Store the bins info without training on it
        self.register_buffer("bins", bins)
    
    def forward(self, x):
        logits = self.model(x)
        
        ## Compute the expectation value (used only for inference)
        probs = F.softmax(logits, dim=-1)
        value = torch.sum(logits * self.bins, dim=-1, keepdim=True)

        return logits, value
        

class RSSM(nn.Module):
    def __init__(self, recurrent_model, posteriror_model, prior_model, stoch_dim=32):
        super().__init__()
        self.recurrent_model = recurrent_model
        self.posteriror_model = posteriror_model
        self.prior_model = prior_model
        self.stoch_dim = stoch_dim
        # self.initial_recurrent_state = nn.Parameter(torch.zeros(recurrent_model.recurrent_state_size, dtype=torch.float32))
    
    def _reset_egru_state(self, state, is_first):
        """Helper to zero out membrane and spikes when is_first is 1"""
        if state is None: return None
        
        # state is (List[c], List[y])
        c_list, y_list = state
        
        # Force is_first into [Batch, 1] regardless of leading dimensions
        # This turns [1, 4, 1] or [1, 1, 4, 1] into [4, 1]
        batch_size = c_list[0].shape[0]
        mask = 1.0 - is_first.reshape(batch_size, 1) 
        
        new_c = [c * mask for c in c_list]
        new_y = [y * mask for y in y_list]
        return (new_c, new_y)

    # def get_initial_states(self, batch_shape):
    #     initial_recurrent_state = torch.tanh(self.initial_recurrent_state).expand(*batch_shape, -1)
    #     initial_posterior = self._prior(initial_recurrent_state, sample_state=False)[1]
    #     return initial_recurrent_state, initial_posterior
    
    def dynamic(self, posterior, recurrent_state, action, encoded_input, is_first):
        
        recurrent_state = self._reset_egru_state(recurrent_state, is_first)
        action = (1.0 - is_first) * action
        posterior = posterior.view(*posterior.shape[:-2], -1)
        posterior = (1.0 - is_first) * posterior
        
        top_y, recurrent_state = self.recurrent_model(
            torch.cat((posterior, action), -1), 
            recurrent_state
        )

        prior_logits, prior = self._prior(top_y)
        posterior_logits, posterior_out = self._posterior(top_y, encoded_input)

        return top_y, recurrent_state, posterior_out, prior, posterior_logits, prior_logits

    def inference(self, prior, recurrent_state, action):
        # recurrent_state is tuple
        top_y, recurrent_state = self.recurrent_model(
            torch.cat((prior.squeeze(0), action.squeeze(0)), -1), 
            recurrent_state
        )
        _, abstract_rep = self._prior(top_y)
        return abstract_rep.unsqueeze(0), recurrent_state
    
    def _prior(self, recurrent_state):
        logits = self.prior_model(recurrent_state)
        return logits, compute_stochastic_state(logits, discrete=self.stoch_dim)
    
    def _posterior(self, recurrent_state, encoded_input):
        logits = self.posteriror_model(torch.cat((recurrent_state, encoded_input), -1))
        return logits, compute_stochastic_state(logits, discrete=self.stoch_dim)



class Actor(nn.Module):
    def __init__(self, actions_dim, mlp_params):
        super().__init__()
        input_dim, output_dim, hidden_sizes = mlp_params
        
        # shared MLP
        self.model = MLP(input_dim, output_dim, hidden_sizes)
        # for each action
        self.mlp_heads = nn.ModuleList([nn.Linear(output_dim, action) for action in actions_dim])
        
    def forward(self, state, greedy=False):
        out = self.model(state)
        pre_dist = [head(out) for head in self.mlp_heads]

        actions_dist = []
        actions = []
        for logits in pre_dist:
            actions_dist.append(OneHotCategoricalStraightThrough(logits=logits))
            if not greedy:
                actions.append(actions_dist[-1].rsample())
            else:
                actions.append(actions_dist[-1].mode)

        return tuple(actions), tuple(actions_dist)
    

class WorldModel(nn.Module):
    def __init__(self, encoder, rssm, decoder, reward_predictor, continue_predictor):
        super().__init__()
        self.encoder = encoder
        self.rssm = rssm # Including sequence model, prior and posterior
        self.decoder = decoder
        self.reward_predictor = reward_predictor
        self.continue_predictor = continue_predictor