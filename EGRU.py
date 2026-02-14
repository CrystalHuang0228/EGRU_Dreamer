import torch
import torch.nn as nn
import torch.nn.functional as F

class SpikeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, membrane, theta):
        ctx.save_for_backward(membrane, theta)
        return torch.where(membrane >= theta, membrane, torch.zeros_like(membrane))

    @staticmethod
    def backward(ctx, grad_output):
        membrane, theta = ctx.saved_tensors
        
        lamb = 0.5  # steepness parameter
        epsilon = 1
        
        surrogate_grad = lamb * torch.relu(1 - torch.abs(membrane) / epsilon)
        
        grad_membrane = grad_output * surrogate_grad
        grad_theta = -grad_membrane

        return grad_membrane, grad_theta
    

class EGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, thr_mean=0.3):
        super(EGRUCell, self).__init__()
        self.input_size = input_size ### X_t
        self.hidden_size = hidden_size ### Y_t-1
        
        ### Parameters for Update gate 
        self.W_u = nn.Linear(input_size + hidden_size, hidden_size)
        
        ### Parameters for Reset gate
        self.W_r = nn.Linear(input_size + hidden_size, hidden_size)
        
        ### Parameters for State candidate
        self.W_z = nn.Linear(input_size + hidden_size, hidden_size)
        
        ### Threshold parameter
        beta = 3
        alpha = beta * thr_mean / (1 - thr_mean)
        distribution = torch.distributions.beta.Beta(alpha, beta)
        self.theta = nn.Parameter(distribution.sample(torch.Size([self.hidden_size])))
        
        self.leak = 0.95 
        
        
    def forward(self, X_curr, C_prev, Y_prev):
        # print('X_curr size:', X_curr.size())
        # print('Y_prev size:', Y_prev.size())
        combined = torch.cat((X_curr, Y_prev), dim=1) ### check the concatenation dim later
        
        ### update gate
        u = torch.sigmoid(self.W_u(combined))
        ### reset gate
        r = torch.sigmoid(self.W_r(combined))
        ### state candidate
        h_prev = torch.mul(r, Y_prev)
        combined_z = torch.cat((X_curr, h_prev), dim=1)
        z = torch.tanh(self.W_z(combined_z))
        
        ###------**** Unique EGRU part ****------###
        safe_theta = F.softplus(self.theta) 
        C_curr = u * z + (1 - u) * self.leak * C_prev - Y_prev
        # Y_curr = torch.where(C_curr >= self.theta, C_curr, torch.zeros_like(C_curr)) ### Gradient 0
        Y_curr = SpikeFunction.apply(C_curr, safe_theta)  ### Surrogate Gradient
        Y_curr = torch.relu(Y_curr) 
        
    
        return C_curr, Y_curr

class EGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(EGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cells = nn.ModuleList([
            EGRUCell(input_size if i == 0 else hidden_size, hidden_size) 
            for i in range(num_layers)
        ])
        
    def forward(self, x, state=None):
        """
        x: [Batch, Dim] (Interaction) or [Steps, Batch, Dim] (Training)
        state: tuple(List[c], List[y])
        """
        # Handle 2D vs 3D inputs
        if x.dim() == 2:
            x = x.unsqueeze(0)
            is_seq = False
        else:
            is_seq = True
        
        # print(x.size())
        steps, batch_size, _ = x.size()
        
        # Initialize zero state if none provided
        if state is None:
            c_curr = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in range(self.num_layers)]
            y_curr = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in range(self.num_layers)]
        else:
            c_curr, y_curr = state

        # print('Recurrent_state_size:', y_curr[0].shape)
        layer_outputs = []
        for t in range(steps):
            inp = x[t]
            new_c = []
            new_y = []
            for i, cell in enumerate(self.cells):
                ci, yi = cell(inp, c_curr[i], y_curr[i])
                new_c.append(ci)
                new_y.append(yi)
                inp = yi # Next layer input is current layer spike
            c_curr, y_curr = new_c, new_y
            layer_outputs.append(y_curr[-1].unsqueeze(0)) # Store top-layer spike

        output = torch.cat(layer_outputs, dim=0)
        if not is_seq:
            output = output.squeeze(0)
            
        return output, (c_curr, y_curr)