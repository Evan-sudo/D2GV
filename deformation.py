import torch
import torch.nn as nn
import torch.nn.functional as F


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, input_dims):
    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


class DeformNetwork(nn.Module):
    def __init__(self, D=2, W=256, pos_multires=10, time_multires=6):
        super(DeformNetwork, self).__init__()

        # Positional Embedding for position (2D) and time (1D)
        self.embed_pos, self.pos_dim = get_embedder(pos_multires, input_dims=2)
        self.embed_time, self.time_dim = get_embedder(time_multires, input_dims=1)

        # Input dimension to the MLP
        self.input_ch = self.pos_dim + self.time_dim
        self.D = D
        self.W = W

        # Define the layers of the MLP
        self.linear = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] +  # First layer
            [nn.Linear(W, W) for _ in range(D - 1)]  # Remaining layers
        )

        # # Separate branches for xyz, cholesky, and features_dc
        # self.xyz_branch = nn.Linear(W, 2)       # Output xyz deformation
        # self.opacity_branch = nn.Linear(W, 1)  # Output cholesky deformation
        # self.features_branch = nn.Linear(W, 3)  # Output features_dc deformation
        
        # Separate branches for xyz, opacity, and features_dc
        self.xyz_branch = nn.Sequential(
            nn.Linear(W, W),  # W -> W
            nn.ReLU(),
            nn.Linear(W, 2)   # W -> 2
        )

        self.opacity_branch = nn.Sequential(
            nn.Linear(W, W),  # W -> W
            nn.ReLU(),
            nn.Linear(W, 1)   # W -> 1
        )

        self.features_branch = nn.Sequential(
            nn.Linear(W, W),  # W -> W
            nn.ReLU(),
            nn.Linear(W, 3)   # W -> 3
        )

    def forward(self, pos, time):
        pe_pos = self.embed_pos(pos)  # Shape: (N, pos_dim)
        pe_time = self.embed_time(time)  # Shape: (N, time_dim)

        # Concatenate position and time embeddings
        h = torch.cat([pe_pos, pe_time], dim=-1)  # Shape: (N, input_ch)

        # Pass through the MLP
        for i, layer in enumerate(self.linear):
            h = layer(h)
            h = F.relu(h)

        # Pass through separate branches
        deformed_xyz = self.xyz_branch(h)
        deformed_opacity = self.opacity_branch(h)
        deformed_features = self.features_branch(h)

        return deformed_xyz, deformed_opacity, deformed_features
    
    

