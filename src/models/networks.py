import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np
import os
import math
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
#====================================Utils for model=============================================
class Dict2Obj:
    def __init__(self, dictionary):
        """Convert a dictionary to a class with attribute access"""
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, Dict2Obj(value))
            else:
                setattr(self, key, value)
def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32,
                              num_channels=in_channels,
                              eps=1e-6,
                              affine=True)
def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)
def get_timestep_embedding(timesteps, embedding_dim, diffusion=False):

    if diffusion:
        half_dim = embedding_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        emb = emb.to(device=timesteps.device)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    else:
        assert len(timesteps.shape) == 1
        """
        Create sinusoidal timestep embeddings.
        :param timesteps: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an [N x dim] Tensor of positional embeddings.
        """
        dim = embedding_dim
        max_period = 10000
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        else:
            emb = embedding
    return emb

class Attention(nn.Module):
    def __init__(self, embedding_dim):
        super(Attention, self).__init__()
        self.fc = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        # x shape: (batch_size, num_attributes, embedding_dim)
        weights = self.fc(x)  # shape: (batch_size, num_attributes, 1)
        # apply softmax along the attributes dimension
        weights = F.softmax(weights, dim=1)
        return weights
class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv1d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x,
                                            scale_factor=2.0,
                                            mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv1d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (1, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels=None,
                 conv_shortcut=False,
                 dropout=0.1,
                 temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv1d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv1d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv1d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv1d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        h = h + self.temb_proj(nonlinearity(temb))[:, :, None]
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h

class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv1d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        b, c, w = q.shape
        q = q.permute(0, 2, 1)  # b,hw,c
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)
        # attend to values
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, w)

        h_ = self.proj_out(h_)

        return x + h_
#==============================================================================================

#====================================Basic Model===============================================
class Swish(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(x) * x
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=None, od_finer=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = input_dim if output_dim is None else output_dim
        self.od_finer = od_finer

        # Main network layers defined using a loop for conciseness
        self.fc_layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim),
            *[nn.Linear(hidden_dim, hidden_dim) for _ in range(5)]
        ])

        # Output layers
        if self.od_finer:
            self.traj_segments_out = nn.Linear(hidden_dim, self.output_dim)
            # Linear layer + Norm to 0~1 for OD parameters
            self.od_finer_out = nn.Sequential(
                nn.Linear(hidden_dim, 4),  # Assuming 4 parameters for OD
                nn.Sigmoid(),
            )
        else:
            self.output = nn.Linear(hidden_dim, self.output_dim)

        self.activation = nn.SiLU()

    def forward(self, x):
        # Process through all fc layers
        features = x
        for layer in self.fc_layers:
            features = self.activation(layer(features))

        if self.od_finer:
            traj_segments = self.traj_segments_out(features)
            od_finer_paras = self.od_finer_out(features)
            # Concatenate both outputs instead of returning them separately
            return torch.cat([traj_segments, od_finer_paras], dim=1)
        else:
            return self.output(features)

class CNN(nn.Module):
    def __init__(self, input_dim: int = 2, time_dim: int = 1, hidden_dim: int = 128, kernel_size: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim

        # Time embedding
        self.time_embed = nn.Linear(time_dim, hidden_dim)

        # 1D CNN layers
        self.conv1 = nn.Conv1d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2)

        # Final projection
        self.final = nn.Linear(hidden_dim, input_dim)

        self.act = Swish()

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        original_shape = x.shape
        batch_size = x.shape[0]

        # Reshape x to [batch, channels, sequence_length]
        x = x.reshape(batch_size, self.input_dim, -1)

        # Embed time and expand
        t = t.reshape(-1, self.time_dim).float()
        t_emb = self.time_embed(t).unsqueeze(-1).expand(-1, -1, x.shape[-1])

        # Concatenate time embedding along channel dimension
        x_t = torch.cat([x, t_emb], dim=1)

        # Apply CNN layers
        h = self.act(self.conv1(x_t))
        h = self.act(self.conv2(h))
        h = self.act(self.conv3(h))

        # Final projection
        h = h.permute(0, 2, 1)  # [batch, seq, channels]
        output = self.final(h).permute(0, 2, 1)  # [batch, channels, seq]

        # Reshape back to original shape
        return output.reshape(*original_shape)

class TransformerVelocity(nn.Module):
    def __init__(self, input_dim: int = 2, time_dim: int = 1, hidden_dim: int = 128, n_heads: int = 4,
                 n_layers: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim

        # Time embedding
        self.time_embed = nn.Linear(time_dim, hidden_dim)

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 2,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        original_shape = x.shape
        batch_size = x.shape[0]

        # Reshape x to [batch, sequence_length, features]
        seq_length = x.shape[1] // self.input_dim
        x = x.reshape(batch_size, seq_length, self.input_dim)

        # Embed time
        t = t.reshape(-1, self.time_dim).float()
        t_emb = self.time_embed(t).unsqueeze(1).expand(-1, seq_length, -1)

        # Project input
        h = self.input_proj(x)

        # Add time embedding
        h = h + t_emb

        # Apply transformer
        h = self.transformer(h)

        # Project to output
        output = self.output_proj(h)

        # Reshape back to original shape
        return output.reshape(*original_shape)

class BiLSTMVelocity(nn.Module):
    def __init__(self, input_dim: int = 2, time_dim: int = 1, hidden_dim: int = 128, n_layers: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim

        # Time embedding
        self.time_embed = nn.Linear(time_dim, hidden_dim)

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # BiLSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,  # Will be bidirectional, so final output is hidden_dim
            num_layers=n_layers,
            bidirectional=True,
            batch_first=True
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        original_shape = x.shape
        batch_size = x.shape[0]

        # Reshape x to [batch, sequence_length, features]
        seq_length = x.shape[1] // self.input_dim
        x = x.reshape(batch_size, seq_length, self.input_dim)

        # Embed time
        t = t.reshape(-1, self.time_dim).float()
        t_emb = self.time_embed(t).unsqueeze(1).expand(-1, seq_length, -1)

        # Project input
        h = self.input_proj(x)

        # Add time embedding
        h = h + t_emb

        # Apply BiLSTM
        h, _ = self.lstm(h)

        # Project to output
        output = self.output_proj(h)

        # Reshape back to original shape
        return output.reshape(*original_shape)
#==============================================================================================

#====================================Unet Model================================================
class TrajUnet(nn.Module):
    def __init__(self, config):
        super().__init__()
        config = Dict2Obj(config) # dict to object
        self.config = config
        ch, out_ch, ch_mult = config.unet.ch, config.unet.out_ch, tuple(config.unet.ch_mult)
        num_res_blocks = config.unet.num_res_blocks
        attn_resolutions = config.unet.attn_resolutions
        dropout = config.unet.dropout
        in_channels = config.unet.in_channels
        if config.data.parametrized == True:
            resolution = config.data.parametrized_M
        else:
            resolution = config.data.trajectory_length
        resamp_with_conv = config.unet.resamp_with_conv

        if config.ddpm.enabled:
            num_timesteps = config.ddpm.num_diffusion_timesteps
            if config.model.type == 'bayesian':
                self.logvar = nn.Parameter(torch.zeros(num_timesteps))

        self.ch = ch
        self.temb_ch = self.ch * 4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # timestep embedding for diffusion
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch, self.temb_ch),
            nn.SiLU(), # for Flow Matching
            torch.nn.Linear(self.temb_ch, self.temb_ch),
        ])

        # downsampling
        self.conv_in = torch.nn.Conv1d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1, ) + ch_mult
        self.down = nn.ModuleList()
        block_in = None
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(in_channels=block_in,
                                out_channels=block_out,
                                temb_channels=self.temb_ch,
                                dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            skip_in = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                if i_block == self.num_res_blocks:
                    skip_in = ch * in_ch_mult[i_level]
                block.append(
                    ResnetBlock(in_channels=block_in + skip_in,
                                out_channels=block_out,
                                temb_channels=self.temb_ch,
                                dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv1d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x, t, extra_embed=None):
        if self.config.data.od_finer == True:
            assert x.shape[2] == self.resolution+2 # 4/2
        else:
            pass # assert x.shape[1] == self.resolution
        # timestep embedding
        if self.config.ddpm.enabled:
            # For Diffusion models, use sinusoidal embedding
            if t.dim() == 2:
                t = t.squeeze(1)
            temb = get_timestep_embedding(t, self.ch,self.config.ddpm.enabled)
            temb = self.temb.dense[0](temb)
            temb = nonlinearity(temb)
            temb = self.temb.dense[1](temb)
            if extra_embed is not None:
                temb = temb + extra_embed
        else:
            # For Flow Matching, use simpler time representation
            if t.dim() == 2:
                t = t.squeeze(1)
            # Linear embedding of time instead of sinusoidal embedding
            temb = get_timestep_embedding(t, self.ch,self.config.ddpm.enabled)
            temb = self.temb.dense[0](temb)
            temb = nonlinearity(temb)
            temb = self.temb.dense[1](temb)
            # Add extra embedding if provided
            if extra_embed is not None: # Test: not add t info to each layer!
                temb = temb + extra_embed

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]  # [10, 256, 4, 4]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                ht = hs.pop()
                if ht.size(-1) != h.size(-1):
                    h = torch.nn.functional.pad(h,
                                                (0, ht.size(-1) - h.size(-1)))
                h = self.up[i_level].block[i_block](torch.cat([h, ht], dim=1),
                                                    temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h
#==============================================================================================

#====================================Condition Embedding========================================
class SimpleMLPEmb(nn.Module):
    """Simple MLP for condition embedding using one-hot encoding"""

    def __init__(self, input_dim, embedding_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim

        self.o_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # Input is one-hot vector
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )

        self.d_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # Input is one-hot vector
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )

        self.combine_net = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, c, geohash=False):
        # Extract origin and destination indices from input
        o, d = c[:, 0].long(), c[:, 1].long()

        # Convert to one-hot vectors
        o_onehot = F.one_hot(o, num_classes=self.input_dim).float()
        d_onehot = F.one_hot(d, num_classes=self.input_dim).float()

        # Get embeddings from both networks
        o_embed = self.o_net(o_onehot)
        d_embed = self.d_net(d_onehot)

        # Concatenate embeddings and process them together
        combined = torch.cat([o_embed, d_embed], dim=1)
        return self.combine_net(combined)
class SimpleMLPEmb2(nn.Module):
    """Simple MLP for condition embedding"""

    def __init__(self, input_dim,embedding_dim, hidden_dim):
        super().__init__()
        self.o_net = nn.Sequential(
            nn.Embedding(input_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        self.d_net = nn.Sequential(
            nn.Embedding(input_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        self.combine_net = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
    def forward(self, c, geohash=False):
        # Extract origin and destination from input
        o, d = c[:, 0].long(), c[:, 1].long()

        # Get embeddings from both networks
        o_embed = self.o_net(o)
        d_embed = self.d_net(d)

        # Concatenate embeddings and process them together
        combined = torch.cat([o_embed, d_embed], dim=1)
        return self.combine_net(combined)

class WideAndDeep(nn.Module):
    def __init__(self, embedding_dim=128, hidden_dim=256,location_emb_dim=5000,config=None):
        super(WideAndDeep, self).__init__()

        self.encoding_bias = 0 #1
        location_emb_dim = location_emb_dim + self.encoding_bias
        config = Dict2Obj(config)
        self.config = config
        # Wide part (linear model for continuous attributes)
        self.wide_fc = nn.Linear(5, embedding_dim)
        # Deep part (neural network for categorical attributes)
        self.depature_embedding = nn.Embedding(288, hidden_dim)
        if config.data.geohash == True:
            self.sid_embedding = nn.Linear(location_emb_dim, hidden_dim)
            self.eid_embedding = nn.Linear(location_emb_dim, hidden_dim)
        else:
            self.sid_embedding = nn.Embedding(location_emb_dim, hidden_dim)
            self.eid_embedding = nn.Embedding(location_emb_dim, hidden_dim)
        if config.condition.transportation_mode == False:
            self.deep_fc1 = nn.Linear(hidden_dim*3, embedding_dim)
        else:
            tmode_number = 5 + self.encoding_bias
            self.tmode_embedding = nn.Embedding(tmode_number, hidden_dim)
            self.deep_fc1 = nn.Linear(hidden_dim*4, embedding_dim)
        self.deep_fc2 = nn.Linear(embedding_dim, embedding_dim)
        # Open-source configs omit AOI features; default them to disabled.
        if not hasattr(config.data, 'AOITYPE'):
            config.data.AOITYPE = False
        if not hasattr(config.data, 'AOIEMB'):
            config.data.AOIEMB = False
        if config.data.AOITYPE == True:
            aoitype_number = 7 + self.encoding_bias
            self.aoi_embedding = nn.Embedding(aoitype_number, hidden_dim)
            self.sid_aoi_embedding = nn.Linear(hidden_dim*2, hidden_dim)
            self.eid_aoi_embedding = nn.Linear(hidden_dim*2, hidden_dim)
        elif config.data.AOIEMB == True:
            self.aoi_emb_dim = 64
            self.sid_aoi_embedding = nn.Linear(self.aoi_emb_dim, hidden_dim)
            self.eid_aoi_embedding = nn.Linear(self.aoi_emb_dim, hidden_dim)
            pass
        else:
            pass

    def forward(self, attr, geohash=False):
        # Continuous attributes
        continuous_attrs = attr[:, 1:6]

        # Categorical attributes

        if geohash == True:
            depature = attr[:, 0].long()
            # trans attr[:, 6] and attr[:, 7] to string
            sid = attr[:, 6 : 6+self.location_emb_dim]
            eid = attr[:, 6+self.location_emb_dim:6+self.location_emb_dim*2]
        else:
            depature, sid, eid = attr[:, 0].long(
            ), attr[:, 6].long()+ self.encoding_bias, attr[:, 7].long()+ self.encoding_bias
        if self.config.data.AOITYPE == True:
            s_aoi = attr[:, 8].long()+ self.encoding_bias
            e_aoi = attr[:, 9].long()+ self.encoding_bias
        elif self.config.data.AOIEMB == True:
            s_aoi = attr[:, 8 : 8+self.aoi_emb_dim]
            e_aoi = attr[:, 8+self.aoi_emb_dim:8+self.aoi_emb_dim*2]
        else:
            pass

        # Wide part
        wide_out = self.wide_fc(continuous_attrs)

        # Deep part
        depature_embed = self.depature_embedding(depature)
        sid_embed = self.sid_embedding(sid)
        eid_embed = self.eid_embedding(eid)
        if self.config.data.AOITYPE == True:
            s_aoi_emd = self.aoi_embedding(s_aoi)
            e_aoi_emd = self.aoi_embedding(e_aoi)
            sid_embed = torch.cat((sid_embed, s_aoi_emd), dim=1)
            eid_embed = torch.cat((eid_embed, e_aoi_emd), dim=1)
            sid_embed = self.sid_aoi_embedding(sid_embed)
            eid_embed = self.eid_aoi_embedding(eid_embed)
        elif self.config.data.AOIEMB == True:
            sid_embed = self.sid_aoi_embedding(s_aoi) + sid_embed
            eid_embed = self.eid_aoi_embedding(e_aoi) + eid_embed
        else:
            pass

        if self.config.condition.transportation_mode == False:
            categorical_embed = torch.cat((depature_embed, sid_embed, eid_embed), dim=1)
        else:
            tmode = attr[:, 8].long()
            tmode_embed = self.tmode_embedding(tmode)
            categorical_embed = torch.cat((depature_embed, sid_embed, eid_embed,tmode_embed), dim=1)
        deep_out = F.relu(self.deep_fc1(categorical_embed))
        deep_out = self.deep_fc2(deep_out)
        # Combine wide and deep embeddings
        combined_embed = wide_out + deep_out

        return combined_embed
#================================================================================================

#====================================Flow Matching Model=========================================
class ConditionalVelocityModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, condition_dim=0, embedding_dim=128, dropout_prob=0.1, config=None,dataset=None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.config = config
        self.dropout_prob = dropout_prob
        self.model_type = config['model']['type']

        # Determine if conditional from config
        self.conditional = self._is_conditional(config, condition_dim)

        if self.conditional:
            # Initialize condition embedding and model
            self.condition_embedding_model = self._get_condition_embedding(config, embedding_dim, hidden_dim,dataset)
            self.condition_dropout = nn.Dropout(self.dropout_prob)
            self.net = self._build_model(input_dim + 1 + embedding_dim, hidden_dim) # for MLP default
        else:
            # Unconditional model
            self.net = self._build_model(input_dim + 1, hidden_dim)

    def _is_conditional(self, config, condition_dim):
        """Determine if model should be conditional based on config or condition_dim"""
        if config:
            if isinstance(config, dict):
                return config.get('condition', {}).get('enabled', False)
            else:
                return getattr(config, 'condition', {}).enabled if hasattr(getattr(config, 'flow_matching', {}), 'enabled') else False
        return condition_dim > 0

    def _get_condition_embedding(self, config, embedding_dim, hidden_dim,dataset):
        """Get the appropriate condition embedding module based on config"""
        embedding_type = config['condition']['embedding_type']
        location_emb_dim = dataset.location_dim

        if embedding_type == 'wide_and_deep':
            return WideAndDeep(embedding_dim=embedding_dim, hidden_dim=hidden_dim,
                               location_emb_dim=location_emb_dim, config=config)
        elif embedding_type == 'simple_mlp':
            return SimpleMLPEmb(input_dim=location_emb_dim,embedding_dim=embedding_dim, hidden_dim=hidden_dim)
        else:
            # Default to SimpleMLPEmb
            return SimpleMLPEmb(input_dim=location_emb_dim,embedding_dim=embedding_dim, hidden_dim=hidden_dim)

    def _build_model(self, input_dim, hidden_dim):
        """Build the appropriate model based on config"""
        model_type = self.model_type
        if self.config:
            if isinstance(self.config, dict):
                model_type = self.config.get('model', {}).get('type', 'mlp')
            else:
                model_type = getattr(getattr(self.config, 'model', object()), 'type', 'mlp')
        if self.config['data']['od_finer' ]== True:
            input_dim = input_dim + 4

        if model_type == 'cnn':
            return CNN(input_dim, time_dim=1, hidden_dim=hidden_dim)
        elif model_type == 'transformer':
            return TransformerVelocity(input_dim, time_dim=1, hidden_dim=hidden_dim)
        elif model_type == 'bilstm':
            return BiLSTMVelocity(input_dim, time_dim=1, hidden_dim=hidden_dim)
        elif model_type == 'unet':
            return TrajUnet(self.config)
        else:  # Default to MLP
            return MLP(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=self.input_dim,
                od_finer=self.config['data']['od_finer']
            )

    def forward(self, x, t, c=None, force_drop_ids=None):
        """Forward pass with optional condition embedding"""
        # Ensure t has correct shape
        if t.dim() == 1:
            t = t.unsqueeze(1)

        if self.model_type == 'unet':
            x = x.reshape(x.size(0), -1, 2)
            x = x.swapaxes(1, 2)
            # Process condition embedding
            c = self.condition_embedding_model(c)
            y_t = self.net(x,t,c) # (self, x, t, extra_embed=None)
            y_t = y_t.swapaxes(1, 2)
            if self.config['data']['od_finer'] == False:
                y_t = y_t
            else:
                y_t = y_t
        else:
            x = x.view(x.size(0), -1)
            if self.conditional and c is not None:
                # Apply dropout for classifier-free guidance
                if force_drop_ids is not None:
                    mask = (force_drop_ids == 0).float().unsqueeze(1)
                    c = c * mask
                elif self.training:
                    pass
                # Process condition embedding
                c = self.condition_embedding_model(c)
                # Concatenate inputs
                x_t = torch.cat([x, t, c], dim=1)
                y_t = self.net(x_t)
            else:
                # Unconditional case
                x_t = torch.cat([x, t], dim=1)
                y_t = self.net(x_t)
            # Reshape output to match input shape
            if self.config['data']['od_finer'] == False:
                y_t = y_t.view(x.size(0), self.input_dim//2, 2)
            else:
                y_t = y_t
        return y_t
# ================================================================================================
