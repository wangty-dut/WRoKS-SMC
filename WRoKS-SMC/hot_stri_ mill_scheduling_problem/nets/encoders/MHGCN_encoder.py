import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import math


class SkipConnection(nn.Module):

    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input, mask=None):
        return input + self.module(input, mask=mask)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, input_dim, embed_dim=None, val_dim=None, key_dim=None):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads

        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        if embed_dim is not None:
            self.W_out = nn.Parameter(torch.Tensor(n_heads, key_dim, embed_dim))

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        """
        :param q: queries (batch_size, n_query, input_dim)  # Query tensor
        :param h: data (batch_size, graph_size, input_dim)  # Graph data
        :param mask: mask (batch_size, n_query, graph_size) or mask of that shape (can be 2-dimensional if n_query == 1)
                     Note: positions where mask contains 1 indicate that attention is not allowed (additive attention)
        """
        if h is None:
            h = q  # If h is None, perform self-attention

        # h should have shape (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)  # Number of queries
        assert q.size(0) == batch_size  # Ensure first dimension of q matches batch_size
        assert q.size(2) == input_dim  # Ensure third dimension of q matches input_dim
        assert input_dim == self.input_dim, "Input embedding dimension error"  # Ensure input dimension is correct

        # Flatten h and q into 2D tensors
        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        # Last dimension can be used for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)  # Calculate shape
        shp_q = (self.n_heads, batch_size, n_query, -1)  # Calculate query shape

        # Calculate queries (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        # # Optionally apply mask to prevent attention to certain positions
        # if mask is not None:
        #     if not isinstance(mask, torch.Tensor):
        #         mask = torch.tensor(mask, dtype=torch.bool)  # Ensure mask is a PyTorch tensor
        #     compatibility = compatibility.masked_fill(mask[None, :, :, :].expand_as(compatibility), -1e10)

        # Calculate attention scores
        attn = F.softmax(compatibility, dim=-1)

        # Apply attention scores to values
        heads = torch.matmul(attn, V)

        # Reshape heads and apply output weights
        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)

        return out  # Return output


# Define normalization class
class Normalization(nn.Module):

    # Initialization function, parameters include embedding dimension, normalization type, whether to learn normalization parameters, and whether to track statistics
    def __init__(self, embed_dim, normalization='batch', learn_norm=True, track_norm=False):
        super(Normalization, self).__init__()

        # Select normalizer based on normalization type
        self.normalizer = {
            "layer": nn.LayerNorm(embed_dim, elementwise_affine=learn_norm),  # Layer normalization
            "batch": nn.BatchNorm1d(embed_dim, affine=learn_norm, track_running_stats=track_norm)  # Batch normalization
        }.get(normalization, None)

    # Forward propagation function
    def forward(self, input, mask=None):
        if self.normalizer:
            # Perform normalization
            return self.normalizer(
                input.view(-1, input.size(-1))
            ).view(*input.size())
        else:
            # If no normalizer, return input directly
            return input


# Define position-wise feedforward neural network class
class PositionWiseFeedforward(nn.Module):

    # Initialization function, parameters include embedding dimension and feedforward network dimension
    def __init__(self, embed_dim, feed_forward_dim):
        super(PositionWiseFeedforward, self).__init__()
        self.sub_layers = nn.Sequential(
            nn.Linear(embed_dim, feed_forward_dim, bias=True),  # Linear layer
            nn.ReLU(),  # ReLU activation function
            nn.Linear(feed_forward_dim, embed_dim, bias=True),  # Linear layer
        )

        self.init_parameters()  # Initialize parameters

    # Parameter initialization function
    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    # Forward propagation function
    def forward(self, input, mask=None):
        return self.sub_layers(input)


# Define multi-head attention layer class
class MultiHeadAttentionLayer(nn.Module):
    """Implement a configurable Transformer layer

    References:
        - W. Kool, H. van Hoof, and M. Welling. Attention, learn to solve routing problems! In International Conference on Learning Representations, 2019.
        - M. Deudon, P. Cournut, A. Lacoste, Y. Adulyasak, and L.-M. Rousseau. Learning heuristics for the tsp by policy gradient. In International Conference on the Integration of Constraint Programming, Artificial Intelligence, and Operations Research, pages 170–181. Springer, 2018.
    """

    # Initialization function, parameters include number of heads, embedding dimension, feedforward network dimension, normalization type, etc.
    def __init__(self, n_heads, embed_dim, feed_forward_dim,
                 norm='batch', learn_norm=True, track_norm=False):
        super(MultiHeadAttentionLayer, self).__init__()

        # Define skip-connection multi-head attention layer
        self.self_attention = SkipConnection(
            MultiHeadAttention(
                n_heads=n_heads,
                input_dim=embed_dim,
                embed_dim=embed_dim
            )
        )
        # Define first normalization layer
        self.norm1 = Normalization(embed_dim, norm, learn_norm, track_norm)

        # Define skip-connection feedforward network layer
        self.positionwise_ff = SkipConnection(
            PositionWiseFeedforward(
                embed_dim=embed_dim,
                feed_forward_dim=feed_forward_dim
            )
        )
        # Define second normalization layer
        self.norm2 = Normalization(embed_dim, norm, learn_norm, track_norm)

    # Forward propagation function
    def forward(self, h, mask):
        h = self.self_attention(h, mask=mask)  # Pass through self-attention layer
        h = self.norm1(h, mask=mask)  # Pass through first normalization layer
        h = self.positionwise_ff(h, mask=mask)  # Pass through feedforward network layer
        h = self.norm2(h, mask=mask)  # Pass through second normalization layer
        return h


# Define graph attention encoder class
class GraphAttentionEncoder(nn.Module):

    # Initialization function, parameters include number of layers, number of heads, hidden dimension, normalization type, etc.
    def __init__(self, n_layers, n_heads, hidden_dim, norm='batch',
                 learn_norm=True, track_norm=False, *args, **kwargs):
        super(GraphAttentionEncoder, self).__init__()

        feed_forward_hidden = hidden_dim * 4  # Hidden layer dimension of feedforward network

        # Define multiple multi-head attention layers
        self.layers = nn.ModuleList([
            MultiHeadAttentionLayer(n_heads, hidden_dim, feed_forward_hidden, norm, learn_norm, track_norm)
            for _ in range(n_layers)
        ])

    # Forward propagation function
    def forward(self, x, graph):
        for layer in self.layers:
            x = layer(x, graph)  # Pass through each multi-head attention layer
        return x