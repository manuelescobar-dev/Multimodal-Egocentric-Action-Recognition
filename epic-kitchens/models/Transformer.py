import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
import torch.utils.data as data
import math
import copy


class TransformerEncoder(nn.Module):
    def __init__(self, num_classes, model_config, **kwargs):
        super(TransformerEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=1024, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=5)
        self.dropout = nn.Dropout(model_config.dropout)
        self.fc1 = nn.Linear(1024, num_classes)
        self.classifier = nn.Sequential(self.dropout, self.fc1)

    def forward(self, x: torch.Tensor, mask = None, src_key_padding_mask = None) -> Tensor:
        output = self.transformer_encoder(x)
        output = self.classifier(output)
        return output, {}

# class MultiHeadAttention(nn.Module):
#     def __init__(self, d_model, num_heads):
#         super(MultiHeadAttention, self).__init__()
#         # Ensure that the model dimension (d_model) is divisible by the number of heads
#         assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
#         # Initialize dimensions
#         self.d_model = d_model # Model's dimension
#         self.num_heads = num_heads # Number of attention heads
#         self.d_k = d_model // num_heads # Dimension of each head's key, query, and value
        
#         # Linear layers for transforming inputs
#         self.W_q = nn.Linear(d_model, d_model) # Query transformation
#         self.W_k = nn.Linear(d_model, d_model) # Key transformation
#         self.W_v = nn.Linear(d_model, d_model) # Value transformation
#         self.W_o = nn.Linear(d_model, d_model) # Output transformation
        
#     def scaled_dot_product_attention(self, Q, K, V, mask=None):
#         # Calculate attention scores
#         attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
#         # Apply mask if provided (useful for preventing attention to certain parts like padding)
#         if mask is not None:
#             attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
#         # Softmax is applied to obtain attention probabilities
#         attn_probs = torch.softmax(attn_scores, dim=-1)
        
#         # Multiply by values to obtain the final output
#         output = torch.matmul(attn_probs, V)
#         return output
        
#     def split_heads(self, x):
#         # Reshape the input to have num_heads for multi-head attention
#         batch_size, seq_length, d_model = x.size()
#         return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
#     def combine_heads(self, x):
#         # Combine the multiple heads back to original shape
#         batch_size, _, seq_length, d_k = x.size()
#         return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
#     def forward(self, Q, K, V, mask=None):
#         # Apply linear transformations and split heads
#         Q = self.split_heads(self.W_q(Q))
#         K = self.split_heads(self.W_k(K))
#         V = self.split_heads(self.W_v(V))
        
#         # Perform scaled dot-product attention
#         attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
#         # Combine heads and apply output transformation
#         output = self.W_o(self.combine_heads(attn_output))
#         return output
    
# class PositionWiseFeedForward(nn.Module):
#     def __init__(self, d_model, d_ff):
#         super(PositionWiseFeedForward, self).__init__()
#         self.fc1 = nn.Linear(d_model, d_ff)
#         self.fc2 = nn.Linear(d_ff, d_model)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         return self.fc2(self.relu(self.fc1(x)))

# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_seq_length):
#         super(PositionalEncoding, self).__init__()
        
#         pe = torch.zeros(max_seq_length, d_model)
#         position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
        
#         self.register_buffer('pe', pe.unsqueeze(0))
        
#     def forward(self, x):
#         return x + self.pe[:, :x.size(1)]

# class EncoderLayer(nn.Module):
#     def __init__(self, num_classes, model_config, **kwargs):
#         super(EncoderLayer, self).__init__()
#         self.num_classes = num_classes
#         self.self_attn = MultiHeadAttention(1024, num_classes)
#         self.feed_forward = PositionWiseFeedForward(1024, 512)
#         self.norm1 = nn.LayerNorm(1024)
#         self.norm2 = nn.LayerNorm(1024)
#         self.dropout = nn.Dropout(0.7)
#         self.fc1 = nn.Linear(1024, num_classes)
        
#     def forward(self, x):
#         attn_output = self.self_attn(x, x, x)
#         x = self.norm1(x + self.dropout(attn_output))
#         ff_output = self.feed_forward(x)
#         x = self.norm2(x + self.dropout(ff_output))
#         x = self.fc1(x)
#         return x, {}


# class TransformerEncoder(Module):
#     r"""TransformerEncoder is a stack of N encoder layers

#     Args:
#         encoder_layer: an instance of the TransformerEncoderLayer() class (required).
#         num_layers: the number of sub-encoder-layers in the encoder (required).
#         norm: the layer normalization component (optional).

#     Examples::
#         >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
#         >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
#         >>> src = torch.rand(10, 32, 512)
#         >>> out = transformer_encoder(src)
#     """
#     __constants__ = ['norm']

#     def __init__(self, num_classes, model_config, **kwargs):
#         super(TransformerEncoder, self).__init__()
#         # self.layers = _get_clones(encoder_layer, num_layers)
#         self.layer = EncoderLayer(num_classes, model_config, **kwargs)
#         self.num_layers = 5
#         self.norm = None

#     def forward(self, src: torch.Tensor, mask = None, src_key_padding_mask = None) -> Tensor:
#         r"""Pass the input through the encoder layers in turn.

#         Args:
#             src: the sequence to the encoder (required).
#             mask: the mask for the src sequence (optional).
#             src_key_padding_mask: the mask for the src keys per batch (optional).

#         Shape:
#             see the docs in Transformer class.
#         """
#         output = src

#         # for mod in self.layers:
#         for _ in range(self.num_layers):
#             # output, attn_weights = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
#             output, _ = self.layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

#         if self.norm is not None:
#             output = self.norm(output)

#         return output, {}