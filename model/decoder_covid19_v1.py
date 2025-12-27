import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch_scatter import scatter_add, scatter_softmax
import math
from sparsemax import Sparsemax
from entmax import entmax15

def sample_gumbel(shape, eps=1e-10):

    U = torch.rand(shape).float()
    return -torch.log(eps - torch.log(U + eps))

class MaskedMultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, dual_attention=True):
        super(MaskedMultiHeadAttentionLayer, self).__init__()
        self.in_dim = in_dim
        # out_dim is the dimension per head
        self.d_k = out_dim 
        self.num_heads = num_heads
        self.dual_attention = dual_attention
        
        # Total dimension for all heads
        total_out_dim = self.d_k * self.num_heads

        # Linear projections for Q, K, V.
        # self.W_qkv = nn.Linear(in_dim, total_out_dim * 3, bias=True)
        self.W_q = nn.Linear(in_dim, total_out_dim, bias=True) 
        self.W_k = nn.Linear(in_dim, total_out_dim, bias=True) 
    
        # 独立的 V 投影
        self.W_v_in = nn.Linear(in_dim, total_out_dim, bias=True)
        self.W_v_out = nn.Linear(in_dim, total_out_dim, bias=True)
        # Final output projection
        self.W_i = nn.Linear(total_out_dim, out_dim, bias=True)
        if self.dual_attention:
            self.W_o = nn.Linear(total_out_dim, out_dim, bias=True)

    def forward(self, h, g):
        """
        h: Input node features, shape (batch_size, num_nodes, in_dim)
        g: Adjacency matrix for masking, shape (batch_size, num_nodes, num_nodes)
        """
        B, N, _ = h.shape


        q = self.W_q(h)
        k = self.W_k(h)
        
        q = q.view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算 Attention Scores
        attention_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        # mask = g.unsqueeze(0).unsqueeze(0) # Shape: (B, 1, N, N)
        mask = g.unsqueeze(1) 
        
        attention_weights = F.softplus(attention_weights)
        
        attention_weights = attention_weights * mask
        v_in = self.W_v_in(h)
        v_in = v_in.view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        output_in = torch.matmul(attention_weights, v_in)
        output_in = output_in.transpose(1, 2).contiguous().view(B, N, -1)
        output_in = self.W_i(output_in)
        if self.dual_attention:
            v_out = self.W_v_out(h)
            v_out = v_out.view(B, N, self.num_heads, self.d_k).transpose(1, 2)
            attention_weights_transposed = attention_weights.transpose(-2, -1)
            output_out = torch.matmul(attention_weights_transposed, v_out)
            output_out = output_out.transpose(1, 2).contiguous().view(B, N, -1)
            output_out = self.W_o(output_out)
            
            return output_in, output_out
        
        return output_in, None


class MaskedGraphTransformerLayer(nn.Module):

    def __init__(self, in_dim, out_dim, num_heads, dropout=0.0, dual_attention=True, gates=False):
        super(MaskedGraphTransformerLayer, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.dual_attention = dual_attention
        self.gates = gates

        self.attention = MaskedMultiHeadAttentionLayer(self.in_dim, self.out_dim, self.num_heads, self.dual_attention)
        if self.dual_attention:
            self.O = nn.Sequential(
            nn.Linear(3*self.out_dim, self.out_dim),
            nn.Softplus(),
            )
        else:
            self.O = nn.Sequential(
            nn.Linear(2*self.out_dim, self.out_dim),
            nn.Softplus(),
            )
        self.in_fc = nn.Linear(self.out_dim, self.out_dim)
        self.out_fc = nn.Linear(self.out_dim, self.out_dim)
        self.batch_norm1 = nn.LayerNorm(self.out_dim)
        # FFN
        self.FFN_layer1 = nn.Linear(self.out_dim, self.out_dim * 2)
        self.FFN_layer2 = nn.Linear(self.out_dim * 2, self.out_dim)

        self.batch_norm2 = nn.LayerNorm(self.out_dim)

    def forward(self, h, g):

        h_res1 = h

        output_in, output_out = self.attention(h, g)  

        gate_output = torch.cat((h_res1, output_in, output_out), dim=-1)
        h = gate_output

        h = self.O(h)
        h = self.batch_norm1(h)
        h_res2 = h
        # FFN
        h = self.FFN_layer1(h)
        h = F.softplus(h)
        if self.dropout > 0:
            h = F.dropout(h, self.dropout, training=self.training)

        h = self.FFN_layer2(h)
        h = F.softplus(h)
        if self.dropout > 0:
            h = F.dropout(h, self.dropout, training=self.training)

        # if self.residual:
        h = h_res2 + h  # residual connection
        h = self.batch_norm2(h)

        return h


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()

        in_channels = args.in_channels
        out_channels = 1
        hidden_channels = args.hidden_channels
        
        self.dropout = args.dropout
        self.preds = 1
        
        self.transformer_layers = nn.ModuleList(
            [MaskedGraphTransformerLayer(hidden_channels,hidden_channels, 4,
                                         dropout=self.dropout,
                                         dual_attention=True, gates=False) for _ in range(1)])
        
        self.input_fc = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Softplus(),
            )
        
        self.norm_input = nn.LayerNorm(hidden_channels)
        
        self.out_fc1 = nn.Linear(2*hidden_channels, hidden_channels)
        self.out_fc2 = nn.Linear(hidden_channels, hidden_channels)
        self.out_fc3 = nn.Linear(hidden_channels, out_channels)

    def single_step_forward(self, x, g):

        # x0 = x.clone()

        h_base = self.input_fc(x) 

        h_res = h_base.clone() 
        
        # 注意：这里需要修改 Transformer 接收两个输入
        h_agg = self.transformer_layers[0](h_base, g)
        
        all_info = torch.cat((h_res, h_agg), dim=-1)
        # Output mlp
        pred = F.dropout(F.softplus(self.out_fc1(all_info)), p=self.dropout, training=self.training)
        pred = F.dropout(F.tanh(self.out_fc2(pred)), p=self.dropout, training=self.training)
        pred = self.out_fc3(pred)
        return pred

    def forward(self, inputs, g):
        last_pred = inputs[:,:,0,:]
        # print(f"last_pred shape:{last_pred.shape}")

        pred = self.single_step_forward(last_pred, g)
        
        # print(f"preds shape:{pred.shape}")
        return pred