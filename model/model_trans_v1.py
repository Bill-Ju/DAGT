import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch_scatter import scatter_add
import math
from sparsemax import Sparsemax

def sample_gumbel(shape, eps=1e-10):

    U = torch.rand(shape).float()
    return -torch.log(eps - torch.log(U + eps))

def get_edge_prob(logits, gumbel_noise, beta=0.5, hard=False):
    if gumbel_noise:
        y = logits + sample_gumbel(logits.size()).to(logits.device)
    else:
        y = logits
    edge_prob_soft = torch.softmax(beta * y, dim=0)
    if hard:
        _, edge_prob_hard = torch.max(edge_prob_soft.data, dim=0)
        edge_prob_hard = F.one_hot(edge_prob_hard)
        edge_prob_hard = edge_prob_hard.permute(1,0)
        edge_prob = edge_prob_hard - edge_prob_soft.data + edge_prob_soft
    else:
        edge_prob = edge_prob_soft
    return edge_prob

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
        self.W_qk = nn.Linear(in_dim, total_out_dim * 2, bias=True) 
        
        self.W_v_in = nn.Linear(in_dim, total_out_dim, bias=True)

        # Final output projection
        self.W_i = nn.Linear(total_out_dim, out_dim, bias=True)
        if self.dual_attention:
            self.W_v_out = nn.Linear(in_dim, total_out_dim, bias=True)
            self.W_o = nn.Linear(total_out_dim, out_dim, bias=True)

    def forward(self, h, g):
        """
        h: Input node features, shape (batch_size, num_nodes, in_dim)
        g: Adjacency matrix for masking, shape (batch_size, num_nodes, num_nodes)
        """
        B, N, _ = h.shape # B: batch_size, N: num_nodes

 
        qk = self.W_qk(h)
        q, k = torch.chunk(qk, 2, dim=-1)
        q = q.view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        
        # 4. Scaled Dot-Product
        attention_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        mask = g.unsqueeze(0).unsqueeze(0) # Shape: (B, 1, N, N)
        
        # attention_weights = F.softmax(attention_weights, dim=-1)   
        attention_weights = F.softplus(attention_weights)
        attention_weights = attention_weights * mask
        
        v_in = self.W_v_in(h)
        v_in = v_in.view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        output_in = torch.matmul(attention_weights, v_in)
        # print(f"output{output.shape}")
        # 8. Concatenate heads and apply final linear layer
        output_in = output_in.transpose(1, 2).contiguous().view(B, N, -1)
        output_in = self.W_i(output_in)
        if self.dual_attention:
            attention_weights_transposed = attention_weights.transpose(-2, -1)
            v_out = self.W_v_out(h)
            v_out = v_out.view(B, N, self.num_heads, self.d_k).transpose(1, 2)
            output_out = torch.matmul(attention_weights_transposed, v_out)
            output_out = output_out.transpose(1, 2).contiguous().view(B, N, -1)            
            output_out = self.W_o(output_out)
            return output_in, output_out
        
        return output_in, None


class MaskedGraphTransformerLayer(nn.Module):

    def __init__(self, in_dim, out_dim, num_heads, dropout=0.0, dual_attention=True, gates=True):
        super(MaskedGraphTransformerLayer, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.dual_attention = dual_attention
        self.gates = gates

        self.attention = MaskedMultiHeadAttentionLayer(self.in_dim, self.out_dim, self.num_heads, self.dual_attention)
        if self.dual_attention:
            if self.gates:
                self.gate_mlp = nn.Sequential(
                nn.Linear(self.out_dim * 3, self.out_dim),
                nn.Softplus(),
                nn.Linear(self.out_dim, self.out_dim),
                nn.Softplus()
                )
                self.gate_norm = nn.LayerNorm(self.out_dim)
            self.O = nn.Sequential(
                nn.Linear(self.out_dim * 3, self.out_dim),
                nn.Softplus(),
                )
            
        else:
            self.O = nn.Sequential(
                nn.Linear(self.out_dim * 2, self.out_dim),
                nn.Softplus(),
                )

        # self.batch_norm1 = nn.BatchNorm1d(self.num_nodes)
        self.batch_norm1 = nn.LayerNorm(self.out_dim)
        # FFN
        self.FFN_layer1 = nn.Linear(self.out_dim, self.out_dim * 2)
        self.FFN_layer2 = nn.Linear(self.out_dim * 2, self.out_dim)

        # self.batch_norm2 = nn.BatchNorm1d(self.num_nodes)
        self.batch_norm2 = nn.LayerNorm(self.out_dim)

    def forward(self, h, g):

        h_res1 = h.clone()  # for first residual connection, shape as self.hid_dim

        if self.dual_attention:
            output_in, output_out = self.attention(h, g)
            if self.gates:
                gate_input = torch.cat((h_res1, output_in, output_out), dim=-1)
                gate_signal = F.dropout(self.gate_mlp(gate_input), self.dropout, training=self.training)
                gate_signal = self.gate_norm(gate_signal)
                gate_signal = F.dropout(F.softplus(gate_signal), self.dropout, training=self.training)
                output_out = gate_signal * output_out
            h = torch.torch.cat([h_res1, output_in, output_out], dim=-1)
        else:
            output_in, _ = self.attention(h, g)
            h = torch.torch.cat([h_res1, output_in], dim=-1)        

        h = self.O(h)
        
        h = h_res1 + h  # residual connection
        h = self.batch_norm1(h)

        h_res2 = h.clone()  # for second residual connection
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

        return h  # relations=Q*K^T/sqrt(d_k)


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()

        in_channels = args.in_channels
        hidden_channels = args.hidden_channels
        
        self.skip_first_edge_type = args.skip_first_edge_type
        self.dropout = args.dropout
        self.preds = args.Tstep -1
        self.gumbel_noise = args.gumbel_noise

        self.transformer_layers = nn.ModuleList(
            [MaskedGraphTransformerLayer(hidden_channels,hidden_channels, args.num_head,
                                         dropout=self.dropout,
                                         dual_attention=args.dual_attention, gates=args.gate) for _ in range(1)])
        
        
        self.input_fc = nn.Linear(in_channels, hidden_channels)
        self.norm_input = nn.LayerNorm(hidden_channels)
        self.norm_agg = nn.LayerNorm(hidden_channels)
        
        self.out_fc1 = nn.Linear(2*hidden_channels, hidden_channels)
        self.out_fc2 = nn.Linear(hidden_channels, hidden_channels)
        self.out_fc3 = nn.Linear(hidden_channels, in_channels)

    def single_step_forward(self, x, g):

        x0 = x.clone()
        h = F.softplus(self.input_fc(x))
        h = self.norm_input(h)
        h_res = h.clone()
        
        # for transformer_layer in self.transformer_layers:
        h_agg = self.transformer_layers[0](h, g)
        all_info = torch.cat((h_res, h_agg), dim=-1)
        # Output mlp
        pred = F.dropout(F.softplus(self.out_fc1(all_info)), p=self.dropout, training=self.training)
        pred = F.dropout(F.softplus(self.out_fc2(pred)), p=self.dropout, training=self.training)
        pred = self.out_fc3(pred)
        return x0 + pred

    def forward(self, inputs, logits):
        last_pred = inputs[...,0]
        g = get_edge_prob(logits, self.gumbel_noise)[0]
        I = torch.eye(g.shape[0], device=g.device, dtype=g.dtype)
        g = g * (1 - I) 

        preds = []
        for step in range(0,self.preds):
            last_pred = self.single_step_forward(last_pred, g)
            preds.append(last_pred.unsqueeze(-1))

        preds = torch.cat(preds,dim=-1)
        return preds
