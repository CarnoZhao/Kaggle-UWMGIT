import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np

class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, attention_drop_rate):
        super(Attention, self).__init__()
        self.num_attention_heads = num_heads
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.out = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(attention_drop_rate)
        self.proj_dropout = nn.Dropout(attention_drop_rate)

        self.softmax = nn.Softmax(dim = -1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output

class MLP(nn.Module):
    def __init__(self, hidden_size, channels, drop_rate):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(hidden_size, channels)
        self.fc2 = nn.Linear(channels, hidden_size)
        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(drop_rate)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std =1e-6)
        nn.init.normal_(self.fc2.bias, std = 1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, args):
        super(TransformerBlock, self).__init__()
        hidden_size = args.hidden_size

        self.hidden_size = hidden_size
        self.attention_norm = nn.LayerNorm(hidden_size, eps = 1e-6)
        self.ffn_norm = nn.LayerNorm(hidden_size, eps = 1e-6)
        self.ffn = MLP(hidden_size, args.channels, args.drop_rate)
        self.attn = Attention(hidden_size, args.num_heads, args.attention_drop_rate)

    def forward(self, x):
        out = self.attention_norm(x)
        out = self.attn(out)
        x = out + x

        out = self.ffn_norm(x)
        out = self.ffn(out)
        x = out + x
        return x

class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer, self).__init__()
        self.norm = nn.LayerNorm(args.hidden_size, eps = 1e-6)
        layers = []
        for _ in range(args.num_layers):
            layer = TransformerBlock(args)
            layers.append(layer)
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        out = self.norm(x)
        return out


class TransUnetEncoderWrapper(nn.Module):
    def __init__(self, backbone, args):
        super(TransUnetEncoderWrapper, self).__init__()
        self.backbone = backbone
        self.out_channels = list(self.backbone.out_channels)

        img_size = args.image_size
        grid_size = args.grid_size
        stride = 2 ** (len(self.out_channels) - 1)
        patch_size = (img_size[0] // stride // grid_size[0], img_size[1] // stride // grid_size[1])
        patch_size_real = (patch_size[0] * stride, patch_size[1] * stride)
        num_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])

        # args.hidden_size = self.out_channels[-1]
        self.patch_embeddings = nn.Conv2d(in_channels = self.out_channels[-1],
                                       out_channels = args.hidden_size,
                                       kernel_size = patch_size,
                                       stride = patch_size)
        self.out_channels[-1] = args.hidden_size
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, num_patches, args.hidden_size))
        self.dropout = nn.Dropout(args.drop_rate)

        self.transformer = Transformer(args)

    def forward(self, x):
        features = self.backbone(x)

        out = self.patch_embeddings(features[-1])  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        out = out.flatten(2)
        out = out.transpose(-1, -2)  # (B, n_patches, hidden)

        embeddings = out + self.position_embeddings
        embeddings = self.dropout(embeddings)

        out = self.transformer(embeddings)

        B, num_patches, hidden_size = out.shape  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(num_patches)), int(np.sqrt(num_patches))
        out = out.permute(0, 2, 1)
        out = out.contiguous().view(B, hidden_size, h, w)

        features[-1] = out
        return features
