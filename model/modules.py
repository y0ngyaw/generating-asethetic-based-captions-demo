import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from torch.nn import Parameter

from model.utils import *

import time
import math


def get_pretrained_cnn(arch='resnet50'):
    model_func = getattr(models, arch)
    model = model_func(pretrained=True)

    return nn.Sequential(
        *listify(model.children())[:-2],
    )

class AestheticEncoder(nn.Module):
    def __init__(self, arch='resnet50', aesthetic_weight_path=None):
        super().__init__()
        self.imagenet_model = get_pretrained_cnn(arch)
        self.aesthetic_model = get_pretrained_cnn(arch)
        if aesthetic_weight_path is not None: self.aesthetic_model.load_state_dict(torch.load(aesthetic_weight_path, map_location=torch.device('cpu')))
        self.aesthetic_model = nn.Sequential(*listify(self.aesthetic_model.children())[:8])

    def reshape_feat(self, x):
        encoder_feat = x.permute(0,2,3,1)
        return encoder_feat.view(encoder_feat.shape[0], -1, encoder_feat.shape[-1])

    def forward(self, x):
        factual_feat   = self.imagenet_model(x)
        aesthetic_feat = self.aesthetic_model(x)
        factual_feat, aesthetic_feat = self.reshape_feat(factual_feat), self.reshape_feat(aesthetic_feat)
        return torch.cat([factual_feat, aesthetic_feat], dim=-1)


class Attention(nn.Module):
    def __init__(self, feat_dim, hidden_dim, nf=512):
        super().__init__()
        self.feat_att   = nn.Linear(feat_dim, nf)
        self.hidden_att = nn.Linear(hidden_dim, nf)
        self.v          = nn.Linear(nf, 1)
        self.softmax    = nn.Softmax(dim=1)
        self.relu       = nn.ReLU()

        self.init_linear(self.feat_att)
        self.init_linear(self.hidden_att)
        self.init_linear(self.v)

    def init_linear(self, linear_lyr):
        nn.init.kaiming_normal_(linear_lyr.weight)
        if hasattr(linear_lyr, 'bias'): nn.init.zeros_(linear_lyr.bias)

    def forward(self, encoder_feat, hidden_state):
        feat_att   = self.feat_att(encoder_feat) 
        hidden_att = self.hidden_att(hidden_state.unsqueeze(1))
        att        = self.relu(feat_att + hidden_att)
        e          = self.v(att).squeeze(2)
        alpha      = self.softmax(e)
        context    = (encoder_feat * alpha.unsqueeze(2)).sum(dim=1)
        return context, alpha


def _weight_drop(module, weights, dropout):
    for name_w in weights:
        w = getattr(module, name_w)
        del module._parameters[name_w]
        module.register_parameter(name_w + '_raw', Parameter(w))

    original_module_forward = module.forward

    def forward(*args, **kwargs):
        if module.training:
            for name_w in weights:
                raw_w = getattr(module, name_w + '_raw')
                w = torch.nn.functional.dropout(raw_w, p=dropout)
                setattr(module, name_w, w)

        return original_module_forward(*args, **kwargs)

    setattr(module, 'forward', forward)


class WeightDrop(torch.nn.Module):
    def __init__(self, module, weights, dropout=0.0):
        super(WeightDrop, self).__init__()
        _weight_drop(module, weights, dropout)
        self.forward = module.forward


class DocoderNoConcatEncoderDropConnect(nn.Module):
    def __init__(self, feat_dim, hidden_dim, attention_dim, embed_dim, vocab_size, weight_drop):
        super(DocoderNoConcatEncoderDropConnect, self).__init__()
        self.feat_dim      = feat_dim
        self.hidden_dim    = hidden_dim
        self.attention_dim = attention_dim
        self.embed_dim     = embed_dim
        self.vocab_size    = vocab_size
        self.weight_drop   = weight_drop

        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        self.attention = Attention(self.feat_dim, self.hidden_dim)
        self.init_h    = nn.Linear(self.feat_dim, self.hidden_dim)
        self.init_c    = nn.Linear(self.feat_dim, self.hidden_dim)
        self.init_dropout = nn.Dropout(p=0.5)
        self.bn_h      = nn.BatchNorm1d(self.hidden_dim)
        self.bn_c      = nn.BatchNorm1d(self.hidden_dim)
        self.f_beta    = nn.Linear(self.hidden_dim, self.feat_dim)  
        self.lstm_base    = nn.LSTMCell(self.embed_dim+self.feat_dim, self.hidden_dim, bias=True)
        self.lstm    = WeightDrop(self.lstm_base, ['weight_hh'], self.weight_drop) # Weight drop is a wrapper around lstm forward function
        self.fc      = nn.Linear(self.hidden_dim, self.vocab_size)
        self.dropout = nn.Dropout(p=0.5)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.init_linear(self.init_h)
        self.init_linear(self.init_c)
        self.init_linear(self.f_beta)
        self.init_linear(self.fc)
        self.embedding.weight.data.uniform_(-0.1, 0.1)


    def init_linear(self, linear_lyr):
        nn.init.kaiming_normal_(linear_lyr.weight)
        if hasattr(linear_lyr, 'bias'): nn.init.zeros_(linear_lyr.bias)

    def init_hidden(self, encoder_feat):
        feat = encoder_feat.mean(dim=1)
        h    = self.init_dropout(self.bn_h(self.sigmoid(self.init_h(feat))))
        c    = self.init_dropout(self.bn_c(self.sigmoid(self.init_c(feat))))
        return h, c

    def forward(self, encoder_feat, gt_captions, captions_len):
        batch_size = encoder_feat.size(0)
        num_pixels = encoder_feat.size(1)

        # Convert word index to word vector
        embed_vector = self.embedding(gt_captions)

        # Initiate hidden state
        h, c = self.init_hidden(encoder_feat)

        # Since we are not decoding the <end> token (last token), captions length should be minus 1
        captions_len = (captions_len - 1).tolist()

        # Create tensor to hold predictions and alphas
        predictions = torch.zeros((batch_size, max(captions_len), self.vocab_size))
        alphas = torch.zeros((batch_size, max(captions_len), num_pixels))

        # At each timestep, predict the word by
        # attention-weighting the encoder's output based on the decoder's previous hidden state
        for t in range(max(captions_len)):
            batch_size_t = sum([l > t for l in captions_len])
            attn_weighted_encoding, alpha = self.attention(encoder_feat[:batch_size_t], h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            attn_weighted_encoding = gate * attn_weighted_encoding

            # Passing concatenation results of attn and encoder_feat to lstm, alongisde with h and c
            h, c = self.lstm(torch.cat([embed_vector[:batch_size_t, t, :], attn_weighted_encoding], dim=1), (h[:batch_size_t], c[:batch_size_t]))

            preds = self.fc(self.dropout(h))
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, alphas
