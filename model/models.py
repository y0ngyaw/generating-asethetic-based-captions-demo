import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from torch.nn import Parameter

from model.utils import *

import time
import math


# Aesthetic Scoring
def get_pretrained_cnn(arch='resnet50'):
    model_func = getattr(models, arch)
    model = model_func(pretrained=True)

    return nn.Sequential(
        *listify(model.children())[:-2],
    )

class Flatten(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.nf = nf
    
    def __repr__(self): return f'{self.__class__.__name__}({self.nf})'

    def forward(self, x):
        return x.view((-1, self.nf))

class AestheticRegressor(nn.Module):
    def __init__(self, arch='resnet50', dropout=0.25):
        super().__init__()
        model = get_pretrained_cnn(arch)
        self.model = nn.Sequential(
            *listify(model),
            nn.AdaptiveAvgPool2d((1,1)),
            Flatten(2048),
            nn.Linear(2048, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 10),
            nn.BatchNorm1d(10),
            nn.Dropout(0.25),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.model(x)

class EMDLoss(nn.Module):
    def __init__(self):
        super(EMDLoss, self).__init__()
    
    def forward(self, y_pred, y_true):
        y_true_cumsum = torch.cumsum(y_true, dim=-1)
        y_pred_cumsum = torch.cumsum(y_pred, dim=-1)
        return torch.sqrt(torch.pow(y_true_cumsum - y_pred_cumsum, 2).mean())


# Aesthetic Captioning
class Encoder(nn.Module):
    def __init__(self, arch='resnet50', last_lyrs=None):
        super().__init__()
        model_func = getattr(models, arch)
        model = model_func(pretrained="True")
        lyrs = listify(model.children())[:-2]
        if last_lyrs is not None: lyrs += listify(last_lyrs)
        self.model = nn.Sequential(*lyrs)

    def forward(self, x):
        return self.model(x)


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


class DecoderWithAttention(nn.Module):
    def __init__(self, feat_dim, hidden_dim, attention_dim, embed_dim, vocab_size):
        super(DecoderWithAttention, self).__init__()
        self.feat_dim      = feat_dim
        self.hidden_dim    = hidden_dim
        self.attention_dim = attention_dim
        self.embed_dim     = embed_dim
        self.vocab_size    = vocab_size

        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        self.attention = Attention(self.feat_dim, self.hidden_dim)
        self.init_h    = nn.Linear(self.feat_dim, self.hidden_dim)
        self.init_c    = nn.Linear(self.feat_dim, self.hidden_dim)
        self.f_beta    = nn.Linear(self.hidden_dim, self.feat_dim)  
        self.lstm    = nn.LSTMCell(self.embed_dim+self.feat_dim, self.hidden_dim, bias=True)
        self.fc      = nn.Linear(self.hidden_dim, self.vocab_size)
        self.dropout = nn.Dropout(p=0.3)

        self.sigmoid = nn.Sigmoid()
        self.tanh    = nn.Tanh()

        self.init_linear(self.init_h)
        self.init_linear(self.init_c)
        self.init_linear(self.f_beta)
        self.init_linear(self.fc)
        self.embedding.weight.data.uniform_(-0.1, 0.1)

        self.current_device = torch.cuda.current_device()

    def init_linear(self, linear_lyr):
        nn.init.kaiming_normal_(linear_lyr.weight)
        if hasattr(linear_lyr, 'bias'): nn.init.zeros_(linear_lyr.bias)

    def load_embeddings(self, embed_vector):
        if not isinstance(embed_vector, torch.Tensor): embed_vector = torch.tensor(embed_vector)
        self.embedding = nn.Embedding.from_pretrained(embed_vector, freeze=True)

    def init_hidden(self, encoder_feat):
        feat = encoder_feat.mean(dim=1)
        h    = self.tanh(self.init_h(feat))
        c    = self.tanh(self.init_c(feat))
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
        predictions = torch.zeros((batch_size, max(captions_len), self.vocab_size), device=self.current_device)
        alphas = torch.zeros((batch_size, max(captions_len), num_pixels), device=self.current_device)

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


class Decoder(nn.Module):
    def __init__(self, feat_dim, hidden_dim, embed_dim, vocab_size):
        super().__init__()
        self.feat_dim   = feat_dim
        self.hidden_dim = hidden_dim
        self.embed_dim  = embed_dim
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        self.init_h    = nn.Linear(self.feat_dim, self.hidden_dim)
        self.init_c    = nn.Linear(self.feat_dim, self.hidden_dim)
        self.lstm    = nn.LSTM(self.embed_dim, self.hidden_dim, batch_first=True)
        self.fc      = nn.Linear(self.hidden_dim, self.vocab_size)
        self.dropout = nn.Dropout(p=0.3)

        self.tanh    = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

        self.init_linear(self.init_h)
        self.init_linear(self.init_c)
        self.init_linear(self.fc)

    def init_hidden(self, encoder_feat):
        encoder_feat = self.reshape_encoder_feat(encoder_feat)
        feat = encoder_feat.mean(dim=1)
        h    = self.tanh(self.init_h(feat))
        c    = self.tanh(self.init_c(feat))
        return h.unsqueeze(0), c.unsqueeze(0) 

    def init_linear(self, linear_lyr):
        nn.init.kaiming_normal_(linear_lyr.weight)
        if hasattr(linear_lyr, 'bias'): nn.init.zeros_(linear_lyr.bias)

    def load_embeddings(self, embed_vector):
        if not isinstance(embed_vector, torch.Tensor): embed_vector = torch.tensor(embed_vector)
        self.embedding = nn.Embedding.from_pretrained(embed_vector, freeze=True)

    def reshape_encoder_feat(self, encoder_feat):
        batch_size   = encoder_feat.shape[0]
        encoder_feat = encoder_feat.permute(0,2,3,1)
        encoder_feat = encoder_feat.view(encoder_feat.shape[0], -1, encoder_feat.shape[-1])
        return encoder_feat

    def forward(self, encoder_feat, gt_captions, captions_len):
        self.h, self.c = self.init_hidden(encoder_feat)
        word_vectors = self.embedding(gt_captions)
        lstm_input = nn.utils.rnn.pack_padded_sequence(word_vectors, torch.tensor(captions_len)-1, batch_first=True) # ignore <end> token as lstm input, thus -1 in captions len
        lstm_output, (self.h, self.c) = self.lstm(lstm_input, (self.h, self.c))

        pred = self.fc(self.dropout(lstm_output.data))
        return pred

    def predict_one_step(self, prev_word, h, c):
        word_vectors = self.embedding(prev_word)
        lstm_output, (h, c) = self.lstm(word_vectors, (h, c))
        return self.fc(lstm_output), (h, c)


class BaselineCaptioningModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(feat_dim=2048, embed_dim=300, hidden_dim=512, vocab_size=400003)

    def forward(self, data):
        xb = data[0]
        batch_caption = data[1]
        return self.decoder(self.encoder(xb), batch_caption.captions, batch_caption.captions_len)


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


class AestheticCaptioningModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AestheticEncoder()
        self.decoder = Decoder(feat_dim=4096, embed_dim=300, hidden_dim=512, vocab_size=400003)

    def forward(self, data):
        xb = data[0]
        batch_caption = data[1]
        return self.decoder(self.encoder(xb), batch_caption.captions, batch_caption.captions_len)


class AestheticEncoderWithoutConcatFeat(nn.Module):
    def __init__(self, arch='resnet50', aesthetic_weight_path=None):
        super().__init__()
        model_func = getattr(models, arch)
        model = model_func(pretrained=True)
        lyrs = listify(model.children())[:-2]
        self.imagenet_model = nn.Sequential(*lyrs)

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
        return factual_feat, aesthetic_feat


class DecoderWithAttentionNoConcatEncoder(nn.Module):
    def __init__(self, feat_dim, hidden_dim, attention_dim, embed_dim, vocab_size):
        super(DecoderWithAttentionNoConcatEncoder, self).__init__()
        self.feat_dim      = feat_dim
        self.hidden_dim    = hidden_dim
        self.attention_dim = attention_dim
        self.embed_dim     = embed_dim
        self.vocab_size    = vocab_size

        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        self.attention = Attention(self.feat_dim, self.hidden_dim)
        self.init_h    = nn.Linear(self.feat_dim, self.hidden_dim)
        self.init_c    = nn.Linear(self.feat_dim, self.hidden_dim)
        self.init_feat = nn.Linear(self.feat_dim*2, self.embed_dim) # Multiply by 2 because taking mean and max pooling
        self.f_beta    = nn.Linear(self.hidden_dim, self.feat_dim)  
        self.lstm    = nn.LSTMCell(self.embed_dim+self.feat_dim, self.hidden_dim, bias=True)
        self.fc      = nn.Linear(self.hidden_dim, self.vocab_size)
        self.dropout = nn.Dropout(p=0.3)

        self.sigmoid = nn.Sigmoid()
        self.tanh    = nn.Tanh()

        self.init_linear(self.init_h)
        self.init_linear(self.init_c)
        self.init_linear(self.init_feat)
        self.init_linear(self.f_beta)
        self.init_linear(self.fc)
        self.embedding.weight.data.uniform_(-0.1, 0.1)

        self.current_device = torch.cuda.current_device()

    def init_linear(self, linear_lyr):
        nn.init.kaiming_normal_(linear_lyr.weight)
        if hasattr(linear_lyr, 'bias'): nn.init.zeros_(linear_lyr.bias)

    def load_embeddings(self, embed_vector):
        if not isinstance(embed_vector, torch.Tensor): embed_vector = torch.tensor(embed_vector)
        self.embedding = nn.Embedding.from_pretrained(embed_vector, freeze=True)

    def init_hidden(self, encoder_feat):
        feat = encoder_feat.mean(dim=1)
        h    = self.tanh(self.init_h(feat))
        c    = self.tanh(self.init_c(feat))
        return h, c

    def get_first_word(self, factual_feat):
        avg_feat = factual_feat.mean(dim=1)
        max_feat = factual_feat.max(dim=1).values
        feat = torch.cat([avg_feat, max_feat], dim=1)
        return self.tanh(self.init_feat(feat))

    def forward(self, factual_feat, aesthetic_feat, gt_captions, captions_len):
        batch_size = factual_feat.size(0)
        num_pixels = factual_feat.size(1)

        first_word = self.get_first_word(aesthetic_feat) # (batch_size, 512)

        # Convert word index to word vector, starting from second word onwards because first word will be start token
        embed_vector = self.embedding(gt_captions[:, 1:])
        embed_vector = torch.cat([first_word.unsqueeze(1), embed_vector], dim=1) # (batch_size, max_word_length, 512)

        # Initiate hidden state
        h, c = self.init_hidden(factual_feat)

        # Since we are not decoding the <end> token (last token), captions length should be minus 1
        captions_len = (captions_len - 1).tolist()

        # Create tensor to hold predictions and alphas
        predictions = torch.zeros((batch_size, max(captions_len), self.vocab_size), device=self.current_device)
        alphas = torch.zeros((batch_size, max(captions_len), num_pixels), device=self.current_device)

        # At each timestep, predict the word by
        # attention-weighting the encoder's output based on the decoder's previous hidden state
        for t in range(max(captions_len)):
            batch_size_t = sum([l > t for l in captions_len])
            attn_weighted_encoding, alpha = self.attention(factual_feat[:batch_size_t], h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            attn_weighted_encoding = gate * attn_weighted_encoding

            # Passing concatenation results of attn and encoder_feat to lstm, alongisde with h and c
            h, c = self.lstm(torch.cat([embed_vector[:batch_size_t, t, :], attn_weighted_encoding], dim=1), (h[:batch_size_t], c[:batch_size_t]))

            preds = self.fc(self.dropout(h))
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, alphas

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

        self.current_device = torch.cuda.current_device()

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
        predictions = torch.zeros((batch_size, max(captions_len), self.vocab_size), device=self.current_device)
        alphas = torch.zeros((batch_size, max(captions_len), num_pixels), device=self.current_device)

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


class DocoderZeroHiddenDropConnect(nn.Module):
    def __init__(self, feat_dim, hidden_dim, embed_dim, vocab_size, weight_drop):
        super(DocoderZeroHiddenDropConnect, self).__init__()
        self.feat_dim      = feat_dim
        self.hidden_dim    = hidden_dim
        self.embed_dim     = embed_dim
        self.vocab_size    = vocab_size
        self.weight_drop   = weight_drop

        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        self.init_start   = nn.Linear(self.feat_dim, self.embed_dim)
        self.bn_start     = nn.BatchNorm1d(self.embed_dim)
        self.dropout_start = nn.Dropout(p=0.5)
        self.lstm_base    = nn.LSTMCell(self.embed_dim, self.hidden_dim, bias=True)
        self.lstm    = WeightDrop(self.lstm_base, ['weight_hh'], self.weight_drop) # Weight drop is a wrapper around lstm forward function
        self.fc      = nn.Linear(self.hidden_dim, self.vocab_size)
        self.dropout = nn.Dropout(p=0.5)

        self.sigmoid = nn.Sigmoid()
        self.tanh    = nn.Tanh()

        self.init_linear(self.init_start)
        self.init_linear(self.fc)
        self.embedding.weight.data.uniform_(-0.1, 0.1)

        self.current_device = torch.cuda.current_device()

    def init_linear(self, linear_lyr):
        nn.init.kaiming_normal_(linear_lyr.weight)
        if hasattr(linear_lyr, 'bias'): nn.init.zeros_(linear_lyr.bias)

    def load_embeddings(self, embed_vector):
        if not isinstance(embed_vector, torch.Tensor): embed_vector = torch.tensor(embed_vector)
        self.embedding = nn.Embedding.from_pretrained(embed_vector, freeze=True)

    def generate_lstm_input(self, encoder_feat):
        feat = encoder_feat.mean(dim=1)
        return self.dropout_start(self.bn_start(self.sigmoid(self.init_start(feat))))

    def forward(self, encoder_feat, gt_captions, captions_len):
        batch_size = encoder_feat.size(0)

        # Convert word index to word vector
        embed_vector = self.embedding(gt_captions)

        # Since we are not decoding the <end> token (last token), captions length should be minus 1
        captions_len = (captions_len - 1).tolist()

        # Create tensor to hold predictions
        predictions = torch.zeros((batch_size, max(captions_len), self.vocab_size), device=self.current_device)

        # For first timestep, input feature from encoder
        # For second timestp onwards, using word vector from embedding
        for t in range(max(captions_len)+1):
            if t == 0:
                feat = self.generate_lstm_input(encoder_feat)
                h, c = self.lstm(feat)
            else:
                timestep = t -1
                batch_size_t = sum([l > timestep for l in captions_len])

                # Passing concatenation results of attn and encoder_feat to lstm, alongisde with h and c
                h, c = self.lstm(embed_vector[:batch_size_t, timestep, :], (h[:batch_size_t], c[:batch_size_t]))

                preds = self.fc(self.dropout(h))
                predictions[:batch_size_t, timestep, :] = preds

        return predictions

#######################################
# Transformer Module
#######################################
class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        self.current_device = torch.cuda.current_device()
        
        self.query_conv = nn.Conv2d(in_channels=in_dim , out_channels=in_dim//8 , kernel_size=1)
        self.key_conv   = nn.Conv2d(in_channels=in_dim , out_channels=in_dim//8 , kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim , out_channels=in_dim , kernel_size=1)
        self.gamma      = nn.Parameter(torch.zeros(1, device=self.current_device))
        self.softmax    = nn.Softmax(dim=-1)

    def forward(self, x):
        bs, channel, width, height = x.size()
        proj_query = self.query_conv(x).view(bs, -1, width*height).permute(0, 2, 1)
        proj_key   = self.key_conv(x).view(bs, -1, width*height)
        energy     = torch.bmm(proj_query, proj_key)
        attention  = self.softmax(energy)
        proj_value = self.value_conv(x).view(bs, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0,2,1))
        out = out.view(bs, channel, width, height)
        
        out = self.gamma * out + x
        return out, attention


class AestheticEncoderSellfAttention(nn.Module):
    def __init__(self, arch='resnet50', aesthetic_weight_path=None, decoder_dim=512):
        super(AestheticEncoderSellfAttention, self).__init__()
        model_func = getattr(models, arch)
        model = model_func(pretrained=True)
        lyrs = listify(model.children())[:-2]
        self.imagenet_model = nn.Sequential(*lyrs)

        self.aesthetic_model = get_pretrained_cnn(arch)
        if aesthetic_weight_path is not None: self.aesthetic_model.load_state_dict(torch.load(aesthetic_weight_path, map_location=torch.device('cpu')))
        self.aesthetic_model = nn.Sequential(*listify(self.aesthetic_model.children())[:8])

        self.sa_lyr = SelfAttention(4096)

        self.decoder_dim = decoder_dim
        self.linear = nn.Linear(4096, self.decoder_dim)
        self.activation = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(64)

    def forward(self, x):
        factual_feat   = self.imagenet_model(x)
        aesthetic_feat = self.aesthetic_model(x)
        feat = torch.cat([factual_feat, aesthetic_feat], dim=1) # (N, 4096, 8, 8)
        output, attn_weight = self.sa_lyr(feat) # (N, 2048, 8, 8)
        output = output.view(output.size(0), output.size(1), -1).permute(0,2,1) # (N, 64, 2048)
        output = self.linear(output) # (N, 64, 512)
        return self.batch_norm(self.activation(output))


class PositionalEncoding(nn.Module):
    def __init__(self, decoder_dim, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.current_device = torch.cuda.current_device()

        pe = torch.zeros(max_len, decoder_dim, device=self.current_device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, decoder_dim, 2).float() * (-math.log(10000.0) / decoder_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerDecoderModule(nn.Module):
    def __init__(self, vocab_size=30000, decoder_dim=512, num_head=8, num_lyrs=6):
        super(TransformerDecoderModule, self).__init__()
        self.vocab_size = vocab_size
        self.decoder_dim = decoder_dim
        self.num_head = num_head
        self.num_lyrs = num_lyrs
        self.current_device = torch.cuda.current_device()

        self.embedding = nn.Embedding(self.vocab_size, self.decoder_dim)
        self.pos_encode = PositionalEncoding(self.decoder_dim)
        self.transformer_lyr = nn.TransformerDecoderLayer(d_model=self.decoder_dim, nhead=self.num_head)
        self.transformer_decoder = nn.TransformerDecoder(self.transformer_lyr, self.num_lyrs)

        self.fc      = nn.Linear(self.decoder_dim, self.vocab_size)
        self.dropout = nn.Dropout(p=0.5)

        self.init_linear(self.fc)
        self.embedding.weight.data.uniform_(-0.1, 0.1)

    def init_linear(self, linear_lyr):
        nn.init.kaiming_normal_(linear_lyr.weight)
        if hasattr(linear_lyr, 'bias'): nn.init.zeros_(linear_lyr.bias)

    def get_subsequent_mask(self, gt_captions):
        sz = gt_captions.size(-1)
        mask = (torch.triu(torch.ones(sz, sz, device=self.current_device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.cuda()

    def forward(self, encoder_feat, gt_captions, captions_len):
        word_vector  = self.embedding(gt_captions) # (N, batch_max_seq_len, 512)
        pe_output = self.pos_encode(word_vector) # (N, batch_max_seq_len, 512)

        # Pad index is 0
        padding_mask = (gt_captions == 0).cuda() # (N, batch_max_seq_len)
        subsequent_mask = self.get_subsequent_mask(gt_captions) # (batch_max_seq_len, batch_max_seq_len)

        encoder_feat = encoder_feat.permute(1,0,2) # (N, 64, decoder_dim) => (64, N, decoder_dim)
        pe_output = pe_output.permute(1,0,2) # (N, batch_max_seq_len, 512) => (batch_max_seq_len, N, 512)

        pred = self.transformer_decoder(pe_output, encoder_feat, tgt_mask=subsequent_mask.transpose(0,1), tgt_key_padding_mask=padding_mask) # (batch_max_seq_len, N, 512)
        return self.dropout(self.fc(pred.permute(1,0,2))) # (N, batch_max_seq_len, 512)