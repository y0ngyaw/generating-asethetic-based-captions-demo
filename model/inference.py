import torch.utils
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn

from model.modules import * 
from model.utils import *

import os
import json
import numpy as np
from random import randint
from PIL import Image

path = os.path.join(os.getcwd(), 'model')
checkpoint = os.path.join(path, 'checkpoint_epoch_67.pth.tar')
arch = 'resnet50'

with open(os.path.join(path, 'word2idx.json'), 'r') as json_file:
	word2idx = json.load(json_file)
	print('word2idxx JSON loading completed.')
idx2word = {v:k for k, v in word2idx.items()}
vocab_size = len(word2idx)

ckpt = torch.load(checkpoint, map_location=torch.device('cpu'))
encoder = AestheticEncoder(arch=arch)
encoder.load_state_dict(ckpt['encoder'])
encoder.eval()
decoder = DocoderNoConcatEncoderDropConnect(feat_dim=4096, hidden_dim=512, attention_dim=256 ,embed_dim=256, vocab_size=vocab_size, weight_drop=0.5)
decoder.load_state_dict(ckpt['decoder'])
decoder.eval()

module = decoder.lstm_base
w = getattr(module, 'weight_hh_raw')
del module._parameters['weight_hh_raw']
module.register_parameter('weight_hh', Parameter(w))

# Normalization
# tfms = transforms.Compose([
# 	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])


def preprocess_img(img):
	img = img.convert('RGB')
	resized_img = img.resize((256, 256), resample=Image.LANCZOS)
	numpy_img = np.array(resized_img).transpose(2,0,1)
	img_tensor = torch.FloatTensor(numpy_img / 255.)
	assert img_tensor.shape == (3, 256, 256)
	return img_tensor

def evaluate(img, beam_size=3):
	k = beam_size
	# Convert image to image tensor
	img_tensor = preprocess_img(img).unsqueeze(dim=0)
	print(img_tensor.shape)

	# Encode image
	encoder_feat = encoder(img_tensor)
	encoder_feat = encoder_feat.expand(k, encoder_feat.size(1), encoder_feat.size(2))

	k_prev_words = torch.LongTensor([[word2idx['<start>']]] * k)
	seqs = k_prev_words

	top_k_scores = torch.zeros(k, 1)

	complete_seqs = []
	complete_seqs_scores = []

	# Initializing hidden states
	step = 1
	h, c = decoder.init_hidden(encoder_feat)

	while True: 
		embed_vector = decoder.embedding(k_prev_words).squeeze(1)

		attn_weight_encoding, _ = decoder.attention(encoder_feat, h)
		gate = decoder.sigmoid(decoder.f_beta(h))
		attn_weight_encoding = gate * attn_weight_encoding

		h, c = decoder.lstm(torch.cat([embed_vector, attn_weight_encoding], dim=1), (h, c))

		scores = decoder.fc(h)
		scores = nn.functional.log_softmax(scores, dim=1)

		scores = top_k_scores.expand_as(scores) + scores

		if step == 1:
			top_k_scores, top_k_words = scores[0].topk(k, 0)
		else:
			top_k_scores, top_k_words = scores.view(-1).topk(k, 0)

		prev_word_idx = top_k_words / vocab_size
		next_word_idx = top_k_words % vocab_size

		seqs = torch.cat([seqs[prev_word_idx], next_word_idx.unsqueeze(1)], dim=1)

		incomplete_idx = [idx for idx, next_word in enumerate(next_word_idx) if next_word != word2idx['<end>']]
		complete_idx = list(set(range(len(next_word_idx))) - set(incomplete_idx))

		if len(complete_idx) > 0:
			complete_seqs.extend(seqs[complete_idx].tolist())
			complete_seqs_scores.extend(top_k_scores[complete_idx])

		k -= len(complete_idx)

		# Exit loop if all completed
		if k == 0: break

		seqs = seqs[incomplete_idx]
		h = h[prev_word_idx[incomplete_idx]]
		c = c[prev_word_idx[incomplete_idx]]
		encoder_feat = encoder_feat[prev_word_idx[incomplete_idx]]
		top_k_scores = top_k_scores[incomplete_idx].unsqueeze(1)
		k_prev_words = next_word_idx[incomplete_idx].unsqueeze(1)

		# Break if things have been going on too long
		if step > 80: 
			complete_seqs.extend(seqs.tolist())
			complete_seqs_scores.extend(top_k_scores)
			break 

		step += 1

	all_captions = [[w for w in s if w not in [word2idx['<start>'], word2idx['<end>'], word2idx['<pad>']]] for s in complete_seqs]
	all_captions = [' ' .join([idx2word[w] for w in c]) for c in all_captions]
	return	all_captions

