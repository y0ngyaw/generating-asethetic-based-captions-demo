import torch.utils
import torch
import torchvision.transforms as  transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from nltk.translate.bleu_score import corpus_bleu

from coco_caption.pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from coco_caption.pycocoevalcap.bleu.bleu import Bleu
from coco_caption.pycocoevalcap.meteor.meteor import Meteor
from coco_caption.pycocoevalcap.rouge.rouge import Rouge
from coco_caption.pycocoevalcap.cider.cider import Cider
from coco_caption.pycocoevalcap.spice.spice import Spice
from coco_caption.pycocoevalcap.wmd.wmd import WMD

from models import * 
from utils import *
from dataset import AVACaptioningDataset

import os
import json
from tqdm import tqdm
import pandas as pd
import numpy as np
from random import randint

path = './model_v5'
checkpoint = f'{path}/checkpoint_epoch_39.pth.tar'
arch = 'resnext50_32x4d'
print("Model from", checkpoint)
gpu_num = 0
# dataset_path = '/home/viprlab/Documents/yongyaw/Full Validation'
# dataset_path = '/home/viprlab/Documents/yongyaw'
dataset_path = './dataset/Full Validation'
pred_json_filename = f'{path}/pred_fv_epoch_39.json'
result_csv_filename = f'{path}/results_fv_epoch_39.csv'
# mode = 'val'
mode = 'full_validation'
# aesthetic_weight_path = './aesthetic_encoder_weight/aesthetic_weight_39.pt'
aesthetic_weight_path = './aesthetic_regressor_resnext/aesthetic_encoder_state_dict_epoch_39.pth'

# Config GPU
torch.cuda.set_device(gpu_num)
print('Running on GPU', torch.cuda.current_device())

# Load vocab
with open(os.path.join(dataset_path, 'word2idx.json'), 'r') as json_file:
	word2idx = json.load(json_file)
	print('word2idxx JSON loading completed.')
idx2word = {v:k for k, v in word2idx.items()}
vocab_size = len(word2idx)
print(f"Vocab Size: {vocab_size}")

# Load Model
ckpt = torch.load(checkpoint, map_location=torch.device('cpu'))
encoder = AestheticEncoder(arch=arch, aesthetic_weight_path=aesthetic_weight_path)
encoder.load_state_dict(ckpt['encoder'])
encoder = encoder.cuda()
encoder.eval()
decoder = DocoderNoConcatEncoderDropConnect(feat_dim=4096, hidden_dim=512, attention_dim=256 ,embed_dim=256, vocab_size=len(word2idx), weight_drop=0.5)
decoder.load_state_dict(ckpt['decoder'])
decoder = decoder.cuda()
decoder.eval()

module = decoder.lstm_base
w = getattr(module, 'weight_hh_raw')
del module._parameters['weight_hh_raw']
module.register_parameter('weight_hh', Parameter(w))


# Normalization
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Different metrics
scorer = {
	'BLEU': Bleu(),
	'Meteor': Meteor(),
	'Rouge': Rouge(),
	'Cider': Cider(),
	'WMD': WMD(),
	# 'Spice': Spice()
	}

def evaluate(beam_size=3):
	# Dataloader
	val_dl = DataLoader(AVACaptioningDataset(data_fd=dataset_path, mode=mode, transform=transforms.Compose([normalize])), 
		batch_size=1, 
		shuffle=False,
		num_workers=1,	
		pin_memory=True)

	references = []
	hypotheses = []
	references_dict = {}
	hypotheses_dict = {}
	img_ids = []
	all_captions_list = []

	print('Evaluating ... ')
	print(len(val_dl))
	for i, (image, caption, caption_len, all_captions, captions_image_id) in enumerate(tqdm(val_dl)):
		k = beam_size

		# Move to GPU
		image = image.cuda()

		# Encode image
		encoder_feat = encoder(image) # (1, 64, 4096)

		# Expand encoder_feat batch size to k (beam size), assuming we are duplicating same images
		encoder_feat = encoder_feat.expand(k, encoder_feat.size(1), encoder_feat.size(2)) # (k, 64,4096)

		# Tensor to store top k previous word
		k_prev_words = torch.LongTensor([[word2idx['<start>']]] * k).cuda() # (k, 1)

		# Tensor to store top k sequences
		seqs = k_prev_words # (k, 1)

		# Tensor to store top k sequences' scores; currently zeros
		top_k_scores = torch.zeros(k, 1).cuda()

		# Lists to store completed sequences and scores
		complete_seqs = []
		complete_seqs_scores = []

		# Start decoding
		step = 1
		h, c = decoder.init_hidden(encoder_feat) # (k, 64, 512), (k, 64, 512)

		while True:
			# Convert previous word to word embedding
			embed_vector = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

			# Attention on images features and previous hidden state
			attn_weight_encoding, _ = decoder.attention(encoder_feat, h)
			gate = decoder.sigmoid(decoder.f_beta(h))
			attn_weight_encoding = gate * attn_weight_encoding

			h, c = decoder.lstm(torch.cat([embed_vector, attn_weight_encoding], dim=1), (h, c))

			scores = decoder.fc(h)
			scores = nn.functional.log_softmax(scores, dim=1)

			# Add 
			scores = top_k_scores.expand_as(scores) + scores

			# For the first step, all k points will have the same scores (since same k previous word, h, c)
			if step == 1:
				top_k_scores, top_k_words = scores[0].topk(k, 0)
			else:
				top_k_scores, top_k_words = scores.view(-1).topk(k, 0)

			# Since flatten, to obtain back the matrix position, divide with vocab size will obtain the rows and modulus will obtain the cols
			# In this case, row index represents the previous word location in seqs, col index represent the next word index
			prev_word_idx = top_k_words / vocab_size
			next_word_idx = top_k_words % vocab_size

			# Add new words to sequences
			seqs = torch.cat([seqs[prev_word_idx], next_word_idx.unsqueeze(1)], dim=1) # (k, step+1)

			# Incomplete word index
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

		try: 
			i = complete_seqs_scores.index(max(complete_seqs_scores))
			# i = randint(0, len(complete_seqs)-1)
		except ValueError:
			print('Image Id:', captions_image_id)
			print('Complete_seqs_scores:', complete_seqs_scores)
			print('complete_seqs:', complete_seqs)
			print('Step:', step)
			print('Incomplete_idx:', incomplete_idx)
			print('complete_idx:', complete_idx)
		seqs = complete_seqs[i]

		# References
		image_caption = all_captions[0].tolist()
		image_captions = list(map(lambda c: [w for w in c if w not in [word2idx['<start>'], word2idx['<end>'], word2idx['<pad>']]], image_caption))
		references.append(image_captions)
		references_dict[captions_image_id[0].item()] = [" ".join([idx2word[w] for w in c]) for c in image_captions]

		# Hypotheses
		hypothesis = [w for w in seqs if w not in [word2idx['<start>'], word2idx['<end>'], word2idx['<pad>']]]
		hypotheses.append(hypothesis)
		hypotheses_dict[captions_image_id[0].item()] = [" ".join([idx2word[w] for w in hypothesis])]
		img_ids.append(captions_image_id[0].item())

		# All Captions
		all_captions = [[w for w in s if w not in [word2idx['<start>'], word2idx['<end>'], word2idx['<pad>']]] for s in complete_seqs]
		all_captions = [' ' .join([idx2word[w] for w in c]) for c in all_captions]
		all_captions_list.append(all_captions)

		# Sanity Check
		assert len(references) == len(hypotheses)

	print(len(img_ids))
	print(len(np.unique(img_ids)))

	# Calculate BLEU4 scores
	bleu4 = corpus_bleu(references, hypotheses)

	result = {}
	for method in scorer.keys():
		print(f"Evaluating using {method}")
		res = scorer[method].compute_score(references_dict, hypotheses_dict)
		result[method] = res
		print(f"{method} result: {res[0]}\n")

	print(len(hypotheses_dict))
	with open(pred_json_filename, 'w') as json_file:
		json.dump(hypotheses_dict, json_file)

	assert len(all_captions_list) == len(img_ids)

	data = {
		'image_id': img_ids,
		'captions': all_captions_list
	}
	df = pd.DataFrame(data)
	df.to_csv(result_csv_filename, index=False)

	return result, bleu4


if __name__ == "__main__":
	evaluate()
	