import torch
import torch.nn as nn
import torchvision.models as models
from collections.abc import Iterable
import os
import json

def listify(x):
    if x is None: return []
    if isinstance(x, list): return x
    if isinstance(x, str): return [x]
    if isinstance(x, Iterable): return list(x)
    return [x]

def freeze(model, lyr_type=[nn.BatchNorm2d], freeze_status=True):
    for lyr in model.children():
        if isinstance(lyr, nn.Sequential) or isinstance(lyr, models.resnet.Bottleneck):
            freeze(lyr, lyr_type)
            
        if True in [isinstance(lyr, t) for t in lyr_type]:
            if hasattr(lyr, 'weight') and hasattr(lyr.weight, 'requires_grad'): lyr.weight.requires_grad = not freeze_status
            if hasattr(lyr, 'bias') and hasattr(lyr.bias, 'requires_grad')    : lyr.bias.requires_grad = not freeze_status

def unfreeze(model, lyr_idx=[], lyr_type=[nn.Conv2d]):
    for idx, lyr in enumerate(model.children()):
        if len(lyr_idx) != 0 and idx not in lyr_idx: continue

        if isinstance(lyr, nn.Sequential) or isinstance(lyr, models.resnet.Bottleneck):
            unfreeze(lyr, [], lyr_type=lyr_type)
        
        if True in [isinstance(lyr, t) for t in lyr_type]:
            if hasattr(lyr, 'weight') and hasattr(lyr.weight, 'requires_grad'): lyr.weight.requires_grad = True
            if hasattr(lyr, 'bias') and hasattr(lyr.bias, 'requires_grad'):     lyr.bias.requires_grad = True


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def accuracy(preds, targets, k):
    batch_size = targets.size(0)
    _, ind = preds.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum() 
    return correct_total.item() * (100.0 / batch_size)

class AverageMeter(object):
	def __init__(self, name):
		self.name = name
		self.reset()

	def reset(self):
		self.val, self.sum, self.count, self.avg = 0, 0, 0, 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

	def __repr__(self):
		return f"{self.name.capitalize()} Recorder:\n----------\nValue: {self.val}\nSum: {self.sum}\nCount: {self.count}\nAverage: {self.avg}"

# def save_checkpoint(epoch, encoder, decoder, encoder_optim, decoder_optim, val_res, folder='./model'):
# 	state = {
# 		'epoch': epoch,
# 		'bleu4': val_res,
# 		'encoder': encoder,
# 		'decoder': decoder,
# 		'encoder_optim': encoder_optim,
# 		'decoder_optim': decoder_optim
# 	}

# 	filename = f'{folder}/checkpoint_epoch_{epoch}.pth.tar'
# 	torch.save(state, filename)

def save_checkpoint(epoch, encoder, decoder, encoder_optim, decoder_optim, val_res, folder='./model'):
    state = {
        'epoch': epoch,
        'bleu4': val_res,
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'encoder_optim': encoder_optim.state_dict() if encoder_optim is not None else None,
        'decoder_optim': decoder_optim.state_dict()
    }

    filename = f'{folder}/checkpoint_epoch_{epoch}.pth.tar'
    torch.save(state, filename)


class SaveStats(object):
    def __init__(self, path, load_from_file=False):
        self.train_loss  = []
        self.train_top5  = []
        self.train_lr = []
        self.val_loss  = []
        self.val_top5  = []
        self.val_bleu4 = []
        self.path = path
        self.load_from_file = load_from_file
        if self.load_from_file: self.from_file()

    def from_file(self):
        with open(self.path, 'r') as json_file:
            json_obj = json.load(json_file)

        self.train_loss  = json_obj['train_loss']
        self.train_top5  = json_obj['train_top5']
        self.train_lr = json_obj['train_lr']
        self.val_loss  = json_obj['val_loss']
        self.val_top5  = json_obj['val_top5']
        self.val_bleu4 = json_obj['val_bleu4']

    def to_file(self):
        if os.path.exists(self.path): os.remove(self.path)
        json_obj = {
            'train_loss': self.train_loss,
            'train_top5': self.train_top5,
            'train_lr': self.train_lr,
            'val_loss': self.val_loss,
            'val_top5': self.val_top5,
            'val_bleu4': self.val_bleu4
        }
        with open(self.path, 'w') as json_file:
            json.dump(json_obj, json_file)

    def update(self, mode, loss, top5, bleu4=None, lr=None):
        if mode == 'train':
            self.train_loss.append(loss)
            self.train_top5.append(top5)
            self.train_lr.append(lr)
        elif mode == 'val':
            self.val_loss.append(loss)
            self.val_top5.append(top5)
            self.val_bleu4.append(bleu4)

def set_lr(optim, lr):
    for param_group in optim.param_groups:
        param_group['lr'] = lr
    return optim.param_groups[0]['lr']

def decay_lr(optim, decay_factor):
    for param_group in optim.param_groups:
        param_group['lr'] = param_group['lr'] * decay_factor
    return optim.param_groups[0]['lr']


class SaveStatsDynamic(object):
    def __init__(self, path, data, load_from_file=False):
        self.data = data
        self.path = path
        for d in self.data: setattr(self, d, [])
        if load_from_file: self.from_file()

    def update(self, data_dict):
        for k in data_dict.keys():
            attr = getattr(self, k, None)
            if attr is not None: attr.append(data_dict[k])

    def to_file(self):
        if os.path.exists(self.path): os.remove(self.path)
        json_obj = {}
        for d in self.data:
            json_obj[d] = getattr(self, d)

        with open(self.path, 'w') as json_file:
            json.dump(json_obj, json_file)

    def from_file(self):
        with open(self.path, 'r') as json_file:
            json_obj = json.load(json_file)

        for k in json_obj.keys():
            attr = getattr(self, k)
            attr = json_obj[k]

def mean_square_error(y_pred, y_true):
    return ((y_pred - y_true) ** 2).mean()