import dill
import torch
import numpy as np
import torch.nn.functional as F
from torchtext.legacy.data import Dataset,BucketIterator

def dataLoaders(opt,device):
    batch_size = opt.BS
    data = dill.load(open(opt.data_pkl, 'rb'))

    opt.src_pad_idx = data['vocab']['src'].vocab.stoi['<blank>']
    opt.trg_pad_idx = data['vocab']['trg'].vocab.stoi['<blank>']

    opt.src_vocab_size = len(data['vocab']['src'].vocab)
    opt.trg_vocab_size = len(data['vocab']['trg'].vocab)

    assert data['vocab']['src'].vocab.stoi == data['vocab']['trg'].vocab.stoi, \
        'To sharing word embedding the src/trg word2idx table shall be the same.'

    fields = {'src': data['vocab']['src'], 'trg':data['vocab']['trg']}

    train = Dataset(examples=data['train'], fields=fields)
    val = Dataset(examples=data['valid'], fields=fields)

    train_iterator = BucketIterator(train, batch_size=batch_size, device=device, train=True)
    val_iterator = BucketIterator(val, batch_size=batch_size, device=device)

    return train_iterator, val_iterator

## Scheduled optmizer
class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, lr_mul, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.lr_mul = lr_mul
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0


    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()


    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self._optimizer.zero_grad()


    def _get_lr_scale(self):
        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))


    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_steps += 1
        lr = self.lr_mul * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

## Loss functions
def cal_loss(pred,gold,trg_pad_idx,smoothing):
    
    if smoothing:
        eps = 0.1
        n_class = pred.size(1)
        
        one_hot = torch.zeros_like(pred).scatter(1,gold.view(-1,1),1)
        one_hot = one_hot*(1-eps) + (1-one_hot)*eps /(n_class-1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(trg_pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction='sum')
    return loss

def cal_performance(pred,gold,trg_pad_idx,smoothing):
    gold = gold.contiguous().view(-1)
    loss = cal_loss(pred,gold,trg_pad_idx,smoothing)

    pred = pred.max(1)[1]
    non_pad_mask = gold.ne(trg_pad_idx)

    n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()

    return loss,n_correct,n_word