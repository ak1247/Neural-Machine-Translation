import os
import math
import torch
import argparse
from torch import optim
from tqdm import tqdm
## User defined
from utils.data import dataLoaders,ScheduledOptim,cal_performance
from utils.models import Transformer

### Functions
## Patching
def patch_src(src):
    src = src.transpose(0, 1)
    return src

def patch_trg(trg):
    trg = trg.transpose(0, 1)
    trg, gold = trg[:,:-1], trg[:,1:].contiguous().view(-1)
    return trg, gold

def train_epoch(model,data_loader,optimizer,args,device):
        model.train()
        total_loss,n_word_total,n_word_correct = 0,0,0
        for batch in tqdm(data_loader,mininterval=2,desc='-(Training)-',leave=False):
                ## Data preparation
                src_seq = patch_src(batch.src).to(device)
                trg_seq, gold = map(lambda x: x.to(device), patch_trg(batch.trg))

                optimizer.zero_grad()
                pred = model(src_seq,trg_seq)
                
                loss,n_correct,n_word = cal_performance(
                        pred,gold,args.trg_pad_idx,smoothing=True)
                loss.backward()
                optimizer.step_and_update_lr()

                total_loss += loss.item()
                n_word_total += n_word
                n_word_correct += n_correct

        loss_per_word = total_loss/n_word_total
        accuracy = n_word_correct/n_word_total

        return loss_per_word,accuracy

def eval_epoch(model,data_loader,args,device):
        model.eval()
        total_loss,n_word_total,n_word_correct = 0,0,0
        for batch in tqdm(data_loader,mininterval=2,desc='-(Training)-',leave=False):
                ## Data preparation
                src_seq = patch_src(batch.src).to(device)
                trg_seq, gold = map(lambda x: x.to(device), patch_trg(batch.trg))

                pred = model(src_seq,trg_seq)
                loss,n_correct,n_word = cal_performance(
                        pred,gold,args.trg_pad_idx,smoothing=True)

                total_loss += loss.item()
                n_word_total += n_word
                n_word_correct += n_correct

        loss_per_word = total_loss/n_word_total
        accuracy = n_word_correct/n_word_total

        return loss_per_word,accuracy
        

parser = argparse.ArgumentParser()
parser.add_argument('-data_pkl', required=True)
args = parser.parse_args()
args.output_dir = 'saved/'+args.data_pkl
## Hyper parameters
args.D_MODEL = 512
args.D_FF = 2048
args.BS = 96
args.MAX_SQN_LEN = 100
args.N_LAYERS = 12
args.N_HEADS = 8
args.dropout = 0.1
args.lr_mul = 2.0
args.EPOCHS = 500
args.N_WARMUP_STEPS = 128000

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print("Working device:",device)

if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

## Dataloaders
dl_train,dl_val = dataLoaders(args,device)

### MOdel
transformer = Transformer(
        args.src_vocab_size,args.trg_vocab_size,
        args.src_pad_idx,args.trg_pad_idx,
        args.D_MODEL,
        args.D_FF,
        args.N_HEADS,
        args.N_LAYERS,
        args.dropout,
        trg_emb_prj_weight_sharing=True,emb_src_trg_weight_sharing=True,
        scale_emb_or_prj='prj'
        ).to(device)

## Optimizer
optimizer = ScheduledOptim(
        optim.Adam(transformer.parameters(),betas=(0.9,0.98),eps=1e-09),
        args.lr_mul,args.D_MODEL,args.N_WARMUP_STEPS)

## Log file
log_file = os.path.join(args.output_dir,'log.txt')
### Training
valid_losses,train_losses = [],[]
valid_accuracies,train_accuracies = [],[]
valid_ppls,train_ppls = [],[]
for epoch in range(1,args.EPOCHS+1):
        train_loss,train_acc = train_epoch(transformer,dl_train,optimizer,args,device)
        train_ppl = math.exp(min(train_loss,100))

        lr = optimizer._optimizer.param_groups[0]['lr']

        valid_loss,valid_acc = eval_epoch(transformer,dl_val,args,device)
        valid_ppl = math.exp(min(valid_loss,100))

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        train_ppls.append(train_ppl)

        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_acc)
        valid_ppls.append(valid_ppl)
        
        print('Epcoh:{}/{}, lr:{}'.format(epoch,args.EPOCHS,lr))
        print("Train:: loss:{}, ppl:{}, acc:{}".format(train_loss,train_ppl,train_acc*100))
        print("Val:: loss:{}, ppl:{}, acc:{}".format(valid_loss,valid_ppl,valid_acc*100))

        with open(log_file,'a') as log:
                log.write('Epoch:{}/{},lr:{}\nTrain:: loss:{}, ppl:{}, acc:{},\
                        Val:: loss:{}, ppl:{}, acc:{}\n'.format(
                        epoch,args.EPOCHS,lr,train_loss,train_ppl,train_acc*100,
                        valid_loss,valid_ppl,valid_acc*100))

        ## Save model
        checkpoint = {'epoch':epoch,'settings':args,'model':transformer.state_dict()}
        if epoch==1 or valid_loss<=min(valid_losses):
                model_name = 'model.chkpt'
                torch.save(checkpoint,os.path.join(args.output_dir,model_name))
                print('Model saved')

print(train_losses)
print(train_accuracies)
print(train_ppls)

print(valid_losses)
print(valid_accuracies)
print(valid_ppls)

with open(log_file,'a') as log:
        log.write(valid_losses)
        log.write(valid_accuracies)
        log.write(valid_ppls)