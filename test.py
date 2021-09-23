import argparse
import dill
import torch
import sacrebleu
from sacremoses import MosesDetokenizer
from tqdm import tqdm
from torchtext.legacy.data import Dataset
## My function
from utils.models import Transformer
from utils.translator import Translator

##
def load_model(opt,device):
    checkpoint = torch.load(opt.model,map_location=device)
    model_opt = checkpoint['settings']

    model = Transformer(
        model_opt.src_vocab_size,model_opt.trg_vocab_size,
        model_opt.src_pad_idx,model_opt.trg_pad_idx,
        model_opt.D_MODEL,
        model_opt.D_FF,
        model_opt.N_HEADS,
        model_opt.N_LAYERS,
        model_opt.dropout,
        trg_emb_prj_weight_sharing=True,emb_src_trg_weight_sharing=True,
        scale_emb_or_prj='prj'
        ).to(device)

    model.load_state_dict(checkpoint['model'])
    return model

parser = argparse.ArgumentParser()
parser.add_argument('-model',required=True)
parser.add_argument('-data_pkl',required=True)
parser.add_argument('-trg_lang',required=True)
parser.add_argument('-output',default='prediction.txt')

args = parser.parse_args()
## Hyper parameters
args.MAX_SQN_LEN = 100
args.BEAM_SIZE = 5

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Working device:",device)

data = dill.load(open(args.data_pkl,'rb'))
SRC,TRG = data['vocab']['src'],data['vocab']['trg']
args.src_pad_idx = SRC.vocab.stoi['<blank>']
args.trg_pad_idx = TRG.vocab.stoi['<blank>']
args.trg_bos_idx = TRG.vocab.stoi['<s>']
args.trg_eos_idx = TRG.vocab.stoi['</s>']

test_loader = Dataset(examples=data['test'], fields={'src': SRC, 'trg': TRG})

## Translator
model = load_model(args,device)
translator = Translator(model,args).to(device)

md = MosesDetokenizer(lang=args.trg_lang)
## Translation
refs = []
preds = []
unk_idx = SRC.vocab.stoi[SRC.unk_token]
with open(args.output,'w') as f:
    for example in tqdm(test_loader, mininterval=2, desc=' - (Test) - ', leave=False):
        src_seq = [SRC.vocab.stoi.get(word, unk_idx) for word in example.src]
        trg_line = md.detokenize(example.trg)
        refs.append(trg_line)
        pred_seq = translator.translate_sentence(torch.LongTensor([src_seq]).to(device))
        pred_line = ' '.join(TRG.vocab.itos[idx] for idx in pred_seq)
        pred_line = pred_line.replace('<s>', '').replace('</s>', '')
        pred_line = pred_line.strip()
        preds.append(pred_line)
        f.write(pred_line+'\n')
refs = [refs]
print("Translation fisnished and saved in {}".format(args.output))

bleu = sacrebleu.corpus_bleu(preds,refs)
print("Bleu score : {}".format(bleu.score))         