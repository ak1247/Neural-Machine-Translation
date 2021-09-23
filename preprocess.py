import spacy
import torchtext
import dill
from torchtext.datasets import IWSLT2016
from torchtext.legacy import data,datasets


lang_src = 'en_core_web_sm'
lang_trg = 'de_core_news_sm'
MAX_LEN = 100
MIN_FREQ = 3
data_out = "data/m30k_en2de.pkl"

src_lang_model = spacy.load(lang_src)
trg_lang_model = spacy.load(lang_trg)

def tokenize_src(text):
    return [tok.text for tok in src_lang_model.tokenizer(text)]

def tokenize_trg(text):
    return [tok.text for tok in trg_lang_model.tokenizer(text)]

SRC = data.Field(tokenize=tokenize_src,lower=False,
                pad_token='<blank>',init_token='<s>', eos_token='</s>')
TRG = data.Field(tokenize=tokenize_trg,lower=False,
                pad_token='<blank>',init_token='<s>', eos_token='</s>')

def filter_examples_with_length(x):
    return len(vars(x)['src']) <= MAX_LEN and len(vars(x)['trg']) <= MAX_LEN

## Data Split
train,val,test = datasets.Multi30k.splits(
            exts = ('.en', '.de'),
            fields = (SRC, TRG),
            filter_pred=filter_examples_with_length)
# train,val,test = datasets.WMT14.splits(
#             exts = ('.en', '.de'),
#             fields = (SRC, TRG),
#             train='train', validation='newstest2013', test='newstest2013',
#             filter_pred=filter_examples_with_length)
# train,val,test = datasets.IWSLT.splits(
#             exts = ('.de', '.en'),
#             fields = (SRC, TRG),
#             train='train.de-en', validation='tst2013.de-en', test='tst2014.de-en',
#             filter_pred=filter_examples_with_length)

# train,val,test = torchtext.datasets.IWSLT2016(root='.data', split=('train', 'valid', 'test'), language_pair=('de', 'en'), valid_set='tst2013', test_set='tst2014')
## Vocabulary
SRC.build_vocab(train.src, min_freq=MIN_FREQ)
TRG.build_vocab(train.trg, min_freq=MIN_FREQ)
print('Vocab size:: Source:{}, Target:{}'.format(len(SRC.vocab),len(TRG.vocab)))

## Share vocab
for w, _ in SRC.vocab.stoi.items():
    if w not in TRG.vocab.stoi:
        TRG.vocab.stoi[w] = len(TRG.vocab.stoi)
TRG.vocab.itos = [None] * len(TRG.vocab.stoi)
for w, i in TRG.vocab.stoi.items():
    TRG.vocab.itos[i] = w
SRC.vocab.stoi = TRG.vocab.stoi
SRC.vocab.itos = TRG.vocab.itos 
print("After sharing  vocabulary..")
print('Vocab size:: Source:{}, Target:{}'.format(len(SRC.vocab),len(TRG.vocab)))

data = {'vocab': {'src': SRC, 'trg': TRG},
        'train': train.examples,
        'valid': val.examples,
        'test': test.examples}

dill.dump(data, open(data_out, 'wb'))
print("Data saved in "+data_out)