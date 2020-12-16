import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# load libraries
import nltk
from nltk.corpus import PlaintextCorpusReader

from nltk.util import ngrams
from nltk.lm.preprocessing import pad_both_ends

from tqdm import tqdm

# ngram:
_N = 3

# create a corpus reader
# this includes: sentence segmentation and word tokenization:
wikitext2 = PlaintextCorpusReader(
    'wikitext-2',
    ['wiki.train.raw', 'wiki.valid.raw', 'wiki.test.raw'],
)
word_tokenizer = wikitext2._word_tokenizer

# training and test split:
train = wikitext2.sents('wiki.train.raw')
test = wikitext2.sents('wiki.test.raw')

# the vocabulary based on the training data:
vocab = nltk.lm.Vocabulary([
    word
    for sent in train
    for word in sent
], unk_cutoff=1)


# build n-grams
def build_ngrams(sent, n):
    # pad both ends for corner-ngrams:
    sent = ['<s>']*(n-1) + [word if word in vocab else vocab.unk_label for word in sent] + ['</s>']*(n-1)
    # build the ngrams:
    return list(ngrams(sent, n))


import torch # neural network framework

# encoding the tokens:
vocab_list = [word for word, freq in vocab.counts.most_common() if freq > 1]
word2idx = {word: idx for idx, word in enumerate(['<s>', '</s>', vocab.unk_label]+vocab_list)}
idx2word = {idx: word for idx, word in enumerate(['<s>', '</s>', vocab.unk_label]+vocab_list)}

def token_encoder(tokens):
    if type(tokens) in {list, tuple}:
        return [word2idx[token] if token in word2idx else word2idx[vocab.unk_label] for token in tokens]
    elif type(tokens) == str:
        token = tokens
        return word2idx[token] if token in word2idx else word2idx[vocab.unk_label]
    print(type(tokens))

# moving window language model:
# https://jmlr.org/papers/volume3/tmp/bengio03a.pdf
class BengioLM(torch.nn.Module):
    def __init__(self, context_size=2, dim=50):
        super(BengioLM, self).__init__()
        # defining the parameters of the model
        self.C = torch.nn.Embedding(len(word2idx), dim).cuda() # C
        self.Hx_d = torch.nn.Linear(context_size*dim, dim).cuda() # d, H
        self.tanh = torch.nn.Tanh()
        self.Wx_Uf_b = torch.nn.Linear((context_size + 1) * dim, len(word2idx)).cuda() # b, U, W
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        self.loss_fn = torch.nn.NLLLoss() # negative-log-likelihood loss

    def forward(self, context, target_idx=None):
        # function of the model
        batch_size = context.shape[0]
        x = self.C(context).view(batch_size,-1)
        x = torch.cat([x, self.tanh(self.Hx_d(x))], dim=-1)
        logprob = self.logsoftmax(self.Wx_Uf_b(x))
        if target_idx is None:
            return logprob
        else:
            loss = self.loss_fn(logprob, target_idx)
            return logprob, loss


model = BengioLM()
optimizer = torch.optim.Adam(model.parameters())

sentence_batch_size = 64

for e in range(10):
    progress_bar = tqdm(range(0, len(train), sentence_batch_size))
    for i in progress_bar:
        optimizer.zero_grad()
        contexts, targets = zip(*[
            (token_encoder(tokens[:-1]), token_encoder(tokens[-1]))
            for sent in train[i:i+sentence_batch_size]
            for tokens in build_ngrams(sent, 3)
        ])

        logprob, loss = model.forward(torch.tensor(contexts).cuda(), torch.tensor(targets).cuda())
        progress_bar.set_description_str(f"epoch={e+1},loss={loss.item():.3f}")
        progress_bar.update()
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), 'model.pt')

