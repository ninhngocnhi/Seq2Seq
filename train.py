import torch
import torch.nn as nn
import torch.optim as optim
from model import Encoder, Decoder, Seq2Seq
from torchtext.legacy.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator

import spacy
import numpy as np
import random
import tqdm

seed = 2124
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 32

spacy_de = spacy.load('de_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')

# convert string to lists of string
def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)][::-1]

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

def init_weight(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

# add strgt and end of sequence
src = Field(tokenize = tokenize_de, init_token= '<sos>', eos_token= '<eos>', lower = True)
trg = Field(tokenize = tokenize_en, init_token= '<sos>', eos_token= '<eos>', lower = True)

# div dataset to train, val , test
train, val, test = Multi30k.splits( exts = ('.de','.en'), fields = (src, trg))

print(f"Number of train, val, test data: {len(train.examples),len(val.examples),len(test.examples)}")
print(vars(train.examples[0]))

# create dictionary with words appears at least 2 times
src.build_vocab(train, min_freq = 2)
trg.build_vocab(train, min_freq = 2)

print(f"len of de dic, en dic: {len(src.vocab), len(trg.vocab)}")

# padding sentence with the same length
train_iter, val_iter, test_iter = BucketIterator.splits((train, val, test), batch_size = batch_size, device = device)

input_dim = len(src.vocab)
out_dim = len(trg.vocab)
emb_dim = 256
hid_dim = 512
n_layers = 2
drop = 0.5

enc = Encoder(input_dim, emb_dim, hid_dim, n_layers, drop)
dec = Decoder(out_dim, emb_dim, hid_dim, n_layers, drop)

model = Seq2Seq(enc, dec, device).to(device)
model.apply(init_weight)

optimizer = optim.Adam(model.parameters())
 
# ignore_index (pad to make sentence have same length)
pad_idx = trg.vocab.stoi[trg.pad_token]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

def train(model, iter, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, batch in tqdm.tqdm(enumerate(iter)):
        src = batch.src
        trg = batch.trg
        optimizer.zero_grad()
        output = model(src, trg)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim) # remove first element = 0 of predict
        trg = trg[1:].view(-1) # remove first element = <sos> of input
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),clip) # train fater
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss/len(iter)

def val(model, iter, criterion, clip):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in tqdm.tqdm(enumerate(iter)):
            src = batch.src
            trg = batch.trg
            output = model(src, trg,0)
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim) # remove first element = 0 of predict
            trg = trg[1:].view(-1) # remove first element = <sos> of input
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss/len(iter)

n_epoch = 1
clip = 1
for i in range(n_epoch):
    train_loss = train(model, train_iter, optimizer, criterion, clip)
    val_loss = val(model, val_iter, criterion)
    print(f'Epoch: {i+1}')
    print(f'\tTrain Loss: {train_loss:.3f}')
    print(f'\t Val. Loss: {val_loss:.3f}')