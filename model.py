import torch
import random
import torch.nn as nn
import torch.functional as F
from torch.nn.modules import dropout
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, drop):
# input_dim is the dimensionality of the one-hot input vectors, equal to the input vocabulary size.
# emb_dim is the dimensionality of the embedding layer (converts the one-hot vectors into dense vectors with emb_dim dimensions)
# hid_dim is the dimensionality of the hidden and cell states.
# n_layers is the number of layers in the RNN.
        super(Encoder,self).__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.emb = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = drop)
        self.dro = nn.Dropout(drop)
    
    def forward(self, x):
# x = [len(x), batch_size]
        embedding = self.dro(self.emb(x)) 
# embedding = [len(x), batch_size, enb_dim]
        out, (hid, cell)  = self.rnn(embedding)
#out = [len(x), batch_size, hid_dim * n_directions]
#hid = [n_layers * n_directions, batch_size, hid_dim]
#cell = [n_layers * n_directions, batch_size, hid_dim]
        return hid, cell

class Decoder(nn.Module):
    def __init__(self, out_dim, emb_dim, hid_dim, n_layers, drop):
        super(Decoder,self).__init__()
        self.out_dim = out_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.emb = nn.Embedding(out_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = drop)
        self.li = nn.Linear(hid_dim, out_dim)
        self.dro = nn.Dropout(drop)
        self.pro = nn.LogSoftmax(dim=1)

    def forward(self, x, hid, cell):
        x = x.unsqueeze(0)
        # x = [batch_size]
        embedding = self.dro(self.emb(x)) 
# embedding = [1, batch_size, enb_dim]
        out, (hid, cell)  = self.rnn(embedding, (hid,cell))
#out = [1, batch_size, hid_dim]
#hid = [n_layers, batch_size, hid_dim]
#cell = [n_layers, batch_size, hid_dim]
        out = self.li(out.squeeze(0))
        return out, hid, cell
    
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim 
        assert encoder.n_layers == decoder.n_layers
    def forward(self, src, tar, teach_forcing_ratio = 0.5):
        batch_size = tar.shape[1]
        tar_len = tar.shape[0]
        tar_vocal_size = self.decoder.out_dim
        output = torch.zeros(tar_len, batch_size, tar_vocal_size).to(self.device)
        hid, cell = self.encoder(src)
        input = tar[0,:]
        for t in range(1, tar_len):
            out, hid, cell = self.decoder(input, hid, cell)
            output[t] = out
            teacher_force = random.random() < teach_forcing_ratio
            top1 = out.argmax(1)
            input = tar[t] if teacher_force else top1
        return output

