# was stanza.models.common.hlstm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pack_sequence, PackedSequence

from packed_lstm import PackedLSTM

# Highway LSTM network
class HighwayLSTM(nn.Module):
    """
    A Highway LSTM network, as used in the original Tensorflow version of the Dozat parser.
    """
    def __init__(self, input_size, hidden_size,
                 num_layers=1, bias=True, batch_first=False,
                 dropout=0, bidirectional=False, rec_dropout=0, highway_func=None, pad=False):
        super(HighwayLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.dropout_state = {}
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.highway_func = highway_func
        self.pad = pad

        self.lstm = nn.ModuleList()
        self.highway = nn.ModuleList()
        self.gate = nn.ModuleList()
        self.drop = nn.Dropout(dropout, inplace=True)

        in_size = input_size
        for l in range(num_layers):
            self.lstm.append(PackedLSTM(in_size, hidden_size, num_layers=1, bias=bias,
                batch_first=batch_first, dropout=0, bidirectional=bidirectional, rec_dropout=rec_dropout))
            self.highway.append(nn.Linear(in_size, hidden_size * self.num_directions))
            self.gate.append(nn.Linear(in_size, hidden_size * self.num_directions))
            self.highway[-1].bias.data.zero_()
            self.gate[-1].bias.data.zero_()
            in_size = hidden_size * self.num_directions

    def forward(self, input, seqlens, hx=None):
        highway_func = (lambda x: x) if self.highway_func is None else self.highway_func

        hs = []
        cs = []

        if not isinstance(input, PackedSequence):
            input = pack_padded_sequence(input, seqlens, batch_first=self.batch_first)

        for l in range(self.num_layers):
            if l > 0:
                input = PackedSequence(self.drop(input.data), input.batch_sizes)
            layer_hx = (hx[0][l * self.num_directions:(l+1)*self.num_directions], hx[1][l * self.num_directions:(l+1)*self.num_directions]) if hx is not None else None
            h, (ht, ct) = self.lstm[l](input, seqlens, layer_hx)

            hs.append(ht)
            cs.append(ct)

            input = PackedSequence(h.data + torch.sigmoid(self.gate[l](input.data)) * highway_func(self.highway[l](input.data)), input.batch_sizes)

        if self.pad:
            input = pad_packed_sequence(input, batch_first=self.batch_first)[0]
        return input, (torch.cat(hs, 0), torch.cat(cs, 0))

if __name__ == "__main__":
    T = 10
    bidir = True
    num_dir = 2 if bidir else 1
    rnn = HighwayLSTM(10, 20, num_layers=2, bidirectional=True)
    input = torch.randn(T, 3, 10)
    hx = torch.randn(2 * num_dir, 3, 20)
    cx = torch.randn(2 * num_dir, 3, 20)
    output = rnn(input, (hx, cx))
    print(output)
