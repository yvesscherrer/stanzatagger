# was stanza.models.pos.model

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pack_sequence, PackedSequence

from biaffine import BiaffineScorer
from hlstm import HighwayLSTM
from dropout import WordDropout
from char_model import CharacterModel

class Tagger(nn.Module):
    def __init__(self, args, vocab, emb_matrix=None):
        super().__init__()

        self.vocab = vocab
        self.args = args
        self.use_pretrained = emb_matrix is not None
        self.use_char = args['char_emb_dim'] > 0
        self.use_word = args['word_emb_dim'] > 0
        self.share_hid = args['pos_emb_dim'] < 1
        self.unsaved_modules = []

        def add_unsaved_module(name, module):
            self.unsaved_modules += [name]
            setattr(self, name, module)

        # input layers
        input_size = 0
        if self.use_word:
            # frequent word embeddings
            self.word_emb = nn.Embedding(len(vocab['word']), self.args['word_emb_dim'], padding_idx=0)
            input_size += self.args['word_emb_dim']

        if not self.share_hid:
            # pos embeddings
            self.pos_emb = nn.Embedding(len(vocab['pos']), self.args['pos_emb_dim'], padding_idx=0)

        if self.use_char:
            self.charmodel = CharacterModel(args, vocab, bidirectional=args['char_bidir'])
            ndir = 2 if args['char_bidir'] else 1
            self.trans_char = nn.Linear(ndir * self.args['char_hidden_dim'], self.args['transformed_dim'], bias=False)
            input_size += self.args['transformed_dim']

        if self.use_pretrained:
            # pretrained embeddings, by default this won't be saved into model file
            add_unsaved_module('pretrained_emb', nn.Embedding.from_pretrained(torch.from_numpy(emb_matrix), freeze=True))
            self.trans_pretrained = nn.Linear(emb_matrix.shape[1], self.args['transformed_dim'], bias=False)
            input_size += self.args['transformed_dim']
        
        # recurrent layers
        self.taggerlstm = HighwayLSTM(input_size, self.args['tag_hidden_dim'], self.args['tag_num_layers'], batch_first=True, bidirectional=True, dropout=self.args['dropout'], rec_dropout=self.args['tag_rec_dropout'], highway_func=torch.tanh)
        self.drop_replacement = nn.Parameter(torch.randn(input_size) / np.sqrt(input_size))
        self.taggerlstm_h_init = nn.Parameter(torch.zeros(2 * self.args['tag_num_layers'], 1, self.args['tag_hidden_dim']))
        self.taggerlstm_c_init = nn.Parameter(torch.zeros(2 * self.args['tag_num_layers'], 1, self.args['tag_hidden_dim']))

        # classifiers
        self.pos_hid = nn.Linear(self.args['tag_hidden_dim'] * 2, self.args['deep_biaff_hidden_dim'])
        self.pos_clf = nn.Linear(self.args['deep_biaff_hidden_dim'], len(vocab['pos']))
        self.pos_clf.weight.data.zero_()
        self.pos_clf.bias.data.zero_()

        if self.share_hid:
            clf_constructor = lambda insize, outsize: nn.Linear(insize, outsize)
        else:
            self.feats_hid = nn.Linear(self.args['tag_hidden_dim'] * 2, self.args['composite_deep_biaff_hidden_dim'])
            clf_constructor = lambda insize, outsize: BiaffineScorer(insize, self.args['pos_emb_dim'], outsize)

        self.feats_clf = nn.ModuleList()
        for l in vocab['feats'].lens():
            if self.share_hid:
                self.feats_clf.append(clf_constructor(self.args['deep_biaff_hidden_dim'], l))
                self.feats_clf[-1].weight.data.zero_()
                self.feats_clf[-1].bias.data.zero_()
            else:
                self.feats_clf.append(clf_constructor(self.args['composite_deep_biaff_hidden_dim'], l))

        # criterion
        self.crit = nn.CrossEntropyLoss(ignore_index=0) # ignore padding
        self.drop = nn.Dropout(args['dropout'])
        self.worddrop = WordDropout(args['word_dropout'])

    def forward(self, word, word_mask, wordchars, wordchars_mask, pos, feats, pretrained, word_orig_idx, sentlens, wordlens):
        
        def pack(x):
            return pack_padded_sequence(x, sentlens, batch_first=True)
        
        def get_batch_sizes(sentlens):
            b = []
            for i in range(max(sentlens)):
                c = len([x for x in sentlens if x > i])
                b.append(c)
            return torch.tensor(b)
        
        inputs = []
        if self.use_word:
            word_emb = self.word_emb(word)
            word_emb = pack(word_emb)
            inputs += [word_emb]
            batch_sizes = word_emb.batch_sizes
        else:
            batch_sizes = get_batch_sizes(sentlens)

        if self.use_pretrained:
            pretrained_emb = self.pretrained_emb(pretrained)
            pretrained_emb = self.trans_pretrained(pretrained_emb)
            pretrained_emb = pack(pretrained_emb)
            inputs += [pretrained_emb]

        def pad(x):
            return pad_packed_sequence(PackedSequence(x, batch_sizes), batch_first=True)[0]

        if self.use_char:
            char_reps = self.charmodel(wordchars, wordchars_mask, word_orig_idx, sentlens, wordlens)
            char_reps = PackedSequence(self.trans_char(self.drop(char_reps.data)), char_reps.batch_sizes)
            inputs += [char_reps]

        lstm_inputs = torch.cat([x.data for x in inputs], 1)
        lstm_inputs = self.worddrop(lstm_inputs, self.drop_replacement)
        lstm_inputs = self.drop(lstm_inputs)
        lstm_inputs = PackedSequence(lstm_inputs, inputs[0].batch_sizes)

        lstm_outputs, _ = self.taggerlstm(lstm_inputs, sentlens, hx=(self.taggerlstm_h_init.expand(2 * self.args['tag_num_layers'], word.size(0), self.args['tag_hidden_dim']).contiguous(), self.taggerlstm_c_init.expand(2 * self.args['tag_num_layers'], word.size(0), self.args['tag_hidden_dim']).contiguous()))
        lstm_outputs = lstm_outputs.data

        pos_hid = F.relu(self.pos_hid(self.drop(lstm_outputs)))
        pos_pred = self.pos_clf(self.drop(pos_hid))

        preds = [pad(pos_pred).max(2)[1]]

        pos = pack(pos).data
        loss = self.crit(pos_pred.view(-1, pos_pred.size(-1)), pos.view(-1))

        if self.share_hid:
            feats_hid = pos_hid
            clffunc = lambda clf, hid: clf(self.drop(hid))
        else:
            feats_hid = F.relu(self.feats_hid(self.drop(lstm_outputs)))
            # TODO: self.training is never set, but check if this is a bug
            #if self.training: pos_emb = self.pos_emb(pos) else:
            pos_emb = self.pos_emb(pos_pred.max(1)[1])
            clffunc = lambda clf, hid: clf(self.drop(hid), self.drop(pos_emb))

        feats_preds = []
        feats = pack(feats).data
        for i in range(len(self.vocab['feats'])):
            feats_pred = clffunc(self.feats_clf[i], feats_hid)
            loss += self.crit(feats_pred.view(-1, feats_pred.size(-1)), feats[:, i].view(-1))
            feats_preds.append(pad(feats_pred).max(2, keepdim=True)[1])
        preds.append(torch.cat(feats_preds, 2))

        return loss, preds
