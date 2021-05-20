# was stanza.models.pos.data and stanza.models.common.data

import random
import logging
import torch

from vocab import PAD_ID, UNK_ID, CharVocab, WordVocab, FeatureVocab, MultiVocab

logger = logging.getLogger('stanza')

def map_to_ids(tokens, voc):
    ids = [voc[t] if t in voc else UNK_ID for t in tokens]
    return ids

def get_long_tensor(tokens_list, batch_size, pad_id=PAD_ID):
    """ Convert (list of )+ tokens to a padded LongTensor. """
    sizes = []
    x = tokens_list
    while isinstance(x[0], list):
        sizes.append(max(len(y) for y in x))
        x = [z for y in x for z in y]
    tokens = torch.LongTensor(batch_size, *sizes).fill_(pad_id)
    for i, s in enumerate(tokens_list):
        tokens[i, :len(s)] = torch.LongTensor(s)
    return tokens

def get_float_tensor(features_list, batch_size):
    if features_list is None or features_list[0] is None:
        return None
    seq_len = max(len(x) for x in features_list)
    feature_len = len(features_list[0][0])
    features = torch.FloatTensor(batch_size, seq_len, feature_len).zero_()
    for i,f in enumerate(features_list):
        features[i,:len(f),:] = torch.FloatTensor(f)
    return features

def sort_all(batch, lens):
    """ Sort all fields by descending order of lens, and return the original indices. """
    if batch == [[]]:
        return [[]], []
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2:], sorted_all[1]

def unsort(sorted_list, oidx):
    """
    Unsort a sorted list, based on the original idx.
    """
    assert len(sorted_list) == len(oidx), "Number of list elements must match with original indices."
    if len(sorted_list) == 0:
        return []
    _, unsorted = [list(t) for t in zip(*sorted(zip(oidx, sorted_list)))]
    return unsorted


class DataLoader:
    def __init__(self, doc, batch_size, vocab=None, pretrain=None, word_cutoff=7, evaluation=False):
        self.batch_size = batch_size
        self.word_cutoff = word_cutoff
        self.eval = evaluation

        # get data from document
        data = doc.provide_data()

        # handle vocab
        if vocab is None:
            self.vocab = self.init_vocab(data)
        else:
            self.vocab = vocab
        
        # handle pretrain
        self.pretrain_vocab = pretrain.vocab if pretrain else None

        data = self.preprocess(data, self.vocab, self.pretrain_vocab)
        # shuffle for training
        if not self.eval:
            random.shuffle(data)
        self.num_examples = len(data)

        # chunk into batches
        self.data = self.chunk_batches(data)
        logger.info("{} batches created from {} sentences (batch size: {})".format(len(self.data), self.num_examples, self.batch_size))

    def init_vocab(self, data):
        assert self.eval == False # for eval vocab must exist
        charvocab = CharVocab(data, idx=0)
        wordvocab = WordVocab(data, idx=1, cutoff=self.word_cutoff, lower=True)
        posvocab = WordVocab(data, idx=2)
        featsvocab = FeatureVocab(data, idx=3)
        vocab = MultiVocab({'char': charvocab,
                            'word': wordvocab,
                            'pos': posvocab,
                            'feats': featsvocab})
        return vocab

    def preprocess(self, data, vocab, pretrain_vocab):
        processed = []
        for sent in data:
            processed_sent = [vocab['word'].map([w[1] for w in sent])]
            processed_sent += [[vocab['char'].map([x for x in w[0]]) for w in sent]]
            processed_sent += [vocab['pos'].map([w[2] for w in sent])]
            processed_sent += [vocab['feats'].map([w[3] for w in sent])]
            if pretrain_vocab is not None:
                # always use lowercase lookup in pretrained vocab
                processed_sent += [pretrain_vocab.map([w[1].lower() for w in sent])]
            else:
                processed_sent += [[PAD_ID] * len(sent)]
            processed.append(processed_sent)
        return processed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch_size = len(batch)
        batch = list(zip(*batch))
        assert len(batch) == 5

        # sort sentences by lens for easy RNN operations
        lens = [len(x) for x in batch[0]]
        batch, orig_idx = sort_all(batch, lens)

        # sort words by lens for easy char-RNN operations
        batch_words = [w for sent in batch[1] for w in sent]
        word_lens = [len(x) for x in batch_words]
        batch_words, word_orig_idx = sort_all([batch_words], word_lens)
        batch_words = batch_words[0]
        word_lens = [len(x) for x in batch_words]

        # convert to tensors
        words = batch[0]
        words = get_long_tensor(words, batch_size)
        words_mask = torch.eq(words, PAD_ID)
        wordchars = get_long_tensor(batch_words, len(word_lens))
        wordchars_mask = torch.eq(wordchars, PAD_ID)

        pos = get_long_tensor(batch[2], batch_size)
        feats = get_long_tensor(batch[3], batch_size)
        pretrained = get_long_tensor(batch[4], batch_size)
        sentlens = [len(x) for x in batch[0]]
        return words, words_mask, wordchars, wordchars_mask, pos, feats, pretrained, orig_idx, word_orig_idx, sentlens, word_lens

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def reshuffle(self):
        data = [y for x in self.data for y in x]
        self.data = self.chunk_batches(data)
        random.shuffle(self.data)

    def chunk_batches(self, data):
        res = []

        # sort sentences (roughly) by length for better memory utilization
        if self.eval:
            (data, ), self.data_orig_idx = sort_all([data], [len(x[0]) for x in data])
        else:
            data = sorted(data, key = lambda x: len(x[0]), reverse=random.random() > .5)            

        if self.batch_size < 0:
            res = [[x] for x in data]
        else:
            current = []
            currentlen = 0
            for x in data:
                if len(x[0]) + currentlen > self.batch_size and currentlen > 0:
                    res.append(current)
                    current = []
                    currentlen = 0
                current.append(x)
                currentlen += len(x[0])

            if currentlen > 0:
                res.append(current)

        return res
