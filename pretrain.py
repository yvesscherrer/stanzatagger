# was stanza.models.common.pretrain

"""
Supports for pretrained data.
"""
import os, sys
import re

import lzma, gzip
import logging
import numpy as np
import torch

from vocab import BaseVocab, VOCAB_PREFIX

logger = logging.getLogger('stanza')

class PretrainedWordVocab(BaseVocab):
    def build_vocab(self):
        self._id2unit = VOCAB_PREFIX + self.data
        self._unit2id = {w:i for i, w in enumerate(self._id2unit)}


class Pretrain:
    """ A loader and saver for pretrained embeddings. """

    def __init__(self, from_text=None, from_pt=None, max_vocab=None):
        if max_vocab:
            self.max_vocab = max_vocab
        else:
            self.max_vocab = sys.maxsize
        self.vocab = None
        self.emb = None
        if from_text:
            self.load_from_text(from_text)
        elif from_pt:
            self.load_from_pt(from_pt)
    

    def load_from_pt(self, filename):
        if not filename:
            logger.warning("Cannot load pretrained embeddings, no file name given.")
            return
        if not os.path.exists(filename):
            logger.warning("Cannot load pretrained embeddings, file not found: {}".format(filename))
            return
        
        data = torch.load(filename, lambda storage, loc: storage)
        logger.info("Loaded pretrained embeddings from {}".format(filename))
        self.vocab = PretrainedWordVocab.load_state_dict(data['vocab'])
        self.emb = data['emb']
    

    def save_to_pt(self, filename):
        if not filename:
            logger.warning("Cannot save pretrained embeddings, no file name given.")
            return
        
        # should not infinite loop since the load function sets _vocab and _emb before trying to save
        data = {'vocab': self.vocab.state_dict(), 'emb': self.emb}
        try:
            torch.save(data, filename, _use_new_zipfile_serialization=False, pickle_protocol=3)
            logger.info("Saved pretrained vocab and vectors to {}".format(filename))
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as e:
            logger.warning("Saving pretrained data failed due to the following exception... continuing anyway.\n\t{}".format(e))
    

    def load_from_text(self, filename):
        if not filename:
            logger.warning("Cannot load pretrained embeddings, no file name given.")
            return
        if not os.path.exists(filename):
            logger.warning("Cannot load pretrained embeddings, file not found: {}".format(filename))
            return
        
        logger.info("Loading pretrained vectors from {}".format(filename))
        if filename.endswith(".xz"):
            words, emb, failed = self.read_from_file(filename, open_func=lzma.open)
        elif filename.endswith(".gz"):
            words, emb, failed = self.read_from_file(filename, open_func=gzip.open)
        else:
            words, emb, failed = self.read_from_file(filename, open_func=open)
        
        if failed > 0: # recover failure
            emb = emb[:-failed]
        if len(emb) - len(VOCAB_PREFIX) != len(words):
            raise Exception("Loaded number of vectors does not match number of words.")
        
        # Use a fixed vocab size
        if self.max_vocab > len(VOCAB_PREFIX) and self.max_vocab < len(words):
            words = words[:self.max_vocab - len(VOCAB_PREFIX)]
            emb = emb[:self.max_vocab]
        self.emb = emb
        self.vocab = PretrainedWordVocab(words)
    

    def read_from_file(self, filename, open_func=open):
        """
        Open a vector file using the provided function and read from it.
        """
        # some vector files, such as Google News, use tabs
        tab_space_pattern = re.compile(r"[ \t]+")
        first = True
        words = []
        failed = 0
        with open_func(filename, 'rb') as f:
            for i, line in enumerate(f):
                try:
                    line = line.decode()
                except UnicodeDecodeError:
                    failed += 1
                    continue
                if first:
                    # the first line contains the number of word vectors and the dimensionality
                    first = False
                    line = line.strip().split(' ')
                    rows, cols = [int(x) for x in line]
                    emb = np.zeros((rows + len(VOCAB_PREFIX), cols), dtype=np.float32)
                    continue

                line = tab_space_pattern.split((line.rstrip()))
                emb[i+len(VOCAB_PREFIX)-1-failed, :] = [float(x) for x in line[-cols:]]
                words.append(' '.join(line[:-cols]))
        return words, emb, failed


if __name__ == '__main__':
    with open('test.txt', 'w') as fout:
        fout.write('3 2\na 1 1\nb -1 -1\nc 0 0\n')
    # 1st load: save to pt file
    pretrain = Pretrain()
    pretrain.load_from_text('test.txt')
    print(pretrain.emb)
    pretrain.save_to_pt('test.pt')
    # verify pt file
    x = torch.load('test.pt')
    print(x)
    # 2nd load: load saved pt file
    pretrain = Pretrain()
    pretrain.load_from_pt('test.pt')
    print(pretrain.emb)

