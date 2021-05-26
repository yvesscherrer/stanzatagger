# was stanza.models.common.trainer and stanza.models.pos.trainer

"""
A trainer class to handle training and testing of models.
"""

import logging
import torch

from model import Tagger
from vocab import MultiVocab, UNK_ID
import data

logger = logging.getLogger('stanza')

def unpack_batch(batch, use_cuda):
    """ Unpack a batch from the data loader. """
    if use_cuda:
        inputs = [b.cuda() if b is not None else None for b in batch[:7]]
    else:
        inputs = batch[:7]
    orig_idx = batch[7]
    word_orig_idx = batch[8]
    sentlens = batch[9]
    wordlens = batch[10]
    return inputs, orig_idx, word_orig_idx, sentlens, wordlens


# class Trainer(BaseTrainer):
class Trainer(object):
    """ A trainer for training models. """
    def __init__(self, model_file=None, vocab=None, pretrain=None, args=None, use_cuda=False):
        if model_file is not None:
            # load trained model from file
            checkpoint = torch.load(model_file, lambda storage, loc: storage)
            self.vocab = MultiVocab.load_state_dict(checkpoint['vocab'])
            self.model = Tagger(checkpoint['config'], self.vocab, emb_matrix=pretrain.emb if pretrain else None)
            self.model.load_state_dict(checkpoint['model'], strict=False)
            #TODO: load optimizer here
            #self.optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            # build model from scratch
            self.vocab = vocab
            self.model = Tagger(args, vocab, emb_matrix=pretrain.emb if pretrain else None)

        if args is not None:
            self.set_optimizer(args['optim'], args['lr'], betas=(0.9, args['beta2']), eps=1e-6)
            self.max_grad_norm = args['max_grad_norm']

        self.use_cuda = use_cuda
        if self.use_cuda:
            self.model.cuda()
        else:
            self.model.cpu()

    def set_optimizer(self, name, lr, betas=(0.9, 0.999), eps=1e-8, momentum=0):
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        if name == 'sgd':
            self.optimizer = torch.optim.SGD(parameters, lr=lr, momentum=momentum)
        elif name == 'adagrad':
            self.optimizer = torch.optim.Adagrad(parameters, lr=lr)
        elif name == 'adam':
            self.optimizer = torch.optim.Adam(parameters, lr=lr, betas=betas, eps=eps)
        elif name == 'amsgrad':
            self.optimizer = torch.optim.Adam(parameters, amsgrad=True, lr=lr, betas=betas, eps=eps)
        elif name == 'adamax':
            self.optimizer = torch.optim.Adamax(parameters) # use default lr
        else:
            raise Exception("Unsupported optimizer: {}".format(name))

    def update(self, batch, eval=False):
        inputs, orig_idx, word_orig_idx, sentlens, wordlens = unpack_batch(batch, self.use_cuda)
        word, word_mask, wordchars, wordchars_mask, pos, feats, pretrained = inputs

        if eval:
            self.model.eval()
        else:
            self.model.train()
            self.optimizer.zero_grad()
        loss, _ = self.model(word, word_mask, wordchars, wordchars_mask, pos, feats, pretrained, word_orig_idx, sentlens, wordlens)
        loss_val = loss.data.item()
        if eval:
            return loss_val

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return loss_val

    def predict(self, batch, unsort=True):
        inputs, orig_idx, word_orig_idx, sentlens, wordlens = unpack_batch(batch, self.use_cuda)
        word, word_mask, wordchars, wordchars_mask, pos, feats, pretrained = inputs

        self.model.eval()
        batch_size = word.size(0)
        loss, preds = self.model(word, word_mask, wordchars, wordchars_mask, pos, feats, pretrained, word_orig_idx, sentlens, wordlens)
        pos_seqs = [self.vocab['pos'].unmap(sent) for sent in preds[0].tolist()]
        feats_seqs = [self.vocab['feats'].unmap(sent) for sent in preds[1].tolist()]
        w_unk_seqs = [[(not self.model.use_word) or tokid == UNK_ID for tokid in sent] for sent in word]
        p_unk_seqs = [[(not self.model.use_pretrained) or tokid == UNK_ID for tokid in sent] for sent in pretrained]

        pred_tokens = [[[pos_seqs[i][j], feats_seqs[i][j], w_unk_seqs[i][j] and p_unk_seqs[i][j]] for j in range(sentlens[i])] for i in range(batch_size)]
        if unsort:
            pred_tokens = data.unsort(pred_tokens, orig_idx)
        return pred_tokens, loss.data.item()

    def save(self, filename, skip_modules=True):
        model_state = self.model.state_dict()
        # skip saving modules like pretrained embeddings, because they are large and will be saved in a separate file
        if skip_modules:
            skipped = [k for k in model_state.keys() if k.split('.')[0] in self.model.unsaved_modules]
            for k in skipped:
                del model_state[k]
        params = {
                'model': model_state,
                'vocab': self.vocab.state_dict(),
                #TODO: check if it's useful to save the optimizer
                #'optimizer': self.optimizer.state_dict(),
                'config': self.model.args
                }
        try:
            torch.save(params, filename, _use_new_zipfile_serialization=False)
            logger.info("Model saved to {}".format(filename))
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            logger.warning(f"Saving failed... {e} continuing anyway.")
