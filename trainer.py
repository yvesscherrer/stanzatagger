# was stanza.models.common.trainer and stanza.models.pos.trainer

"""
A trainer class to handle training and testing of models.
"""

import logging
import torch

from model import Tagger
from vocab import MultiVocab
import data

logger = logging.getLogger('stanza')


def get_optimizer(name, parameters, lr, betas=(0.9, 0.999), eps=1e-8, momentum=0):
    if name == 'sgd':
        return torch.optim.SGD(parameters, lr=lr, momentum=momentum)
    elif name == 'adagrad':
        return torch.optim.Adagrad(parameters, lr=lr)
    elif name == 'adam':
        return torch.optim.Adam(parameters, lr=lr, betas=betas, eps=eps)
    elif name == 'adamax':
        return torch.optim.Adamax(parameters) # use default lr
    else:
        raise Exception("Unsupported optimizer: {}".format(name))

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


# base class, not sure if this is useful at all
# class BaseTrainer:
#     def change_lr(self, new_lr):
#         for param_group in self.optimizer.param_groups:
#             param_group['lr'] = new_lr

#     def save(self, filename):
#         savedict = {
#                    'model': self.model.state_dict(),
#                    'optimizer': self.optimizer.state_dict()
#                    }
#         torch.save(savedict, filename)

#     def load(self, filename):
#         savedict = torch.load(filename, lambda storage, loc: storage)

#         self.model.load_state_dict(savedict['model'])
#         if self.args['mode'] == 'train':
#             self.optimizer.load_state_dict(savedict['optimizer'])


# class Trainer(BaseTrainer):
class Trainer(object):
    """ A trainer for training models. """
    def __init__(self, model_file=None, vocab=None, pretrain=None, args=None, use_cuda=False):
        if model_file is not None:
            # load trained model from file
            checkpoint = torch.load(model_file, lambda storage, loc: storage)
            self.args = checkpoint['config']
            self.vocab = MultiVocab.load_state_dict(checkpoint['vocab'])
            self.model = Tagger(self.args, self.vocab, emb_matrix=pretrain.emb if pretrain else None, share_hid=self.args['share_hid'])
            self.model.load_state_dict(checkpoint['model'], strict=False)
        else:
            # build model from scratch
            self.args = args
            self.vocab = vocab
            self.model = Tagger(args, vocab, emb_matrix=pretrain.emb if pretrain else None, share_hid=args['share_hid'])
        
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.model.cuda()
        else:
            self.model.cpu()
        self.optimizer = get_optimizer(self.args['optim'], self.parameters, self.args['lr'], betas=(0.9, self.args['beta2']), eps=1e-6)

    def update(self, batch, eval=False):
        inputs, orig_idx, word_orig_idx, sentlens, wordlens = unpack_batch(batch, self.use_cuda)
        word, word_mask, wordchars, wordchars_mask, upos, ufeats, pretrained = inputs

        if eval:
            self.model.eval()
        else:
            self.model.train()
            self.optimizer.zero_grad()
        loss, _ = self.model(word, word_mask, wordchars, wordchars_mask, upos, ufeats, pretrained, word_orig_idx, sentlens, wordlens)
        loss_val = loss.data.item()
        if eval:
            return loss_val

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args['max_grad_norm'])
        self.optimizer.step()
        return loss_val

    def predict(self, batch, unsort=True):
        inputs, orig_idx, word_orig_idx, sentlens, wordlens = unpack_batch(batch, self.use_cuda)
        word, word_mask, wordchars, wordchars_mask, upos, ufeats, pretrained = inputs

        self.model.eval()
        batch_size = word.size(0)
        _, preds = self.model(word, word_mask, wordchars, wordchars_mask, upos, ufeats, pretrained, word_orig_idx, sentlens, wordlens)
        upos_seqs = [self.vocab['upos'].unmap(sent) for sent in preds[0].tolist()]
        feats_seqs = [self.vocab['feats'].unmap(sent) for sent in preds[1].tolist()]

        pred_tokens = [[[upos_seqs[i][j], feats_seqs[i][j]] for j in range(sentlens[i])] for i in range(batch_size)]
        if unsort:
            pred_tokens = data.unsort(pred_tokens, orig_idx)
        return pred_tokens

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
                'config': self.args
                }
        try:
            torch.save(params, filename, _use_new_zipfile_serialization=False)
            logger.info("Model saved to {}".format(filename))
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            logger.warning(f"Saving failed... {e} continuing anyway.")
