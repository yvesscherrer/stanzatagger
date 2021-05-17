"""
Entry point for training and evaluating a POS/morphological features tagger.

This tagger uses highway BiLSTM layers with character and word-level representations, and biaffine classifiers
to produce consistent POS and UFeats predictions.
For details please refer to paper: https://nlp.stanford.edu/pubs/qi2018universal.pdf.
"""

# was stanza.models.tagger

import os
import time
import argparse
import logging
import random
import numpy as np
import torch
from torch import optim

from data import DataLoader, unsort
from trainer import Trainer
from pretrain import Pretrain
from document import Document

logger = logging.getLogger('stanza')
logger.setLevel(logging.DEBUG)
log_handler = logging.StreamHandler()
log_formatter = logging.Formatter(fmt="%(asctime)s %(levelname)s: %(message)s",
                              datefmt='%Y-%m-%d %H:%M:%S')
log_handler.setFormatter(log_formatter)
logger.addHandler(log_handler)


def set_random_seed(seed, cuda):
    """
    Set a random seed on all of the things which might need it.
    torch, np, python random, and torch.cuda
    """
    if seed is None:
        seed = random.randint(0, 1000000000)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
    return seed

def ensure_dir(path):
    if path is None:
        return
    d = os.path.split(path)[0]
    if not os.path.exists(d):
        logger.info("Directory {} does not exist, creating it...".format(d))
        os.makedirs(d)

# def get_adaptive_eval_interval(cur_dev_size, thres_dev_size, base_interval):
#     """ Adjust the evaluation interval adaptively.
#     If cur_dev_size <= thres_dev_size, return base_interval;
#     else, linearly increase the interval (round to integer times of base interval).
#     """
#     if cur_dev_size <= thres_dev_size:
#         return base_interval
#     else:
#         alpha = round(cur_dev_size / thres_dev_size)
#         return base_interval * alpha  # nb_dev_sents / 20

def get_adaptive_eval_interval(nb_train_batches, nb_dev_batches, min_interval=100, max_interval=None, multiplier=2):
    if not max_interval:
        # evaluate at least once per epoch
        max_interval = nb_train_batches
    if max_interval <= min_interval:
        return min_interval
    opt_interval = round(multiplier * nb_dev_batches / 100) * 100
    if opt_interval < min_interval:
        return min_interval
    elif opt_interval > max_interval:
        return max_interval
    else:
        return opt_interval

def get_adaptive_log_interval(batch_size, min_interval=10, max_interval=1000, gpu=False):
    available_intervals = [x for x in (10, 20, 50, 100, 200, 500, 1000)
        if x >= min_interval and x <= max_interval]
    # log 100 times less often if using gpu
    gpu_multiplier = 100 if gpu else 1
    opt_interval = 1000 * gpu_multiplier / max(batch_size, 50)
    if opt_interval < min_interval:
        return min_interval
    elif opt_interval > max_interval:
        return max_interval
    else:
        step_interval = [x for x in available_intervals if x <= opt_interval][-1]
        return step_interval

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-data", dest="training_data", type=str, default=None, help="Input training data file")
    parser.add_argument("--vector-data", dest="vector_data", type=str, default=None, help="File from which to read the pretrained embeddings (supported file types: .txt, .vec, .xz, .gz)")
    parser.add_argument("--dev-data", dest="dev_data", type=str, default=None, help="Input development/validation data file")
    parser.add_argument("--dev-data-out", dest="dev_data_out", type=str, default=None, help="Output file for annotated development/validation data")
    parser.add_argument("--test-data", dest="test_data", type=str, default=None, help="Input test data file")
    parser.add_argument("--test-data-out", dest="test_data_out", type=str, default=None, help="Output file for annotated test data")

    parser.add_argument("--model", dest="model", default=None, help="Binary file (.pt) containing the parameters of a trained model")
    parser.add_argument("--model-save", dest="model_save", default=None, help="Binary file (.pt) in which the parameters of a trained model are saved")
    parser.add_argument("--vectors", dest="vectors", default=None, help="Binary file (.pt) containing the parameters of the pretrained embeddings")
    parser.add_argument("--vectors-save", dest="vectors_save", default=None, help="Binary file (.pt) in which the parameters of the pretrained embeddings are saved")

    # parser.add_argument('--data_dir', type=str, default='data/pos', help='Root dir for saving models.')
    # parser.add_argument('--wordvec_dir', type=str, default='extern_data/wordvec', help='Directory of word vectors.')
    # parser.add_argument('--wordvec_file', type=str, default=None, help='Word vectors filename.')
    # parser.add_argument('--wordvec_pretrain_file', type=str, default=None, help='Exact name of the pretrain file to read')
    # parser.add_argument('--train_file', type=str, default=None, help='Input file for data loader.')
    # parser.add_argument('--eval_file', type=str, default=None, help='Input file for data loader.')
    # parser.add_argument('--output_file', type=str, default=None, help='Output CoNLL-U file.')
    # parser.add_argument('--mode', default='train', choices=['train', 'predict'])

    parser.add_argument('--hidden_dim', type=int, default=200)
    parser.add_argument('--char_hidden_dim', type=int, default=400)
    parser.add_argument('--deep_biaff_hidden_dim', type=int, default=400)
    parser.add_argument('--composite_deep_biaff_hidden_dim', type=int, default=100)
    parser.add_argument('--word_emb_dim', type=int, default=75)
    parser.add_argument('--char_emb_dim', type=int, default=100)
    parser.add_argument('--tag_emb_dim', type=int, default=50)
    parser.add_argument('--transformed_dim', type=int, default=125)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--char_num_layers', type=int, default=1)
    parser.add_argument('--pretrain_max_vocab', type=int, default=250000)
    parser.add_argument('--word_dropout', type=float, default=0.33)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--rec_dropout', type=float, default=0, help="Recurrent dropout")
    parser.add_argument('--char_rec_dropout', type=float, default=0, help="Recurrent dropout")
    parser.add_argument('--no_char', dest='char', action='store_false', help="Turn off character model.")
    # parser.add_argument('--no_pretrain', dest='pretrain', action='store_false', help="Turn off pretrained embeddings.")
    parser.add_argument('--share_hid', action='store_true', help="Share hidden representations for UPOS, XPOS and UFeats.")
    parser.set_defaults(share_hid=False)

    parser.add_argument('--sample_train', type=float, default=1.0, help='Subsample training data.')
    parser.add_argument('--optim', type=str, default='adam', help='sgd, adagrad, adam or adamax.')
    parser.add_argument('--lr', type=float, default=3e-3, help='Learning rate')
    parser.add_argument('--beta2', type=float, default=0.95)

    parser.add_argument('--max_steps', type=int, default=50000)
    parser.add_argument('--eval_interval', type=int, default=None)
    parser.add_argument('--max_steps_before_stop', type=int, default=3000, help='Changes learning method or early terminates after this many steps if the dev scores are not improving')
    parser.add_argument('--batch_size', type=int, default=5000, help='Batch size in tokens. With a negative value, each batch consists of exactly one sentence.')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Gradient clipping.')
    parser.add_argument('--log_interval', type=int, default=None, help='Print log every k steps.')
    # parser.add_argument('--save_dir', type=str, default='saved_models/pos', help='Root dir for saving models.')
    # parser.add_argument('--save_name', type=str, default=None, help="File name to save the model")

    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--cpu', action='store_true', help='Force CPU even if GPU is available')

    parser.add_argument('--augment_nopunct', nargs='?', type=float, const=None, help='Augment the training data by copying this fraction of punct-ending sentences as non-punct.  Default of None will aim for roughly 10%')

    args = parser.parse_args(args=args)
    return args


def main(args=None):
    args = parse_args(args=args)
    ensure_dir(args.model_save)
    ensure_dir(args.vectors_save)
    ensure_dir(args.dev_data_out)
    ensure_dir(args.test_data_out)
    use_cuda = (not args.cpu) and torch.cuda.is_available()
    set_random_seed(args.seed, use_cuda)

    trainer = None
    if args.training_data:
        logger.info("Running tagger in training mode")
        trainer = train(args, use_cuda)
    
    if args.test_data:
        logger.info("Running tagger in prediction mode")
        evaluate(args, trainer, use_cuda)


def train(args, use_cuda=False):
    # TODO: load existing model if given
    
    # load pretrained vectors if needed
    if args.vector_data:
        pretrained = Pretrain(from_text=args.vector_data, max_vocab=args.pretrain_max_vocab)
        pretrained.save_to_pt(args.vectors_save)
    else:
        pretrained = None

    # load data
    logger.info("Loading training data with batch size {}...".format(args.batch_size))
    train_doc = Document(from_file=args.training_data)
    if args.augment_nopunct:
        train_doc.augment_punct(args.augment_nopunct)
    train_data = DataLoader(train_doc, args.batch_size, vars(args), pretrained, evaluation=False)
    vocab = train_data.vocab
    logger.info("Loading development data")
    dev_doc = Document(from_file=args.dev_data)
    dev_data = DataLoader(dev_doc, args.batch_size, vars(args), pretrained, vocab=vocab, evaluation=True, sort_during_eval=True)
    ensure_dir(args.dev_data_out)

    if len(train_data) == 0:
        logger.warning("Skip training because no training data is available.")
        return None

    logger.info("Training tagger...")
    logger.info("Device: {}".format("gpu" if use_cuda else "cpu"))
    trainer = Trainer(vocab=vocab, pretrain=pretrained, args=vars(args), use_cuda=use_cuda)

    global_step = 0
    epoch = 0
    dev_score_history = []
    last_best_step = 0
    max_steps = args.max_steps
    current_lr = args.lr
    global_start_time = time.time()
    format_str = 'Finished step {}/{}, loss = {:.6f} ({:.3f} sec/batch), lr: {:.6f}'

    if not args.eval_interval:
        #args.eval_interval = get_adaptive_eval_interval(dev_data.num_examples, 2000, 100)
        args.eval_interval = get_adaptive_eval_interval(len(train_data), len(dev_data))
    logger.info("Evaluating the model every {} steps...".format(args.eval_interval))
    
    if not args.log_interval:
        args.log_interval = get_adaptive_log_interval(args.batch_size, max_interval=args.eval_interval, gpu=use_cuda)
    logger.info("Showing log every {} steps".format(args.log_interval))

    # start training
    using_amsgrad = False
    train_loss = 0
    while True:
        epoch += 1
        epoch_start_time = time.time()
        do_break = False
        for batch in train_data:
            start_time = time.time()
            global_step += 1
            loss = trainer.update(batch, eval=False) # update step
            train_loss += loss
            if global_step % args.log_interval == 0:
                duration = time.time() - start_time
                logger.info(format_str.format(global_step, max_steps, loss, duration, current_lr))

            if len(dev_data) > 0 and global_step % args.eval_interval == 0:
                # eval on dev
                logger.info("Evaluating on dev set...")
                dev_preds = []
                for dev_batch in dev_data:
                    preds = trainer.predict(dev_batch)
                    dev_preds += preds
                dev_preds = unsort(dev_preds, dev_data.data_orig_idx)
                dev_data.doc.add_predictions(dev_preds)
                dev_data.doc.write_to_file(args.dev_data_out)
                results = dev_data.doc.evaluate()
                #dev_score = results["POS+FEATS micro-F1"]  # more intuitive
                dev_score = results["UFEATS exact match"]   # for backwards compatibility
                for k, v in results.items():
                    logger.info("{}: {:.2f}%".format(k, 100*v))

                train_loss = train_loss / args.eval_interval # avg loss per batch
                logger.info("step {}: train_loss = {:.6f}, dev_score = {:.4f}".format(global_step, train_loss, dev_score))
                train_loss = 0

                # save best model
                if len(dev_score_history) == 0 or dev_score > max(dev_score_history):
                    last_best_step = global_step
                    trainer.save(args.model_save)
                    logger.info("new best model saved.")

                dev_score_history += [dev_score]

            if global_step - last_best_step >= args.max_steps_before_stop:
                if not using_amsgrad:
                    logger.info("Switching to AMSGrad")
                    last_best_step = global_step
                    using_amsgrad = True
                    trainer.optimizer = optim.Adam(trainer.model.parameters(), amsgrad=True, lr=args.lr, betas=(.9, args.beta2), eps=1e-6)
                else:
                    logger.info("Early termination: have not improved in {} steps".format(args.max_steps_before_stop))
                    do_break = True
                    break

            if global_step >= args.max_steps:
                do_break = True
                break

        if do_break: break

        epoch_duration = time.time() - epoch_start_time
        logger.info("Finished epoch {} after step {} ({:.3f} sec/epoch)".format(epoch, global_step, epoch_duration))
        train_data.reshuffle()

    logger.info("Training ended with {} steps in epoch {}.".format(global_step, epoch))

    if len(dev_score_history) > 0:
        best_score, best_step = max(dev_score_history), np.argmax(dev_score_history)+1
        logger.info("Best dev score = {:.2f} at step {}".format(best_score*100, best_step * args.eval_interval))
    else:
        logger.info("Dev set never evaluated. Saving final model.")
        trainer.save(args.model_save)
    return trainer


def evaluate(args, trainer=None, use_cuda=False):
    # load pretrained vectors and model
    if not trainer:
        # load pretrained vectors
        pretrained = Pretrain(from_pt=args.vectors)
        # load model
        logger.info("Loading model from: {}".format(args.model))
        trainer = Trainer(model_file=args.model, pretrain=pretrained, use_cuda=use_cuda)
    
    # load data
    logger.info("Loading prediction data with batch size {}...".format(args.batch_size))
    doc = Document(from_file=args.test_data)
    data = DataLoader(doc, args.batch_size, trainer.args, pretrained, vocab=trainer.vocab, evaluation=True, sort_during_eval=True)
    if len(data) == 0:
        logger.warning("Skip prediction because no data is available.")
        return

    logger.info("Start prediction...")
    preds = []
    for batch in data:
        preds += trainer.predict(batch)
    preds = unsort(preds, data.data_orig_idx)

    # write to file and score
    data.doc.add_predictions(preds)
    data.doc.write_to_file(args.test_data_out)
    results = data.doc.evaluate()
    for k, v in results.items():
        logger.info("{}: {:.2f}%".format(k, 100*v))
    

if __name__ == '__main__':
    main()
