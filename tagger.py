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
    if nb_dev_batches < 1:
        return max_interval
    if max_interval <= min_interval:
        return min_interval
    opt_interval = round(multiplier * nb_dev_batches / 100) * 100
    if opt_interval < min_interval:
        return min_interval
    elif opt_interval > max_interval:
        return max_interval
    else:
        return opt_interval

def get_adaptive_log_interval(batch_size, min_interval=10, max_interval=1000, avail_intervals=(10, 20, 50, 100, 200, 500, 1000), gpu=False):
    available_intervals = [x for x in avail_intervals
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

    # data loading and saving
    parser_paths = parser.add_argument_group('File paths')
    parser_paths.add_argument("--training-data", type=str, default=None, help="Input training data file")
    parser_paths.add_argument("--emb-data", type=str, default=None, help="File from which to read the pretrained embeddings (supported file types: .txt, .vec, .xz, .gz)")
    parser_paths.add_argument('--emb-max-vocab', type=int, default=250000, help="Limit the pretrained embeddings to the first N entries (default: 250000)")
    parser_paths.add_argument("--dev-data", type=str, default=None, help="Input development/validation data file")
    parser_paths.add_argument("--dev-data-out", type=str, default=None, help="Output file for annotated development/validation data")
    parser_paths.add_argument("--test-data", type=str, default=None, help="Input test data file")
    parser_paths.add_argument("--test-data-out", type=str, default=None, help="Output file for annotated test data")
    parser_paths.add_argument("--model", default=None, help="Binary file (.pt) containing the parameters of a trained model")
    parser_paths.add_argument("--model-save", default=None, help="Binary file (.pt) in which the parameters of a trained model are saved")
    parser_paths.add_argument("--embeddings", default=None, help="Binary file (.pt) containing the parameters of the pretrained embeddings")
    parser_paths.add_argument("--embeddings-save", default=None, help="Binary file (.pt) in which the parameters of the pretrained embeddings are saved")

    # data formatting (default options are fine for UD-formatted files)
    parser_data = parser.add_argument_group('Data formatting and evaluation')
    parser_data.add_argument("--number-index", type=int, help="Field in which the word numbers are stored (default: 0)", default=0)
    parser_data.add_argument("--number-index-out", type=int, help="Field in which the word numbers are saved in the output file (default: 0). Use negative value to skip word numbers.", default=0)
    parser_data.add_argument("--c-token-index", type=int, help="Field in which the tokens used for the character embeddings are stored (default: 1). Use negative value if character embeddings should be disabled.", default=1)
    parser_data.add_argument("--c-token-index-out", type=int, help="Field in which the character embedding tokens are saved in the output file (default: 1). Use negative value to skip tokens.", default=1)
    parser_data.add_argument("--w-token-index", type=int, help="Field in which the tokens used for the word embeddings are stored (default: 1). Use negative value if word embeddings should be disabled.", default=1)
    parser_data.add_argument("--w-token-index-out", type=int, help="Field in which the tokens used for the word embeddings are saved in the output file (default: -1). Use negative value to skip tokens.", default=-1)
    parser_data.add_argument("--pos-index", type=int, help="Field in which the main POS is stored (default [UPOS tags]: 3)", default=3)
    parser_data.add_argument("--pos-index-out", type=int, help="Field in which the main POS is saved in the output file (default: 3)", default=3)
    parser_data.add_argument("--morph-index", type=int, help="Field in which the morphology features are stored (default: 5). Use negative value if morphology features should not be considered", default=5)
    parser_data.add_argument("--morph-index-out", type=int, help="Field in which the morphology features are saved in the output file (default: 5). Use negative value to skip features.", default=5)
    parser_data.add_argument("--no-eval-feats", nargs='+', default=[], help="Space-separated list of morphological features that should be ignored during evaluation. Typically used for additional tasks in multitask settings.")
    parser_data.add_argument("--mask-other-fields", dest="copy_untouched", action="store_false", help="Replaces fields in input that are not used by the tagger (e.g. lemmas, dependencies) with '_' instead of copying them.", default=True)
    parser_data.add_argument('--augment-nopunct', nargs='?', type=float, const=None, help='Augment the training data by copying some amount of punct-ending sentences as non-punct (default: 0.1, corresponding to 10%).')
    parser_data.add_argument('--sample-train', type=float, default=1.0, help='Subsample training data (default: 1.0)')

    parser_net = parser.add_argument_group('Network architecture')
    parser_net.add_argument('--word-emb-dim', type=int, default=75, help="Size of word embedding layer (default: 75). Use negative value to turn off word embeddings")
    parser_net.add_argument('--char-emb-dim', type=int, default=100, help="Size of character embedding layer (default: 100). Use negative value to turn off character embeddings")
    parser_net.add_argument('--transformed-emb-dim', dest="transformed_dim", type=int, default=125, help="Size of transformed output layer of character embeddings and pretrained embeddings (default: 125)")
    parser_net.add_argument('--pos-emb-dim', type=int, default=50, help="Size of POS embeddings that are fed to predict the morphology features (default: 50). Use negative value to use shared, i.e. non-hierarchical representations for POS and morphology")
    parser_net.add_argument('--char-hidden-dim', type=int, default=400, help="Size of character LSTM hidden layers (default: 400)")
    parser_net.add_argument('--char-num-layers', type=int, default=1, help="Number of character LSTM layers (default: 1). Use 0 to disable character LSTM")
    parser_net.add_argument('--tag-hidden-dim', type=int, default=200, help="Size of tagger LSTM hidden layers (default: 200)")
    parser_net.add_argument('--tag-num-layers', type=int, default=2, help="Number of tagger LSTM layers (default: 2)")
    # TODO: merge two lines below, improve help
    parser_net.add_argument('--deep-biaff-hidden-dim', type=int, default=400, help="Size of biaffine hidden layers (default: 400)")
    parser_net.add_argument('--composite-deep-biaff-hidden-dim', type=int, default=100, help="Size of composite biaffine hidden layers (default: 100)")
    parser_net.add_argument('--dropout', type=float, default=0.5, help="Input dropout (default: 0.5)")
    parser_net.add_argument('--char-rec-dropout', type=float, default=0, help="Recurrent dropout for character LSTM (default: 0). Should only be used with more than one layer")
    parser_net.add_argument('--tag-rec-dropout', type=float, default=0, help="Recurrent dropout for the tagger LSTM (default: 0). Should only be used with more than one layer")
    parser_net.add_argument('--word-dropout', type=float, default=0.33, help="Word dropout (default: 0.33)")

    parser_train = parser.add_argument_group('Training and optimization')
    parser_train.add_argument('--batch-size', type=int, default=5000, help='Batch size in tokens (default: 5000). Use negative value to use single sentences')
    parser_train.add_argument('--max-steps', type=int, default=50000, help="Maximum training steps (default: 50000)")
    parser_train.add_argument('--max-steps-before-stop', type=int, default=3000, help='Changes learning method or early terminates after N steps if the dev scores are not improving (default: 3000). Use negative value to disable early stopping')
    parser_train.add_argument('--log-interval', type=int, default=None, help='Print log every N steps. The default value is determined on the basis of batch size and CPU/GPU mode.')
    parser_train.add_argument('--eval-interval', type=int, default=None, help="Evaluate on dev set every N steps. The default value is determined on the basis of the training and dev data sizes.")
    parser_train.add_argument('--learning-rate', dest="lr", type=float, default=3e-3, help='Learning rate (default: 3e-3)')
    parser_train.add_argument('--optimizer', dest="optim", type=str, choices=['sgd', 'adagrad', 'adam', 'adamax'], default='adam', help='Optimization algorithm (default: adam)')
    parser_train.add_argument('--beta2', type=float, default=0.95, help="Beta2 value required for adam optimizer (default: 0.95)")
    parser_train.add_argument('--max-grad-norm', type=float, default=1.0, help='Gradient clipping (default: 1.0)')

    # Other arguments
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--cpu', action='store_true', help='Force CPU even if GPU is available')

    # TODO:
    # parser.add_argument("--add-probs", dest="add_probs", action="store_true", help="Write prediction probabilities to output files")
    # parser.add_argument("--debug", dest="debug", action="store_true", help="Debug mode")
    # - default directory + default output names
    # - word embedding cutoff
    # - per-attribute evaluation
    # - OOV-specific evaluation

    args = parser.parse_args(args=args)
    return args

def validate_args(args):
    if args.c_token_index < 0 or args.char_num_layers < 1 or args.char_emb_dim < 1:
        logger.info("Disable character embeddings")
        args.c_token_index = -1
        args.c_token_index_out = -1
        args.char_num_layers = 0
        args.char_emb_dim = 0
    
    if args.w_token_index < 0:
        logger.info("Disable word embeddings")
        args.w_token_index = -1
        args.w_token_index_out = -1
        args.word_emb_dim = 0
        logger.info("Disable pretrained embeddings")
        args.emb_data = None
        args.embeddings = None
        args.embeddings_save = None
    elif args.emb_data is None and args.embeddings is None:
        if args.word_emb_dim < 1:
            logger.info("Disable word embeddings")
            args.w_token_index = -1
            args.w_token_index_out = -1
        logger.info("Disable pretrained embeddings")
        args.emb_data = None
        args.embeddings = None
        args.embeddings_save = None
    
    if args.c_token_index < 0 and args.w_token_index < 0:
        raise RuntimeError("Cannot use tagger without any token information.")
    if args.pos_index < 0:
        raise RuntimeError("POS tag field is required for using the tagger.")
    
    if args.model_save and args.emb_data and (args.embeddings_save is None):
        logger.warning("Pre-trained embeddings must be saved as a .pt file!")
        args.embeddings_save = args.model_save.replace(".pt", ".emb.pt")
        logger.warning("Saving them as {}".format(args.embeddings_save))
    
    ensure_dir(args.model_save)
    ensure_dir(args.embeddings_save)
    ensure_dir(args.dev_data_out)
    ensure_dir(args.test_data_out)
    

def get_read_format_args(args):
    return {"id": args.number_index,
        "cform": args.c_token_index, "wform": args.w_token_index,
        "pos": args.pos_index, "feats": args.morph_index}

def get_write_format_args(args):
    return {"id": args.number_index_out,
        "cform": args.c_token_index_out, "wform": args.w_token_index_out,
        "pos": args.pos_index_out, "feats": args.morph_index_out}


def main(args=None):
    args = parse_args(args=args)
    validate_args(args)
    use_cuda = (not args.cpu) and torch.cuda.is_available()
    logger.info("Selected device: {}".format("gpu" if use_cuda else "cpu"))
    seed = set_random_seed(args.seed, use_cuda)
    logger.info("Random seed: {}".format(seed))

    trainer, pretrained = None, None
    if args.training_data:
        logger.info("Running tagger in training mode...")
        trainer, pretrained = train(args, use_cuda)
    
    if args.test_data:
        logger.info("Running tagger in prediction mode...")
        evaluate(args, trainer, pretrained, use_cuda)


def train(args, use_cuda=False):
    # load pretrained embeddings if needed
    if args.emb_data:
        pretrained = Pretrain(from_text=args.emb_data, max_vocab=args.emb_max_vocab)
        pretrained.save_to_pt(args.embeddings_save)
    else:
        pretrained = None

    # load data
    logger.info("Loading training data...")
    train_doc = Document(from_file=args.training_data, read_positions=get_read_format_args(args))
    if args.augment_nopunct:
        train_doc.augment_punct(args.augment_nopunct)
    # load existing model if available
    if args.model:
        logger.info("Loading model from {}".format(args.model))
        trainer = Trainer(model_file=args.model, pretrain=pretrained, args=vars(args), use_cuda=use_cuda)
        train_data = DataLoader(train_doc, args.batch_size, vocab=trainer.vocab, pretrain=pretrained, sample_train=args.sample_train, evaluation=False)
    else:
        train_data = DataLoader(train_doc, args.batch_size, vocab=None, pretrain=pretrained, sample_train=args.sample_train, evaluation=False)
        trainer = Trainer(vocab=train_data.vocab, pretrain=pretrained, args=vars(args), use_cuda=use_cuda)

    if len(train_data) == 0:
        raise RuntimeError("Cannot start training because no training data is available")
       
    logger.info("Loading development data...")
    dev_doc = Document(from_file=args.dev_data, read_positions=get_read_format_args(args), write_positions=get_write_format_args(args), copy_untouched=args.copy_untouched)
    dev_data = DataLoader(dev_doc, args.batch_size, vocab=trainer.vocab, pretrain=pretrained, evaluation=True)

    if len(dev_data) > 0:
        if not args.eval_interval:
            #args.eval_interval = get_adaptive_eval_interval(dev_data.num_examples, 2000, 100)
            args.eval_interval = get_adaptive_eval_interval(len(train_data), len(dev_data))
        logger.info("Evaluating the model every {} steps".format(args.eval_interval))
    else:
        logger.info("No dev data given, not evaluating the model")
    
    if not args.log_interval:
        args.log_interval = get_adaptive_log_interval(args.batch_size, max_interval=args.eval_interval, gpu=use_cuda)
    logger.info("Showing log every {} steps".format(args.log_interval))

    global_step = 0
    epoch = 0
    dev_score_history = []
    last_best_step = 0
    max_steps = args.max_steps
    current_lr = args.lr
    global_start_time = time.time()
    format_str = 'Finished step {}/{}, loss = {:.6f}, {:.3f} sec/batch, lr: {:.6f}'

    # start training
    logger.info("Start training...")
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
                dev_doc.add_predictions(dev_preds)
                dev_doc.write_to_file(args.dev_data_out)
                results = dev_doc.evaluate(exclude=args.no_eval_feats)
                #dev_score = results["POS+FEATS micro-F1"]  # more intuitive
                dev_score = results["UFEATS exact match"]   # for backwards compatibility
                for k, v in results.items():
                    logger.info("{}: {:.2f}%".format(k, 100*v))

                train_loss = train_loss / args.eval_interval # avg loss per batch
                logger.info("Step {}/{}: train_loss = {:.6f}, dev_score = {:.4f}".format(global_step, max_steps, train_loss, dev_score))
                train_loss = 0

                # save best model
                if len(dev_score_history) == 0 or dev_score > max(dev_score_history):
                    logger.info("New best model found")
                    last_best_step = global_step
                    trainer.save(args.model_save)

                dev_score_history += [dev_score]

            if global_step - last_best_step >= args.max_steps_before_stop:
                if not using_amsgrad:
                    logger.info("Switching to AMSGrad")
                    last_best_step = global_step
                    using_amsgrad = True
                    trainer.optimizer = optim.Adam(trainer.model.parameters(), amsgrad=True, lr=args.lr, betas=(.9, args.beta2), eps=1e-6)
                else:
                    logger.info("Early stopping: dev_score has not improved in {} steps".format(args.max_steps_before_stop))
                    do_break = True
                    break

            if global_step >= args.max_steps:
                do_break = True
                break

        if do_break: break

        epoch_duration = time.time() - epoch_start_time
        logger.info("Finished epoch {} after step {}, {:.3f} sec/epoch".format(epoch, global_step, epoch_duration))
        train_data.reshuffle()

    logger.info("Training ended with {} steps in epoch {}".format(global_step, epoch))

    if len(dev_score_history) > 0:
        best_score, best_step = max(dev_score_history), np.argmax(dev_score_history)+1
        logger.info("Best dev score = {:.2f} at step {}".format(best_score*100, best_step * args.eval_interval))
    else:
        logger.info("Dev set never evaluated, saving final model")
        trainer.save(args.model_save)
    return trainer, pretrained


def evaluate(args, trainer=None, pretrained=None, use_cuda=False):
    # load pretrained embeddings and model
    if not trainer:
        # load pretrained embeddings
        pretrained = Pretrain(from_pt=args.embeddings)
        # load model
        logger.info("Loading model from {}".format(args.model))
        trainer = Trainer(model_file=args.model, pretrain=pretrained, use_cuda=use_cuda)
    
    # load data
    logger.info("Loading prediction data...")
    doc = Document(from_file=args.test_data, read_positions=get_read_format_args(args), write_positions=get_write_format_args(args), copy_untouched=args.copy_untouched)
    data = DataLoader(doc, args.batch_size, vocab=trainer.vocab, pretrain=pretrained, evaluation=True)
    if len(data) == 0:
        raise RuntimeError("Cannot start prediction because no data is available")

    logger.info("Start prediction...")
    preds = []
    for batch in data:
        preds += trainer.predict(batch)
    preds = unsort(preds, data.data_orig_idx)

    # write to file and score
    doc.add_predictions(preds)
    doc.write_to_file(args.test_data_out)
    results = doc.evaluate(exclude=args.no_eval_feats)
    for k, v in results.items():
        logger.info("{}: {:.2f}%".format(k, 100*v))
    

if __name__ == '__main__':
    main()
