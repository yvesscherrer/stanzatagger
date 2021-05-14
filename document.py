# Reimplementation of stanza.models.common.doc and stanza.utils.conll that allows for more flexibility with not-quite-CoNLL-conform datasets

import logging
import io
import random
from evaluator import Evaluator, POS_KEY

logger = logging.getLogger('stanza')

class Document(object):
    def __init__(self, read_positions={"id": 0, "cform": 1, "wform": 1, "pos": 3, "feats": 5}, write_positions={"id": 0, "cform": 1, "wform": 1, "pos": 3, "feats": 5}):
        self.read_positions = {x: read_positions[x] for x in read_positions if read_positions[x] >= 0}
        self.write_positions = {x: write_positions[x] for x in write_positions if write_positions[x] >= 0}
        self.ignore_comments = ("id" in read_positions and read_positions["id"] == 0)
        self.sentences = []
    
    def __len__(self):
        return len(self.sentences)
    
    def __iter__(self):
        return iter(self.sentences)
    
    def _load(self, f):
        sent = Sentence()
        for line in f:
            if len(line.strip()) == 0:
                if len(sent) > 0:
                    self.sentences.append(sent)
                    sent = Sentence()
            else:
                if self.ignore_comments and line.startswith('#'):
                    continue
                array = line.split('\t')
                array[-1] = array[-1].strip()
                sent.add_token(array)
        if len(sent) > 0:
             self.sentences.append(sent)
    
    def load_from_file(self, filename):
        self._load(open(filename))
        logger.info("{} sentences loaded from file {}".format(len(self), filename))

    def load_from_string(self, s):
        self._load(io.StringIO(s))
        logger.info("{} sentences loaded".format(len(self)))
    
    def _write(self, f, pred=True, copy_untouched=True):
        for sent in self.sentences:
            for token in sent:
                if copy_untouched:
                    array = ["_" for x in range(max(max(self.write_positions.values())+1, len(token.given)))]
                else:
                     array = ["_" for x in range(max(self.write_positions.values())+1)]
                for key, pos in self.write_positions.items():
                    if pred and key in token.pred:
                        array[pos] = token.pred[key]
                    elif key in self.read_positions:
                        array[pos] = token.given[self.read_positions[key]]
                if copy_untouched:
                    for pos in range(len(array)):
                        if pos not in self.write_positions.values():
                            array[pos] = token.given[pos]

                f.write("\t".join(array) + "\n")
            f.write("\n")
    
    def write_to_file(self, filename, pred=True, copy_untouched=True):
        self._write(open(filename, 'w'), pred=pred, copy_untouched=copy_untouched)
    
    def write_to_string(self, pred=True, copy_untouched=True):
        s = io.StringIO()
        self._write(s, pred=pred, copy_untouched=copy_untouched)
        return s
    
    def provide_data(self):
        doc_array = []
        for sent in self.sentences:
            sent_array = []
            for token in sent:
                if "id" in self.read_positions and ("." in token.given[self.read_positions["id"]] or "-" in token.given[self.read_positions["id"]]):
                    continue
                token_array = []
                for key in ("cform", "wform", "pos", "feats"):
                    if key in self.read_positions:
                        token_array.append(token.given[self.read_positions[key]])
                    else:
                        token_array.append("")
                sent_array.append(token_array)
            doc_array.append(sent_array)
        return doc_array
    
    def add_predictions(self, doc_array):
        assert(len(doc_array) == len(self.sentences))
        for sent_array, sent in zip(doc_array, self.sentences):
            ti = 0
            for token in sent:
                if "id" in self.read_positions and ("." in token.given[self.read_positions["id"]] or "-" in token.given[self.read_positions["id"]]):
                    continue
                token.pred = {"pos": sent_array[ti][0], "feats": sent_array[ti][1]}
                ti += 1
            assert(ti == len(sent_array))
    
    def evaluate(self):
        if "pos" not in self.read_positions and "feats" not in self.read_positions:
            logger.info("Cannot evaluate predictions because gold annotations are not available.")
            return {}

        feats_evaluator = Evaluator(mode="by_feats")
        exact_evaluator = Evaluator(mode="exact", only_univ=True)
        for sent in self.sentences:
            for token in sent:
                pred_feats = {}
                gold_feats = {}                
                if "feats" in token.pred and "feats" in self.read_positions:
                    if token.pred["feats"] not in ("_", ""):
                        pred_feats = dict([x.split("=", 1) for x in token.pred["feats"].split("|")])
                    if token.given[self.read_positions["feats"]] not in ("_", ""):
                        gold_feats = dict([x.split("=", 1) for x in token.given[self.read_positions["feats"]].split("|")])
                    exact_evaluator.add_instance(gold_feats, pred_feats)    # do not add POS here
                
                if "pos" in token.pred and "pos" in self.read_positions:
                    pred_feats.update({POS_KEY: token.pred["pos"]})
                    gold_feats.update({POS_KEY: token.given[self.read_positions["pos"]]})
                    feats_evaluator.add_instance(gold_feats, pred_feats)

        results = {}
        results["POS accuracy"] = feats_evaluator.acc(att=POS_KEY)
        results["FEATS micro-F1"] = feats_evaluator.micro_f1(excl=[POS_KEY])
        results["POS+FEATS micro-F1"] = feats_evaluator.micro_f1()
        results["UFEATS exact match"] = exact_evaluator.acc()
        return results


    def get_augment_ratio(self, should_augment_predicate, can_augment_predicate, desired_ratio=0.1, max_ratio=0.5):
        """
        Returns X so that if you randomly select X * N sentences, you get 10%

        The ratio will be chosen in the assumption that the final dataset
        is of size N rather than N + X * N.

        should_augment_predicate: returns True if the sentence has some
        feature which we may want to change occasionally.  for example,
        depparse sentences which end in punct
        can_augment_predicate: in the depparse sentences example, it is
        technically possible for the punct at the end to be the parent
        of some other word in the sentence.  in that case, the sentence
        should not be chosen.  should be at least as restrictive as
        should_augment_predicate
        """
        n_data = len(self.sentences)
        n_should_augment = sum(should_augment_predicate(sentence) for sentence in self.sentences)
        n_can_augment = sum(can_augment_predicate(sentence) for sentence in self.sentences)
        n_error = sum(can_augment_predicate(sentence) and not should_augment_predicate(sentence)
                    for sentence in self.sentences)
        if n_error > 0:
            raise AssertionError("can_augment_predicate allowed sentences not allowed by should_augment_predicate")

        if n_can_augment == 0:
            logger.warning("Found no sentences which matched can_augment_predicate {}".format(can_augment_predicate))
            return 0.0
        n_needed = n_data * desired_ratio - (n_data - n_should_augment)
        # if we want 10%, for example, and more than 10% already matches, we can skip
        if n_needed < 0:
            return 0.0
        ratio = n_needed / n_can_augment
        if ratio > max_ratio:
            return max_ratio
        return ratio

    def augment_punct(self, augment_ratio=None):

        """
        Adds extra training data to compensate for some models having all sentences end with PUNCT

        Some of the models (for example, UD_Hebrew-HTB) have the flaw that
        all of the training sentences end with PUNCT.  The model therefore
        learns to finish every sentence with punctuation, even if it is
        given a sentence with non-punct at the end.

        One simple way to fix this is to train on some fraction of training data with punct.

        Params:
        train_data: list of list of dicts, eg a conll doc
        augment_ratio: the fraction to augment.  if None, a best guess is made to get to 10%

        TODO: do this dynamically, as part of the DataLoader or elsewhere?
        One complication is the data comes back from the DataLoader as
        tensors & indices, so it is much more complicated to manipulate
        """
        if len(self.sentences) == 0:
            return []
        
        can_augment_nopunct = lambda x: x.tokens[-1].given[self.read_positions['pos']] == "PUNCT"
        should_augment_nopunct = lambda x: x.tokens[-1].given[self.read_positions['pos']] == "PUNCT"

        if augment_ratio is None:
            augment_ratio = self.get_augment_ratio(should_augment_nopunct, can_augment_nopunct)
            logger.info("Determined augmentation ratio {:.2f}".format(augment_ratio))

        if augment_ratio <= 0:
            logger.info("Skipping data augmentation")
            return

        new_data = []
        for sentence in self.sentences:
            if can_augment_nopunct(sentence):
                if random.random() < augment_ratio and len(sentence) > 1:
                    # todo: could deep copy the words
                    #       or not deep copy any of this
                    new_sentence = sentence.copy(remove_last=True)
                    new_data.append(new_sentence)
        
        self.sentences.extend(new_data)
        logger.info("{} sentences available after augmentation".format(len(self.sentences)))


class Sentence(object):
    def __init__(self):
        self.tokens = []
    
    def __len__(self):
        return len(self.tokens)

    def __iter__(self):
        return iter(self.tokens)

    def add_token(self, token):
        t = Token(token)
        self.tokens.append(t)
    
    def copy(self, remove_last=False):
        new = Sentence()
        if remove_last:
            for t in self.tokens[:-1]:
                new.add_token(t)
        else:
            for t in self.tokens:
                new.add_token(t)
        return new


class Token(object):
    def __init__(self, token):
        self.given = token
        self.pred = {}
