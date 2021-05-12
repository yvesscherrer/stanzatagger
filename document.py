# Reimplementation of stanza.models.common.doc and stanza.utils.conll that allows for more flexibility with not-quite-CoNLL-conform datasets

import logging
import io
from evaluator import Evaluator, UNIV_FEATURES

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

        evaluator = Evaluator()
        for sent in self.sentences:
            for token in sent:
                pred_feats = {}
                gold_feats = {}
                if "pos" in token.pred and "pos" in self.read_positions:
                    pred_feats.update({"POS": token.pred["pos"]})
                    gold_feats.update({"POS": token.given[self.read_positions["pos"]]})
                
                if "feats" in token.pred and "feats" in self.read_positions:
                    if token.pred["feats"] not in ("_", ""):
                        pf = dict([x.split("=", 1) for x in token.pred["feats"].split("|")])
                        pred_feats.update(pf)
                    else:
                        pf = {}
                    if token.given[self.read_positions["feats"]] not in ("_", ""):
                        gf = dict([x.split("=", 1) for x in token.given[self.read_positions["feats"]].split("|")])
                        gold_feats.update(gf)
                    else:
                        gf = {}
                evaluator.add_instance(gold_feats, pred_feats)

        results = {}
        results["POS acc"] = evaluator.acc(att="POS")
        results["FEATS micro-F1"] = evaluator.mic_f1(excl=["POS"])
        results["UFEATS micro-F1"] = evaluator.mic_f1(incl=UNIV_FEATURES)
        results["POS+FEATS micro-F1"] = evaluator.mic_f1()
        results["POS+UFEATS micro-F1"] = evaluator.mic_f1(incl=["POS"]+UNIV_FEATURES)
        return results


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


class Token(object):
    def __init__(self, token):
        self.given = token
        self.pred = {}
