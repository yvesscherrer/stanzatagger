# was lstmtagger/evaluate_morphotags.py

import statistics, collections


POS_KEY = "POS"
UNIV_FEATURES = [
    "PronType", "NumType", "Poss", "Reflex", "Foreign", "Abbr", "Gender",
    "Animacy", "Number", "Case", "Definite", "Degree", "VerbForm", "Mood",
    "Tense", "Aspect", "Voice", "Evident", "Polarity", "Person", "Polite"
]


def f1(corr, gold, obs):
	if gold <= 0 or obs <= 0 or corr <= 0:
		return 0
	rec = corr / gold
	pre = corr / obs
	return (2 * rec * pre) / (rec + pre)


class Evaluator(object):
    '''
	Aggregates and evaluates attribute scores
    :param mode: one of 'by_feats', 'by_values', 'exact' - 'by_feats' pools scores by attribute over values, 'by_values' uses separate scores for each <attribute, value> pair, 'exact' pools scores by each distinct string of all concatenated attribute.value pairs
	:param only_univ: only uses the features evaluated in CoNLL18, i.e. those listed in UNIV_FEATURES
	'''

    def __init__(self, mode="by_feats", only_univ=False):
        self.instance_count = 0
        self.mode = mode
        self.only_univ = only_univ
        self.correct = collections.defaultdict(int)
        self.gold = collections.defaultdict(int)
        self.observed = collections.defaultdict(int)
    
    def keys(self):
        return self.gold.keys() | self.observed.keys()

    def add_instance(self, g, o):
        '''
        :param g: - gold annotation for instance (key-value dict)
        :param o: - observed (inferred) annotation for instance (key-value dict)
        '''
        self.instance_count = self.instance_count + 1
        if self.mode == "exact":
            if self.only_univ:
                gkey = "|".join(["=".join(x) for x in sorted(g.items()) if x[0] == POS_KEY or x[0] in UNIV_FEATURES])
                okey = "|".join(["=".join(x) for x in sorted(o.items()) if x[0] == POS_KEY or x[0] in UNIV_FEATURES])
            else:
                gkey = "|".join(["=".join(x) for x in sorted(g.items())])
                okey = "|".join(["=".join(x) for x in sorted(o.items())])
            self.gold[gkey] += 1
            self.observed[okey] += 1
            if gkey == okey:
                self.correct[gkey] += 1
        
        else:
            for (k, v) in g.items():
                if self.only_univ and k != POS_KEY and k not in UNIV_FEATURES:
                    continue
                key = (k, v) if self.mode == "by_values" else k
                if k in o and o[k] == v:
                    self.correct[key] += 1
                self.gold[key] += 1
            for (k, v) in o.items():
                if self.only_univ and k != POS_KEY and k not in UNIV_FEATURES:
                    continue
                key = (k, v) if self.mode == "by_values" else k
                self.observed[key] += 1

    def micro_f1(self, att=None, excl=[]):
        '''
        Micro F1
        :param att: get f1 for specific attribute (exact match)
        :param excl: get f1 for all attributes except those listed
        '''
        if att is not None:
            return f1(self.correct[att], self.gold[att], self.observed[att])
        else:
            keys = self.gold.keys() | self.observed.keys()
            if excl is not None:
                if self.mode == "by_values":
                    keys = [k for k in keys if k[0] not in excl]
                else:
                    keys = [k for k in keys if k not in excl]
            return f1(
                sum([self.correct[att] for att in self.correct if att in keys]),
                sum([self.gold[att] for att in self.gold if att in keys]),
                sum([self.observed[att] for att in self.observed if att in keys])
            )

    def macro_f1(self, excl=[]):
        '''
        Macro F1
        :param excl: get f1 for all attributes except those listed
        '''
        keys = self.gold.keys() | self.observed.keys()
        if excl is not None:
            if self.mode == "by_values":
                keys = [k for k in keys if k[0] not in excl]
            else:
                keys = [k for k in keys if k not in excl]
        return statistics.mean([f1(self.correct[k], self.gold[k], self.observed[k]) for k in keys])

    def acc(self, att=None):
        '''
        Accuracy
        '''
        if self.instance_count <= 0:
            return 0.0
        if att is not None:
            if self.mode == "by_values":
                corr = sum([self.correct[k] for k in self.correct if k[0] == att])
                gold = sum([self.gold[k] for k in self.gold if k[0] == att])
                return corr / gold
            elif self.gold[att] == 0:
                return 0.0
            else:
                return self.correct[att] / self.gold[att]
        else:
            corr = sum(self.correct.values())
            gold = sum(self.gold.values())
            return corr / gold

    def f1(self, corr, gold, obs):
        if gold <= 0 or obs <= 0 or corr <= 0:
            return 0
        r = corr / gold
        p = corr / obs
        return (2 * r * p) / (r + p)
