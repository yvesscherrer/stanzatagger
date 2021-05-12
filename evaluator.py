# was lstmtagger/evaluate_morphotags.py

import statistics, collections


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
	by_values - if true, use separate scores for each <attribute, value> pair; if false, pool scores by attribute over values
	'''

    def __init__(self, by_values=False):
        self.instance_count = 0
        self.exact_match = 0
        self.correct = collections.defaultdict(int)
        self.gold = collections.defaultdict(int)
        self.observed = collections.defaultdict(int)
        self.by_values = by_values

    def add_instance(self, g, o):
        '''
        :param g: - gold annotation for instance (key-value dict)
        :param o: - observed (inferred) annotation for instance (key-value dict)
        '''
        self.instance_count = self.instance_count + 1
        if g == o:
            self.exact_match += 1
        for (k, v) in g.items():
            key = (k, v) if self.by_values else k
            if k in o and o[k] == v:
                self.correct[key] += 1
            self.gold[key] += 1
        for (k, v) in o.items():
            key = (k, v) if self.by_values else k
            self.observed[key] += 1

    def mic_f1(self, att=None, excl=[], incl=None):
        '''
        Micro F1
        :param att: get f1 for specific attribute (exact match)
        :param excl: get f1 for all attributes except those listed
        :param incl: get f1 only for attributes listed (None for all attributes)
        '''
        if att is not None:
            return f1(self.correct[att], self.gold[att], self.observed[att])
        else:
            keys = self.gold.keys() | self.observed.keys()
            if incl is not None:
                if self.by_values:
                    keys = [k for k in keys if k[0] in incl]
                else:
                    keys = [k for k in keys if k in incl]
            if excl is not None:
                if self.by_values:
                    keys = [k for k in keys if k[0] not in excl]
                else:
                    keys = [k for k in keys if k not in excl]
            return f1(
                sum([self.correct[att] for att in self.correct if att in keys]),
                sum([self.gold[att] for att in self.gold if att in keys]),
                sum([self.observed[att] for att in self.observed if att in keys])
            )

    def mac_f1(self, excl=[], incl=None):
        '''
        Macro F1
        :param excl: get f1 for all attributes except those listed
        :param incl: get f1 only for attributes listed (None for all attributes)
        '''
        keys = self.gold.keys() | self.observed.keys()
        if incl is not None:
            if self.by_values:
                keys = [k for k in keys if k[0] in incl]
            else:
                keys = [k for k in keys if k in incl]
        if excl is not None:
            if self.by_values:
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
            if self.by_values:
                corr = sum([self.correct[k] for k in self.correct if k[0] == att])
                gold = sum([self.gold[k] for k in self.gold if k[0] == att])
                return corr / gold
            else:
                return self.correct[att] / self.gold[att]
        else:
            return self.exact_match / self.instance_count

    def f1(self, corr, gold, obs):
        if gold <= 0 or obs <= 0 or corr <= 0:
            return 0
        r = corr / gold
        p = corr / obs
        return (2 * r * p) / (r + p)
