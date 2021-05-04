# was stanza.models.pos.scorer

"""
Utils and wrappers for scoring taggers.
"""
import logging

from utils import ud_scores

logger = logging.getLogger('stanza')

def score(system_conllu_file, gold_conllu_file, verbose=True):
    """ Wrapper for tagger scorer. """
    evaluation = ud_scores(gold_conllu_file, system_conllu_file)
    el = evaluation['AllUTags']
    p = el.precision
    r = el.recall
    f = el.f1
    if verbose:
        scores = [evaluation[k].f1 * 100 for k in ['UPOS', 'UFeats', 'AllUTags']]
        logger.info("UPOS\tUFeats\tAllUTags")
        logger.info("{:.2f}\t{:.2f}\t{:.2f}".format(*scores))
    return p, r, f

