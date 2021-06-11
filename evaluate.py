import argparse, evaluator, pyconll


POS = evaluator.POS_KEY


def load_predictions(args):
	# Regular CoNLL-U format
	if args.pred_upos_index == 3 and args.pred_xpos_index == 4 and args.pred_feats_index == 5:
		return pyconll.load_from_file(args.prediction)
	
	# other format
	else:
		s = ""
		with open(args.prediction, 'r') as pred_file:
			for line in pred_file:
				if line.strip() == "":
					s += line
				elif line.startswith("#"):
					s += line
				else:
					elements = line.split("\t")
					if args.pred_upos_index >= 0 and args.pred_upos_index < len(elements):
						upos = elements[args.pred_upos_index].strip()
					else:
						upos = "_"
					if args.pred_xpos_index >= 0 and args.pred_xpos_index < len(elements):
						xpos = elements[args.pred_xpos_index].strip()
					else:
						xpos = "_"
					if args.pred_feats_index >= 0 and args.pred_feats_index < len(elements):
						feats = elements[args.pred_feats_index].strip()
					else:
						feats = "_"
					s += "0\t_\t_\t{}\t{}\t{}\t0\t_\t_\t_\n".format(upos, xpos, feats)
		return pyconll.load_from_string(s)


def extract_inconsistencies(training_file):
	coocs = {}
	allkeys = set()
	data = pyconll.load_from_file(training_file)
	for sentence in data:
		for token in sentence:
			if token.upos not in coocs:
				coocs[token.upos] = set()
			for m in token.feats:
				if m in evaluator.UNIV_FEATURES:
					coocs[token.upos].add(m)
					allkeys.add(m)
	inconsistencies = {}
	for postag in coocs:
		inconsistencies[postag] = allkeys - coocs[postag]
	# print("Inconsistencies found:")
	# for postag in inconsistencies:
	# 	print("{}: {}".format(postag, ", ".join(inconsistencies[postag])))
	return inconsistencies


def evaluate_file(args, inconsistencies):
	print("Prediction file:  ", args.prediction)
	print("Gold file:        ", args.gold)
	pred_file = load_predictions(args)
	gold_file = pyconll.load_from_file(args.gold)

	if len(pred_file) != len(gold_file):
		print("Number of sentences does not match!")
		print("Prediction: {}    Gold: {}".format(len(pred_file), len(gold_file)))
		return
	
	upos_evaluator = evaluator.Evaluator(mode="exact")
	xpos_evaluator = evaluator.Evaluator(mode="exact")
	feats_evaluator = evaluator.Evaluator(mode="by_feats")
	ufeats_evaluator = evaluator.Evaluator(mode="exact", only_univ=True)
	upos_feats_evaluator = evaluator.Evaluator(mode="by_feats")
	incons_count = 0
	token_count = 0
	
	for pred_sent, gold_sent in zip(pred_file, gold_file):
		if len(pred_sent) != len(gold_sent):
			print("Number of words in sentence does not match!")
			print("Prediction: {}    Gold: {}".format(len(pred_sent), len(gold_sent)))
			print("Prediction:", pred_sent._meta)
			print("Gold:", gold_sent._meta)
			continue
		
		for pred_token, gold_token in zip(pred_sent, gold_sent):
			if args.upos:
				upos_evaluator.add_instance({POS: gold_token.upos}, {POS: pred_token.upos})
			if args.xpos:
				xpos_evaluator.add_instance({POS: gold_token.xpos}, {POS: pred_token.xpos})
			if args.feats:
				gold_feats = {x: ",".join(gold_token.feats[x]) for x in gold_token.feats}
				pred_feats = {x: ",".join(pred_token.feats[x]) for x in pred_token.feats}
				feats_evaluator.add_instance(gold_feats, pred_feats)
				ufeats_evaluator.add_instance(gold_feats, pred_feats)
				if args.upos:
					if args.incons:
						token_count += 1
						if len(set(pred_feats.keys()) & inconsistencies[pred_token.upos]) > 0:
							incons_count += 1
					gold_feats.update({POS: gold_token.upos})
					pred_feats.update({POS: pred_token.upos})
					upos_feats_evaluator.add_instance(gold_feats, pred_feats)
	
	if upos_evaluator.instance_count > 0:
		print("UPOS accuracy          {:.2f}%".format(100*upos_evaluator.acc()))
	if xpos_evaluator.instance_count > 0:
		print("XPOS accuracy          {:.2f}%".format(100*xpos_evaluator.acc()))
	if feats_evaluator.instance_count > 0:
		print("FEATS micro-F1         {:.2f}%".format(100*feats_evaluator.micro_f1()))
	if upos_feats_evaluator.instance_count > 0:
		print("UPOS+FEATS micro-F1    {:.2f}%".format(100*upos_feats_evaluator.micro_f1()))
	if ufeats_evaluator.instance_count > 0:
		print("UFEATS accuracy        {:.2f}%".format(100*ufeats_evaluator.acc()))
	if token_count > 0:
		print("UFEATS inconsistencies {:.2f}%".format(100*incons_count / token_count))
	print()


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Stand-alone evaluation script.')
	parser.add_argument("prediction", type=str, default="", help="File with the predicted labels")
	parser.add_argument("gold", type=str, default="", help="File with the gold labels")
	parser.add_argument("--upos", action="store_true", help="Whether to evaluate the UPOS column")
	parser.add_argument("--xpos", action="store_true", help="Whether to evaluate the XPOS column")
	parser.add_argument("--feats", action="store_true", help="Whether to evaluate the FEATS column")
	parser.add_argument("--incons", type=str, default="", help="Whether to evaluate inconsistent features and from which file they are extracted (typically training data")
	parser.add_argument("--pred-upos-index", type=int, default=3, help="Zero-based column index for the UPOS label in the prediction file")
	parser.add_argument("--pred-xpos-index", type=int, default=4, help="Zero-based column index for the XPOS label in the prediction file")
	parser.add_argument("--pred-feats-index", type=int, default=5, help="Zero-based column index for the FEATS labels in the prediction file")
	args = parser.parse_args()

	if args.incons != "":
		inconsistencies = extract_inconsistencies(args.incons)
	else:
		inconsistencies = {}
	
	evaluate_file(args, inconsistencies)
