# stanzatagger

This is a stand-alone version of the POS-tagger of the [stanza](https://stanfordnlp.github.io/stanza/) toolkit.

It is designed as a replacement of my earlier [lstmtagger](https://github.com/yvesscherrer/lstmtagger).

## Differences compared to lstmtagger

- Based on PyTorch, supports both CPU and GPU
- Supports early stopping, and a wider range of optimizers (Adam as default)
- Allows any combination of character-LSTM-based embeddings, trainable word embeddings and pretrained (frozen) word embeddings
- Uses biaffine prediction layers
- Avoids inconsistent predictions by feeding the predicted POS tag to the morphological feature predictions
- Supports data augmentation with non-punctuation-ending sentences

## Differences compared to stanza

- Leaner codebase due to focus on POS-tagging
- Does not rely on the folder hierarchies and naming conventions of Universal Dependencies
- Does not support two levels of POS tags in the same model (i.e., either UPOS or XPOS, but not both)
- Uses bidirectional character-LSTM by default
- More data reading and writing options to support datasets that do not exactly conform to CoNLL-U
- Command-line options compatible with lstmtagger
- More fine-grained evaluation

## Usage

### Training

```
python tagger.py \
        --training-data fr_sequoia-ud-train.conllu \
        --dev-data fr_sequoia-ud-dev.conllu \
        --dev-data-out exp_fr/dev_out.conllu \
        --scores-out exp_fr/scores.tsv \
        --model-save exp_fr/model.pt \
        --batch-size 500 \
        --max-steps 50000 \
        --augment-nopunct
```

### Prediction

```
python tagger.py \
        --model exp_fr/model.pt \
        --test-data fr_sequoia-ud-test.conllu \
        --test-data-out exp_fr/test_out.conllu \
        --batch-size 500
```

### Combined command

```
python tagger.py \
        --training-data fr_sequoia-ud-train.conllu \
        --dev-data fr_sequoia-ud-dev.conllu \
        --dev-data-out exp_fr/dev_out.conllu \
        --test-data fr_sequoia-ud-test.conllu \
        --test-data-out exp_fr/test_out.conllu \
        --scores-out exp_fr/scores.tsv \
        --model-save exp_fr/model.pt \
        --batch-size 500 \
        --max-steps 50000 \
        --augment-nopunct
```
