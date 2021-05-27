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

The following command trains a model on the French Sequoia treebank:

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

The following command uses the trained model to annotate the test set:

```
python tagger.py \
        --model exp_fr/model.pt \
        --test-data fr_sequoia-ud-test.conllu \
        --test-data-out exp_fr/test_out.conllu \
        --batch-size 500
```

The two commands above can be combined in a single command to avoid reloading the model:

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

## Full list of parameters

General parameters:

```
  -h, --help            show this help message and exit
  --seed SEED           Set the random seed
  --cpu                 Force CPU even if GPU is available
```

File paths:

```
  --training-data TRAINING_DATA [TRAINING_DATA ...]
                        Input training data file(s), a space-separated list of
                        several file names can be given
  --emb-data EMB_DATA   File from which to read the pretrained embeddings
                        (supported file types: .txt, .vec, .xz, .gz)
  --emb-max-vocab EMB_MAX_VOCAB
                        Limit the pretrained embeddings to the first N entries
                        (default: 250000)
  --dev-data DEV_DATA   Input development/validation data file
  --dev-data-out DEV_DATA_OUT
                        Output file for annotated development/validation data
  --test-data TEST_DATA
                        Input test data file
  --test-data-out TEST_DATA_OUT
                        Output file for annotated test data
  --scores-out SCORES_OUT
                        TSV file in which training scores and statistics are
                        saved (default: None)
  --model MODEL         Binary file (.pt) containing the parameters of a trained
                        model
  --model-save MODEL_SAVE
                        Binary file (.pt) in which the parameters of a trained
                        model are saved
  --embeddings EMBEDDINGS
                        Binary file (.pt) containing the parameters of the
                        pretrained embeddings
  --embeddings-save EMBEDDINGS_SAVE
                        Binary file (.pt) in which the parameters of the
                        pretrained embeddings are saved
```

Data formatting and evaluation:

```
  --number-index NUMBER_INDEX
                        Field in which the word numbers are stored (default: 0)
  --number-index-out NUMBER_INDEX_OUT
                        Field in which the word numbers are saved in the output
                        file (default: 0). Use negative value to skip word
                        numbers.
  --c-token-index C_TOKEN_INDEX
                        Field in which the tokens used for the character
                        embeddings are stored (default: 1). Use negative value
                        if character embeddings should be disabled.
  --c-token-index-out C_TOKEN_INDEX_OUT
                        Field in which the character embedding tokens are saved
                        in the output file (default: 1). Use negative value to
                        skip tokens.
  --w-token-index W_TOKEN_INDEX
                        Field in which the tokens used for the word embeddings
                        are stored (default: 1). Use negative value if word
                        embeddings should be disabled.
  --w-token-index-out W_TOKEN_INDEX_OUT
                        Field in which the tokens used for the word embeddings
                        are saved in the output file (default: -1). Use negative
                        value to skip tokens.
  --w-token-min-freq W_TOKEN_MIN_FREQ
                        Minimum frequency starting from which word embeddings
                        will be considered (default: 7)
  --pos-index POS_INDEX
                        Field in which the main POS is stored (default [UPOS
                        tags]: 3)
  --pos-index-out POS_INDEX_OUT
                        Field in which the main POS is saved in the output file
                        (default: 3)
  --morph-index MORPH_INDEX
                        Field in which the morphology features are stored
                        (default: 5). Use negative value if morphology features
                        should not be considered
  --morph-index-out MORPH_INDEX_OUT
                        Field in which the morphology features are saved in the
                        output file (default: 5). Use negative value to skip
                        features.
  --oov-index-out OOV_INDEX_OUT
                        Field in which OOV information is saved in the output
                        file (default: not written)
  --no-eval-feats NO_EVAL_FEATS [NO_EVAL_FEATS ...]
                        Space-separated list of morphological features that
                        should be ignored during evaluation. Typically used for
                        additional tasks in multitask settings.
  --mask-other-fields   Replaces fields in input that are not used by the tagger
                        (e.g. lemmas, dependencies) with '_' instead of copying
                        them.
  --augment-nopunct [AUGMENT_NOPUNCT]
                        Augment the training data by copying some amount of
                        punct-ending sentences as non-punct (default: 0.1,
                        corresponding to 10%)
  --punct-tag PUNCT_TAG
                        POS tag of sentence-final punctuation used for
                        augmentation (default: PUNCT)
  --sample-train SAMPLE_TRAIN
                        Subsample training data to proportion of N (default:
                        1.0)
  --cut-dev CUT_DEV     Cut dev data to first N sentences (default: keep all)
  --debug               Debug mode. This is a shortcut for '--sample-train 0.05
                        --cut-dev 100 --batch-size -1'
```

Network architecture:

```
  --word-emb-dim WORD_EMB_DIM
                        Size of word embedding layer (default: 75). Use negative
                        value to turn off word embeddings
  --char-emb-dim CHAR_EMB_DIM
                        Size of character embedding layer (default: 100). Use
                        negative value to turn off character embeddings
  --transformed-emb-dim TRANSFORMED_DIM
                        Size of transformed output layer of character embeddings
                        and pretrained embeddings (default: 125)
  --pos-emb-dim POS_EMB_DIM
                        Size of POS embeddings that are fed to predict the
                        morphology features (default: 50). Use negative value to
                        use shared, i.e. non-hierarchical representations for
                        POS and morphology
  --char-hidden-dim CHAR_HIDDEN_DIM
                        Size of character LSTM hidden layers (default: 400)
  --char-num-layers CHAR_NUM_LAYERS
                        Number of character LSTM layers (default: 1). Use 0 to
                        disable character LSTM
  --char-unidir         Uses a unidirectional LSTM for the character embeddings
                        (default: bidirectional)
  --tag-hidden-dim TAG_HIDDEN_DIM
                        Size of tagger LSTM hidden layers (default: 200)
  --tag-num-layers TAG_NUM_LAYERS
                        Number of tagger LSTM layers (default: 2)
  --deep-biaff-hidden-dim DEEP_BIAFF_HIDDEN_DIM
                        Size of biaffine hidden layers (default: 400)
  --composite-deep-biaff-hidden-dim COMPOSITE_DEEP_BIAFF_HIDDEN_DIM
                        Size of composite biaffine hidden layers (default: 100)
  --dropout DROPOUT     Input dropout (default: 0.5)
  --char-rec-dropout CHAR_REC_DROPOUT
                        Recurrent dropout for character LSTM (default: 0).
                        Should only be used with more than one layer
  --tag-rec-dropout TAG_REC_DROPOUT
                        Recurrent dropout for the tagger LSTM (default: 0).
                        Should only be used with more than one layer
  --word-dropout WORD_DROPOUT
                        Word dropout (default: 0.33)
```

Training and optimization:

```
  --batch-size BATCH_SIZE
                        Batch size in tokens (default: 5000). Use negative value
                        to use single sentences
  --max-steps MAX_STEPS
                        Maximum training steps (default: 50000)
  --max-steps-before-stop MAX_STEPS_BEFORE_STOP
                        Changes learning method or early terminates after N
                        steps if the dev scores are not improving (default:
                        3000). Use negative value to disable early stopping
  --log-interval LOG_INTERVAL
                        Print log every N steps. The default value is determined
                        on the basis of batch size and CPU/GPU mode.
  --eval-interval EVAL_INTERVAL
                        Evaluate on dev set every N steps. The default value is
                        determined on the basis of the training and dev data
                        sizes.
  --learning-rate LR    Learning rate (default: 3e-3)
  --optimizer {sgd,adagrad,adam,adamax}
                        Optimization algorithm (default: adam)
  --beta2 BETA2         Beta2 value required for adam optimizer (default: 0.95)
  --max-grad-norm MAX_GRAD_NORM
                        Gradient clipping (default: 1.0)
```
