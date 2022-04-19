# PA4 README

COSI-134A: StatNLP

Nov 24, 2021

Jayeol Chun

[jchun@brandeis.edu](mailto:jchun@brandeis.edu)

## Due Date
Dec. 15, 2021
* one week after the last day of class

## What You Need to Do
1. Implement `BahdanauAttentionDecoder`
2. Implement Dot-Product `LuongAttentionDecoder`
3. Implement General `LuongAttentionDecoder`
4. Run experiments with various combinations of hyperparameters to achieve best parsing performance
   1. Train a **Vanilla** Seq2Seq that does not use any attention mechanism. Tune the hyperparameters to get the best performance
   2. Train a Seq2Seq with **Bahdanau** attentional decoder. Tune the hyperparameters to get the best performance
   3. Train a Seq2Seq with **Luong Dot-Product** attentional decoder. Tune the hyperparameters to get the best performance
   4. Train a Seq2Seq with **Luong General** attentional decoder. Tune the hyperparameters to get the best performance
5. Write and submit a report discussing experimental results
6. Also submit [PA4 Notebook](./PA4.ipynb) (if using Google Colab) or [seq2seq.py](./seq2seq.py), which contains your attentional decoder implementations.
  * Note that this is the only file you need to modify in theory; in case you need to change any other files (which you are welcome to do) make sure to submit them as well


## Attentions
See Recitation Week 13

## Setup
```shell
pip install -r requirements.txt
```
Note that `requirements.txt` does not provide any specific versions.
The source code was tested with versions in `requirements_with_versions.txt`:
```
nltk==3.6.5
numpy==1.21.3
sacrebleu==2.0.0
torch==1.9.0
tqdm==4.62.3
```
You won't have to get the exact same versions, but you may encounter errors if the different versions have any breaking changes.


## Starter Code
We provide 2 either-or options for working on PA4:


## Option 1. using Google Colab
See [PA4 Notebook](./PA4.ipynb).
It will help nevertheless to look at .py files because we import and use them in the Notebook.



## Option 2. running .py files
This is primarily for those who have access to GPU either at home or on some remote server, and hence don't need to use Google Colab.


There are three scripts to be run sequentially.
* Note that the starter code trains a Vanilla Seq2Seq out-of-the-box

### 1. Prepare Data
[prepare_data.py](./prepare_data.py) loads raw PTB dataset, performs preprocessing at the token level and exports the outputs for future use.
```shell 
$ python prepare_data.py --help

usage: PA4 Data Preprocessing Argparser [-h] [--ptb_dir PTB_DIR] [--out_dir OUT_DIR] [--lower] [--reverse] [--prune] [--XX_norm] [--closing_tag] [--keep_index]

optional arguments:
  -h, --help         show this help message and exit
  --ptb_dir PTB_DIR  path to ptb directory
  --out_dir OUT_DIR  path to ptb outputs
  --lower            whether to lower all sentence strings
  --reverse          whether to reverse the source sentences
  --prune            whether to remove parenthesis for leaf POS tags
  --XX_norm          whether to normalize all POS tags to XX
  --closing_tag      whether to attach closing POS tags
  --keep_index       whether to keep trace indices
```

#### Source-side Preprocessing
* `--lower`: lower-case all source sentences
* `--reverse_sent`: reverse all source sentences
* `--keep_index`: drop tracing index
    * e.g. drop `-1` from `*T*-1` to get `*T*`

#### Target-side Preprocessing
Our chief aim here is to linearize a phrase structure tree, which is bracketed and has POS tags as its vocabularies.
* `--prune`: leaf POS tags introduce an additional set of parenthesis with itself being the only element within; whether to remove this parenthesis.
    * e.g. `(TAG1 (TAG2 ) )` -> `(TAG1 TAG2 )`
* `--XX_norm`: normalize all POS tags to XX
    * e.g. `(TAG1 (TAG2 ..` -> `(XX (XX ..`
* `--closing_tag`: append closing POS tag after the closing parenthesis
    * e.g. `(TAG1 (TAG2 ) )` -> `(TAG1 (TAG2 )TAG2 )TAG1`
* `--keep_index`: drop tracing index
    * e.g. drop `-1` from `*T*-1` to get `*T*`

#### Example Usage
```shell
# with default hyperparams
$ python prepare_data.py

# with custom hyperparams
$ python prepare_data.py --out_dir ./outputs/ptb_XX --XX_norm --reverse --prune
```

### 2. Training
[train.py](./train.py) performs training and also periodically evaluates model's performance on dev set.

```shell
$ python train.py --help

usage: PA4 Training Argparser [-h] [--data_dir DATA_DIR] [--model_dir MODEL_DIR] [--glove_dir GLOVE_DIR] [--glove_name {6B,42B,840B,twitter.27B}] [--glove_strategy {glove,overlap,sent}] [--with_torchtext] [--finetune_glove] [--sent_threshold SENT_THRESHOLD] [--tree_threshold TREE_THRESHOLD] [--model {vanilla,bahdanau,luong_dot,luong_general}] [--embed_dim EMBED_DIM] [--rnn {rnn,gru,lstm}] [--num_layers NUM_LAYERS] [--hidden_dim HIDDEN_DIM] [--dropout DROPOUT] [--epochs EPOCHS]
                              [--eval_every EVAL_EVERY] [--batch_size BATCH_SIZE] [--learning_rate LEARNING_RATE] [--teacher_forcing_ratio TEACHER_FORCING_RATIO] [--checkpoint CHECKPOINT] [--seed SEED]

optional arguments:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR   path to processed ptb input data
  --model_dir MODEL_DIR
                        path to model outputs
  --glove_dir GLOVE_DIR
                        path to glove dir; must be specified when using GloVe
  --glove_name {6B,42B,840B,twitter.27B}
                        name of the pre-trained GloVe vectors. If None, will attempt to infer automatically
  --glove_strategy {glove,overlap,sent}
                        how to handle vocabs when using GloVe. See `consts.py` for details
  --with_torchtext      whether to use torchtext when loading GloVe. Note that by default torchtext is not required
  --finetune_glove      whether to make GloVe embeddings trainable
  --sent_threshold SENT_THRESHOLD
                        minimum number of occurrences for sentence tokens to keep as vocab
  --tree_threshold TREE_THRESHOLD
                        minimum number of occurrences for tree tokens to keep as vocab
  --model {vanilla,bahdanau,luong_dot,luong_general}
                        which seq2seq model to train. See `consts.py` for details
  --embed_dim EMBED_DIM
                        embedding dimension
  --rnn {rnn,gru,lstm}  type of rnn to use in encoder and decoder
  --num_layers NUM_LAYERS
                        number of rnn layers in encoder and decoder
  --hidden_dim HIDDEN_DIM
                        rnn hidden dimension
  --dropout DROPOUT     dropout probability
  --epochs EPOCHS       number of training epochs
  --eval_every EVAL_EVERY
                        interval of epochs to perform evaluation on dev set
  --batch_size BATCH_SIZE
                        size of mini batch
  --learning_rate LEARNING_RATE
                        learning rate
  --teacher_forcing_ratio TEACHER_FORCING_RATIO
                        teacher forcing ratio, where higher means more teacher forcing; cannot be 1 with attentional decoders
  --checkpoint CHECKPOINT
                        if specified, attempts to load model params and resume training
  --seed SEED           seed value for replicability
```


#### Comments on Some Flags 
The default values were mostly chosen arbitrarily; be sure to try different hyperparameter values

* `--with_torchtext`: whether to use `torchtext` to load GloVe.
  If you used `torchtext` in PA2 and PA3, you may find this flag useful
* `--glove_strategy`: there are 3 options
  1. `glove`: discard Sentence vocabs and keep all of GloVe's vocabs
  2. `overlap`: only keep the vocabs found in both GloVe's vocabs and Sentence vocabs (default)
  3. `sent`: keep all of Sentence vocabs. For Sentence vocabs that also appear in Glove's vocabs, fetch their corresponding GloVe vectors; for the rest, sample
    from normal distribution parameterized by GloVe embedding's summary statistics
* `--finetune_glove`: if True, the GloVe embedding will be updated during training.
  While you are welcome to play around with this flag, it should remain False when `glove_strategy == glove` as it may raise CUDA OOM error 
* `--model`: there are 4 options
  1. `vanilla`: basic, vanilla Seq2Seq with no attention (default)
  2. `bahdanau`: Bahdanau (additive) attention decoder
  3. `luong_dot`: Luong dot-product attention decoder
  4. `luong_general`: Luong general attention decoder
* `--teacher_forcing_ratio`: in short, how often to use gold input rather than previous time-step's prediction at each decoding step.
  Setting this to 1 will make training significantly faster as PyTorch performs for-loop internally.
  However, this is only compatible with `model == vanilla` setting as other attentional decoders require manual iteration.
  In addition, model's performance on dev or test will suffer as the training environment becomes too different from that during inference


#### Example Usage
```shell
# with default hyperparams
$ python train.py

# with LSTM
$ python train.py --data_dir ./outputs/ptb_XX --model_dir ./outputs/model_LSTM --rnn LSTM

# with GloVe
$ python train.py --glove_dir ./glove_6B --embed_dim 200 --model_dir ./outputs/model_glove_6B --rnn LSTM

# resume training from checkpoint
$ python train.py --checkpoint ./outputs/model/epoch_100.pt --epochs 50
```

### 3. Inference
[inference.py](./inference.py) outputs model predictions on test set.

```shell
$ python inference.py --help

usage: PA4 Inference Argparser [-h] --checkpoint CHECKPOINT

optional arguments:
  -h, --help            show this help message and exit
  --checkpoint CHECKPOINT
                        checkpoint to load model from
```

Inference processes a single sentence at a time. 
While this is conventionally done with a while-loop which breaks when EOS is predicted, we instead set an arbitrary max decoding steps at `3 * source sequence length` and perform a for-loop.
Empirically, on average the linearized trees tend to have length about ~2.5 times that of source sequence, although this will vary by the hyperparameters used when preparing data in Step 1.


#### Example Usage
```shell
$ python inference.py --checkpoint ./outputs/model/epoch_100.pt
```


## Interpreting the Results
There are 2 metrics in use, **BLEU** and **Token-level Accuracy**, both of which are not exactly conventional in syntactic parsing.
Token-level accuracy can be especially deceiving as a model that outputs only closing brackets will score very high when linearized phrase structure tree contains no closing POS tags.
But they give us some sense of training progress.

The reason why we do not use the canonical Bracketed F1 score (EVALB) is that it requires well-formed trees as input, but a model has to be sufficiently trained to produce such high-quality trees. 
But this is difficult to achieve and often requires extensive heuristics to fix ill-formed trees.
Therefore we consider it beyond the scope of this program assignment.

While you are welcome to try to compile EVALB and report Bracketed F1 scores, this is optional.
**You are only required to report BLEU and Token-level Accuracy** in your report.



## Reference
* Seq2Seq for Parsing: [Grammar as a Foreign Language](https://arxiv.org/abs/1412.7449)
* Bahdanau Attention: [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
* Luong Attention: [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025v2)

