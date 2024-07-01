# Training a Speech Recognizer

This is a simple speech recognizer trained on specific data.  In particular,  it implements an offline end-to-end attention-based speech recognizer.  A tokenizer is used to detect the word token to estimate. Search replies on beam search coupled with an RNN language model.

Training such a system requires the following steps:

1. Train a tokenizer.
Given the training transcriptions, the tokenizers decide which word pieces allocate for training. Most atomic units are character,  the least atomic units are words.  Most of the time, it is convenient to use tokens that are something in between characters and full words.
SpeechBrain relies on the popular [SentencePiece](https://github.com/google/sentencepiece) for tokenization. To train the tokenizer:

```
cd Tokenizer
python train.py tokenizer.yaml
```

2. Train a LM
After having our target tokens, we can train a language model on top of that. To do it, we need some large text corpus (better if the language domain is the same as the one of your target application). In this example, we simply train the LM on top of the training transcriptions:

```
cd LM
python train.py RNNLM.yaml
```

In a real case, training LM is extremely computational demanding. It is thus a good practice to reuse existing LM or fine-tune them.

3. Train the speech recognizer
At this point, we can train our speech recognizer. In this case, we are using a simple CRDNN model with an autoregressive GRU decoder. An attention mechanism is employed between encoding and decoder. The final sequence of words is retrieved with beamsearch coupled with the RNN LM trained in the previous step. To train the ASR:

```
cd ASR
python train.py train.yaml
```

4. Inference
Download model CKPT from [Drive](URL) and put in ASR folder

```
cd ASR/inference
python transcribe_wavs.py ../../data/test ../results/CRDNN_BPE_960h_LM/2602/save/CKPT+latest
```

## Why we use **SpeechBrain**?
