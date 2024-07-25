# Training a Speech Recognizer

This is a simple speech recognizer trained on specific data.  In particular,  it implements an offline end-to-end attention-based speech recognizer.  A tokenizer is used to detect the word token to estimate. Search replies on beam search coupled with an RNN language model.

## Inference:
1. Inference ASR
Download **All** model CKPT from [Drive](https://drive.google.com/drive/folders/19xoiiQH8pByVKRv3jblOnPc4nhhH6H1a?usp=sharing) and put in ASR folder

```
cd ASR/inference
python transcribe_wavs.py ../../data/test ../results/CRDNN_BPE_960h_LM/2602/save/CKPT+2024-07-01+14-24-57+00
```

2. Inference Diarization

```
cd ASR/diarization
python diarization.py
```
### Inference Example
[Kaggle](https://www.kaggle.com/code/ahm215/speechrecognition)

### Inference Example
/ASR/output.mp4

## Training such a system requires the following steps:

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


### Why This Model?

Choosing the simple CRDNN model with an autoregressive GRU decoder and an attention mechanism for speech recognition tasks offers several benefits, particularly considering model size and efficiency:

1. **Efficient Feature Extraction**: 
   - The convolutional layers in the CRDNN model efficiently capture spatial hierarchies in the input data, useful for processing the spectrogram or mel-frequency inputs typical in speech tasks.
   - This helps in reducing the dimensionality of the problem early in the network, making it computationally less expensive.

2. **Temporal Modeling**: 
   - The recurrent layers, particularly the GRU (Gated Recurrent Units), are excellent at capturing temporal dependencies and dynamics in the data.
   - GRUs are also known for being more parameter-efficient compared to LSTMs, beneficial for a smaller model size.

3. **Focused Processing**: 
   - The attention mechanism allows the model to focus on specific parts of the input sequence during the decoding process, improving the accuracy and relevance of the output.
   - This selective focus means that the model can be smaller yet still effective, as it does not need to maintain redundant information throughout its layers.

4. **Reduced Model Complexity**: 
   - Combining these components into a single architecture helps to streamline the model, reducing the overall complexity.
   - This integrated approach can lead to fewer parameters and lower memory usage compared to using separate models for each task.

5. **Balance Between Performance and Efficiency**: 
   - This model setup provides a good balance between computational efficiency and performance capability.
   - It allows for real-time processing capabilities which are often required in speech recognition applications, without significant trade-off in accuracy.

6. **Adaptability**: 
   - The model's architecture is adaptable to various types of speech data and can be fine-tuned for specific tasks without extensive reconfiguration.
   - It makes it a versatile choice that remains relatively lightweight.

This configuration makes the CRDNN model with an autoregressive GRU decoder and attention mechanism a strategic choice for achieving high performance in speech recognition while managing model size and computational demands efficiently.

## Why Use SpeechBrain?

SpeechBrain is chosen for the following reasons:

1. **Flexibility and Extensibility**: SpeechBrain is highly modular and flexible, allowing easy customization and extension of models to fit specific needs.

2. **Easy to Train from Scratch**: Provides various architectures that can be trained from scratch, useful for unique datasets or tasks where pre-trained models may not be effective.

3. **Support for Multiple Languages**: Its architecture is suitable for adapting to under-represented languages such as Arabic.

4. **End-to-End Training**: Supports end-to-end learning from raw audio to text, simplifying the training pipeline and potentially improving performance.

5. **PyTorch-based**: Benefits from all features of PyTorch, including easier debugging and integration with other PyTorch-based tools.

6. **Active Community and Support**: Backed by a strong community and continuous updates, ensuring access to the latest speech technologies.

7. **Versatility in Applications**: Capable of handling various speech-related tasks beyond ASR, making it a versatile toolkit.

8. **Open Source and Collaborative**: Promotes collaboration and rapid development as an open-source project.
