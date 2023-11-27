from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from tensorflow_addons.utils.types import FloatTensorLike, TensorLike
from tensorflow_addons.layers import CRF
from tensorflow_addons.losses import SigmoidFocalCrossEntropy
from tensorflow_addons.optimizers import AdamW
import tensorflow as tf
import json
import numpy as np
from tensorflow.keras.models import load_model

def build_model(max_len = 103, input_dim = 51204,embedding_dim = 300):
  # Model definition
  input = Input(shape=(max_len,))

  # Get embeddings
  embeddings = Embedding(
      input_dim,
      embedding_dim,
    input_length=max_len,
       mask_zero=True,
    trainable=True,
      name = 'embedding_layer'
  )(input)

  # variational biLSTM
  output_sequences = Bidirectional(LSTM(units=50, return_sequences=True))(embeddings)
  output_sequences = Bidirectional(LSTM(units=100, return_sequences=True))(output_sequences)
  # Stacking
  output_sequences = Bidirectional(LSTM(units=50, return_sequences=True))(output_sequences)

  # Adding more non-linearity
  dense_out = TimeDistributed(Dense(25, activation="relu"))(output_sequences)

  # CRF layer
  mask = Input(shape=(max_len,), dtype=tf.bool)
  crf = CRF(20, name='crf')
  predicted_sequence, potentials, sequence_length, crf_kernel = crf(dense_out)

  model = Model(input, potentials)
  model.compile(
      optimizer=AdamW(weight_decay=0.001),
      loss= SigmoidFocalCrossEntropy()) # Sigmoid focal cross entropy loss

  return model

def load_model_and_config(model_name, model_path, epoch):
    # Load model configuration
    with open(f'{model_path}/{model_name}_config.json', 'r') as f:
        model_config = json.load(f)

    # Build the model based on the configuration
    loaded_model = build_model(max_len=103, input_dim=len(model_config['word2idx']))

    # Load model weights
    loaded_model.load_weights(f'{model_path}/{model_name}_epoch_{epoch}.h5')

    # Load dictionaries
    word2idx = model_config['word2idx']
    tag2idx = model_config['tag2idx']
    idx2word = model_config['idx2word']
    idx2tag = model_config['idx2tag']

    return loaded_model, word2idx, tag2idx, idx2word, idx2tag


def predict_tags(example_sentence):
    # Example usage:
    loaded_model, loaded_word2idx, loaded_tag2idx, loaded_idx2word, loaded_idx2tag = load_model_and_config('ner_crf', './model', 5)

    

    # Convert the example sentence to numerical input using loaded_word2idx
    numerical_input = [loaded_word2idx[word] if word in loaded_word2idx else 'nan' for word in example_sentence.split()]

    # Pad the input sequence to match the model's input shape
    max_len = 103  # Replace with your actual max_len value
    padded_input = np.pad(numerical_input, (0, max_len - len(numerical_input)), 'constant')

    # Make predictions
    predictions = loaded_model.predict(np.array([padded_input]))

    # Convert predictions to tags using idx2tag
    predicted_tags = [loaded_idx2tag[f'{np.argmax(pred)}'] for pred in predictions[0]]
    count = sum(1 for i in numerical_input if i != 'nan')
    return predicted_tags[0:count]

example_sentence = "Your example sentence here"
print(predict_tags(example_sentence))
