import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# Data Loading and Preprocessing
def load_data(filepath):
    articles = []
    titles = []
    with open(filepath, 'r') as f:
        for line in f:
            data = json.loads(line)
            articles.append(data['article'])
            titles.append(data['title'])
    return articles, titles

def preprocess_text(text):
  text = text.lower()
  text = ''.join(c for c in text if c.isalnum() or c.isspace()) # Keep only alphanumeric and spaces
  return text

articles, titles = load_data('dataset.jsonl')

# Preprocess the data
articles = [preprocess_text(article) for article in articles]
titles = [preprocess_text(title) for title in titles]

# Tokenization
tokenizer = keras.preprocessing.text.Tokenizer(num_words=5000, oov_token="<UNK>") # Limit vocabulary size
tokenizer.fit_on_texts(articles + titles) # Fit on combined text

article_sequences = tokenizer.texts_to_sequences(articles)
title_sequences = tokenizer.texts_to_sequences(titles)

# Padding
max_article_len = max(len(seq) for seq in article_sequences)
max_title_len = max(len(seq) for seq in title_sequences)

article_sequences = keras.preprocessing.sequence.pad_sequences(article_sequences, maxlen=max_article_len, padding='post')
title_sequences = keras.preprocessing.sequence.pad_sequences(title_sequences, maxlen=max_title_len, padding='post')


# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(article_sequences, title_sequences, test_size=0.2, random_state=42)

# Seq2Seq Model
embedding_dim = 256
latent_dim = 512

# Encoder
encoder_inputs = keras.Input(shape=(max_article_len,))
enc_emb = layers.Embedding(5000, embedding_dim)(encoder_inputs) # Use the same vocab size as tokenizer
encoder_lstm = layers.LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = keras.Input(shape=(None,)) # Dynamic input for decoder
dec_emb_layer = layers.Embedding(5000, embedding_dim)
dec_emb = dec_emb_layer(decoder_inputs)
decoder_lstm = layers.LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
decoder_dense = layers.Dense(5000, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compile and Train
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Prepare decoder input for training (shifted by one timestep)
decoder_input_data = np.concatenate([np.zeros((len(y_train), 1)), y_train[:, :-1]], axis=1).astype(int)

model.fit([X_train, decoder_input_data], np.expand_dims(y_train, -1),
          batch_size=64, epochs=10, validation_data=([X_val, np.concatenate([np.zeros((len(y_val), 1)), y_val[:, :-1]], axis=1).astype(int)], np.expand_dims(y_val, -1)))



# Inference Model (for generating titles)
encoder_model = keras.Model(encoder_inputs, encoder_states)

decoder_state_input_h = keras.Input(shape=(latent_dim,))
decoder_state_input_c = keras.Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
dec_emb2 = dec_emb_layer(decoder_inputs)
decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=decoder_states_inputs)
decoder_states2 = [state_h2, state_c2]
decoder_outputs2 = decoder_dense(decoder_outputs2)
decoder_model = keras.Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs2] + decoder_states2)


def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1))
    decoded_sentence = ''

    stop_condition = False
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = tokenizer.index_word.get(sampled_token_index)

        if(sampled_char!='<UNK>'):
          decoded_sentence += ' '+sampled_char

        if (sampled_char == '<end>' or len(decoded_sentence) > 50):
            stop_condition = True

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]

    return decoded_sentence

# Example usage:
test_article = "this is a test article about machine learning and deep learning"
test_article = preprocess_text(test_article)
test_seq = tokenizer.texts_to_sequences([test_article])
test_seq = keras.preprocessing.sequence.pad_sequences(test_seq, maxlen=max_article_len, padding='post')

decoded_title = decode_sequence(test_seq)
print(f"Article: {test_article}")
print(f"Generated Title: {decoded_title}")
