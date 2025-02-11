import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pandas as pd



sentences1 = []
sentences2 = []
labels = [] 

url = "https://raw.githubusercontent.com/MLDroid/quora_duplicate_challenge/master/data/quora_duplicate_questions.tsv"
df = pd.read_csv(url, sep='\t')

# Extract relevant columns
sentences1.extend(df['question1'].tolist())
sentences2.extend(df['question2'].tolist())
labels.extend(df['is_duplicate'].tolist())

# Tokenization and padding
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(sentences1 + sentences2)

sequences1 = tokenizer.texts_to_sequences(sentences1)
sequences2 = tokenizer.texts_to_sequences(sentences2)

max_len = 50  # Adjust based on dataset
X1 = pad_sequences(sequences1, maxlen=max_len)
X2 = pad_sequences(sequences2, maxlen=max_len)
y = np.array(labels)

X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(X1, X2, y, test_size=0.2)

embeddings_index = {}
with open('glove.6B.100d.txt') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 100
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in tokenizer.word_index.items():
    if i < vocab_size:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector



def siamese_lstm(max_len, vocab_size, embedding_dim, embedding_matrix):
    embedding_layer = Embedding(
        vocab_size,
        embedding_dim,
        weights=[embedding_matrix],
        input_length=max_len,
        trainable=False
    )
    
    lstm = LSTM(128, return_sequences=False)
    
    input1 = Input(shape=(max_len,))
    input2 = Input(shape=(max_len,))
    
    vec1 = embedding_layer(input1)
    vec1 = lstm(vec1)
    
    vec2 = embedding_layer(input2)
    vec2 = lstm(vec2)
    
    difference = Lambda(lambda x: tf.abs(x[0] - x[1]))([vec1, vec2])
    
    dense = Dense(64, activation='relu')(difference)
    output = Dense(1, activation='sigmoid')(dense)
    
    model = Model(inputs=[input1, input2], outputs=output)
    return model

model = siamese_lstm(max_len, vocab_size, embedding_dim, embedding_matrix)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



history = model.fit(
    [X1_train, X2_train],
    y_train,
    batch_size=32,
    epochs=10,
    validation_data=([X1_test, X2_test], y_test)
)



loss, accuracy = model.evaluate([X1_test, X2_test], y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")





def predict_similarity(sentence1, sentence2):
    seq1 = tokenizer.texts_to_sequences([sentence1])
    seq2 = tokenizer.texts_to_sequences([sentence2])
    padded1 = pad_sequences(seq1, maxlen=max_len)
    padded2 = pad_sequences(seq2, maxlen=max_len)
    pred = model.predict([padded1, padded2])
    return pred[0][0]

similarity_score = predict_similarity("I enjoy coding", "Programming is fun")
print(f"Similarity Score: {similarity_score:.4f}")
