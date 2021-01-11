import numpy as np
import pandas as pd
import datetime, sys, string
from random import randint
from sklearn.linear_model import Ridge
from sklearn.naive_bayes import MultinomialNB
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.initializers import Constant
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Embedding, SpatialDropout1D, Dropout, Input
from tensorflow.keras.optimizers import Adam

import tokenization
from utils import *



# Read data
train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')

# Change target value to some mislabelled tweets
ids_with_target_error = [328,443,513,2619,3640,3900,4342,5781,6552,6554,6570,6701,6702,6729,6861,7226]
train.loc[train['id'].isin(ids_with_target_error),'target'] = 0

# Expand the training set by adding tweets from: https://www.kaggle.com/kazanova/sentiment140
extra_train = pd.read_csv('input/expand_train_dataset.csv') #They are all label 0
train = train.append(extra_train, sort=False).reset_index()
y_train = train.target.values
print('Train shape: ', train.shape)

# Create arrays
tweets = train['text']
tweets_test = test['text']

del(train)
del(test)

# Preprocessing
for i, line in enumerate(tweets):
	tweets[i] = process_tweet(line)

for i, line in enumerate(tweets_test):
    tweets_test[i] = process_tweet(line)

train = pd.DataFrame({'text': tweets, 'target': y_train})
test = pd.DataFrame({'text': tweets_test})



######################
## BERT model - Feature 1
######################
'''
Uses raw text since BERT has its own preprocessing
Trained for 9 epochs with batch_size of 32 and Adam lr=1e-6
Due to computing limitations, I couldn't use BERT large or increase batch size
'''


module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2"
bert_layer = hub.KerasLayer(module_url, trainable=True)
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

bert_max_len = 32
bert_train_input = bert_encode(tweets, tokenizer, max_len=bert_max_len)
bert_test_input = bert_encode(tweets_test, tokenizer, max_len=bert_max_len)


input_word_ids = Input(shape=(bert_max_len,), dtype=tf.int32, name="input_word_ids")
input_mask = Input(shape=(bert_max_len,), dtype=tf.int32, name="input_mask")
segment_ids = Input(shape=(bert_max_len,), dtype=tf.int32, name="segment_ids")

_, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
cls_feat = sequence_output[:, 0, :]
emb_feat = GlobalAveragePooling1D()(sequence_output)
x = Concatenate()([cls_feat, emb_feat])
x = Dropout(0.3)(x)
out = Dense(1, activation='sigmoid')(x)

bert_model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
bert_model.summary()
path_bert = 'logs/bert_model.h5'
bert_model.load_weights(path_bert)

bert_feat = bert_model.predict(bert_train_input).flatten()
bert_out = bert_model.predict(bert_test_input).flatten()



######################
## LSTM Model + GloVe embeddings - Feature 2
######################

'''
Trained for 4 epochs with adam 5e-3, GloVe embeddings trainable
Embeddings obtained from https://www.kaggle.com/danielwillgeorge/glove6b100dtxt
'''

# Text to sequences
tokenizer_glove = Tokenizer(split=' ', oov_token='<UNK>')
tokenizer_glove.fit_on_texts(train.text.values)
glove_x = tokenizer_glove.texts_to_sequences(train.text.values)
word_index = tokenizer_glove.word_index
print('number of unique words: ', len(word_index))
glove_x = sequence.pad_sequences(glove_x)
glove_x_test = tokenizer_glove.texts_to_sequences(test.text.values)
glove_x_test = sequence.pad_sequences(glove_x_test, maxlen=np.shape(glove_x)[1])
glove_y = train.target.values

# Glove embeddings
embedding_dict = {}

with open('input/glove.6B.100d.txt', encoding="utf8") as glove:
    for line in glove:
        values = line.split()
        word = values[0]
        vectors = np.asarray(values[1:], 'float32')
        embedding_dict[word] = vectors        
glove.close()

num_words = len(word_index) + 1
embedding_matrix = np.zeros((num_words,100))

for word, i in tqdm(word_index.items()):
    if i > num_words:
        continue
    embedding_vector = embedding_dict.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

glove_model = Sequential()
glove_model.add(Embedding(num_words, 100, input_length = np.shape(glove_x)[1], embeddings_initializer=Constant(embedding_matrix), trainable=True))
glove_model.add(SpatialDropout1D(0.5))
glove_model.add(LSTM(128, dropout=0.3, recurrent_dropout=0.2))
glove_model.add(Dense(2, activation='sigmoid'))
glove_model.summary()
path_lstm_glove = 'logs/lstm_glove.h5'
glove_model.load_weights(path_lstm_glove)

glove_feat = glove_model.predict(glove_x)[:,1]
glove_out = glove_model.predict(glove_x_test)[:,1]



######################
## NB + Tfidf features
######################

# Create tf-idf features
tf = TfidfVectorizer(max_features=2500, stop_words=stop).fit(train.text.values)
x_train_tf = tf.transform(notBERT_train.text.values)
x_test_tf = tf.transform(notBERT_test.text.values)

NB = MultinomialNB(alpha= 0.62).fit(x_train_tf, y_train)
NB_tfidf_feat = NB.predict_proba(x_train_tf)[:,1]
NB_tfidf_out = NB.predict_proba(x_test_tf)[:,1]




######################
## ENSEMBLE
######################

feat_train = pd.DataFrame({'bert':bert_feat, 'lstm_glove':glove_feat, 'nb':NB_tfidf_feat})
feat_test = pd.DataFrame({'bert':bert_out, 'lstm_glove':glove_out, 'nb':NB_tfidf_out})


ensemble_model = Ridge(alpha=10)
predictions = ensemble_model.fit(feat_train, y_train).predict(feat_test).round().astype(int)
print(predictions)

name = 'final_ensemble'
submission = pd.read_csv('input/sample_submission.csv')
submission['target'] = predictions
submission.to_csv('output/submit_'+name+'.csv', index=False)