#! -*- coding:utf-8 -*-
import json
import numpy as np
import codecs
import keras
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
import tensorflow as tf
from tensorflow.train import Saver
import keras.backend.tensorflow_backend as ktf
from keras_bert import load_trained_model_from_checkpoint, Tokenizer, get_custom_objects, AdamWarmup, calc_train_steps

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)
ktf.set_session(sess)

maxlen = 25
config_path = '../bert_ernie/bert_config.json'
checkpoint_path = '../bert_ernie/bert_model.ckpt'
dict_path = '../bert_ernie/vocab.txt'

class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]') # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]') # 剩余的字符是[UNK]
        return R

def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X])

class data_generator:
    def __init__(self, data, tokenizer, batch_size=32):
        self.data = data
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1
    def __len__(self):
        return self.steps
    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            X1, X2, Y = [], [], []
            ################
            #### modify here
            for i in idxs:
                d = self.data[i]
                x1, x2 = self.tokenizer.encode(first=d[0], second=d[1], max_len=maxlen)
                y = d[2]
                X1.append(x1)
                X2.append(x2)
                Y.append(y)
            ###############
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    yield [X1, X2], Y
                    [X1, X2, Y] = [], [], []

def create_model(train=True):
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)
    if train:
        for l in bert_model.layers:
            l.trainable = True
    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))

    x = bert_model([x1_in, x2_in])
    # get [CLS] as feature
    x = Lambda(lambda x: x[:, 0])(x)
    ################
    #### modify here, add your network
    p = Dense(units=2, activation='softmax')(x)
    ################

    model = Model([x1_in, x2_in], p)
    return model

def train(tokenizer):
    model = create_model()
    ################
    #### modify here
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(1e-5),
        metrics=['accuracy']
    )
    model.summary()

    data = []
    with codecs.open("data.txt", "r", "utf-8") as f:
        dataset = f.readlines()
        for line in dataset:
            line = line.strip().split("\t")
            line = (line[0].replace(" ", ""), line[1].replace(" ", ""), int(line[2]))
            data.append(line)    

    random_order = list(range(len(data)))
    np.random.shuffle(random_order)
    train_data = [data[j] for i, j in enumerate(random_order) if i<=int(len(data)*0.7)]
    valid_data = [data[j] for i, j in enumerate(random_order) if i>int(len(data)*0.7)]
    
    train_D = data_generator(train_data, tokenizer)
    valid_D = data_generator(valid_data, tokenizer)
    
    epochs = 10
    #model.load('./ernie_{}.h5'.format(i), custom_objects=get_custom_objects())
    for i in range(epochs):
        model.fit_generator(
            train_D.__iter__(),
            steps_per_epoch=len(train_D),
            epochs=1,
            validation_data=valid_D.__iter__(),
            validation_steps=len(valid_D)
        )
        model.save('./ernie_{}.h5'.format(i))
    ################

token_dict = {}

with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)

tokenizer = OurTokenizer(token_dict)

train(tokenizer)
saver = Saver()
sess = keras.backend.get_session()
save_path = saver.save(sess, "./model.ckpt")