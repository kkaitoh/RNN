#!/usr/bin/env python
# coding: utf-8

# 分類問題を想定
import numpy as np
import pandas as pd
import tensorflow as tf
import optuna

from tensorflow.keras import Sequential, datasets
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout
from tensorflow.keras.preprocessing import sequence

from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score, confusion_matrix

from functools import partial

# データを読み込み（文字列長は 100 に調節済み）
X_train = np.load("X_train_hoge.npy")
X_test = np.load("X_test_hoge.npy")
y_train = np.array(pd.read_csv("y_train_hoge.csv", index_col = 0))
y_test = np.array(pd.read_csv("y_test_hoge.csv", index_col = 0))

# モデル構築用データとバリデーションデータに分割
X_model, X_validation, y_model, y_validation = train_test_split(X_train, y_train, test_size = 0.2, random_state = 0, stratify = y_train)

# モデルを作成
def model_building(X_train, y_train, X_test, y_test, learning_rate = 0.1, dropout = 0.2, recurrent_dropout = 0.2):
    embedding_vector_length = 32
    model = Sequential()
    model.add(Embedding(X_train.shape[0], embedding_vector_length, input_length = X_train.shape[1]))
    model.add(LSTM(100, dropout = dropout, recurrent_dropout = recurrent_dropout))
    model.add(Dense(1, activation = "sigmoid"))
    optimizer = tf.keras.optimizers.Adam(lr = learning_rate)
    model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])
    model.fit(X_train, y_train, epochs = 3, batch_size = 64)
    y_pred = model.predict_classes(X_test)
    return matthews_corrcoef(y_test, y_pred)

# optuna による hyperparametr の最適化
def optimized_function(X, y, X_val, y_val, trial):
    params = {
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.001, 0.9) ,
        "dropout": trial.suggest_loguniform("dropout", 0.01, 0.3),
        "recurrent_dropout": trial.suggest_loguniform("recurrent_dropout", 0.01, 0.3)
    }
    
    return - model_building(X_model, y_model, X_validation, y_validation, **params)

# セッションを作成
study = optuna.create_study(sampler = optuna.samplers.RandomSampler(seed = 0))

f = partial(optimized_function, X_model, y_model, X_validation, y_validation)
study.optimize(f, n_trials = 5)

# 最適なパラメーターを選択
best_params = study.best_params

# 全トレーニングデータを用いてモデルを再構築し、テストデータの予測を行う
model_building(X_train, y_train, X_test, y_test, **best_params)
scores = model.evaluate(X_test, y_test, verbose = 0)

model.summary()

print("Accuracy: %.2f%%" % (scores[1] * 100))

