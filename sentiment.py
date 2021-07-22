#! -*- coding: utf-8 -*-
# 测试逻辑回归解析解的效果
# 博客：https://kexue.fm/archives/8578

import numpy as np
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.snippets import sequence_padding
from bert4keras.snippets import open
from bert4keras.optimizers import Adam
from keras.layers import *
from keras.models import Model
from tqdm import tqdm

maxlen = 128

# bert配置
config_path = '/root/kg/bert/chinese_roformer-sim-char-ft_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/root/kg/bert/chinese_roformer-sim-char-ft_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/root/kg/bert/chinese_roformer-sim-char-ft_L-12_H-768_A-12/vocab.txt'

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器

# 建立加载模型
encoder = build_transformer_model(
    config_path, checkpoint_path, model='roformer', with_pool='linear'
)


def load_data(filename):
    """加载数据
    单条格式：(文本, 标签id)
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            text, label = l.strip().split('\t')
            D.append((text, int(label)))
    return D


# 加载数据集
train_data = load_data('datasets/sentiment/sentiment.train.data')
valid_data = load_data('datasets/sentiment/sentiment.valid.data')
test_data = load_data('datasets/sentiment/sentiment.test.data')


def convert(data):
    """数据向量化
    """
    X, S, Y = [], [], []
    for t, l in tqdm(data):
        x, s = tokenizer.encode(t, maxlen=maxlen)
        X.append(x)
        S.append(s)
        Y.append([l])
    X = sequence_padding(X)
    S = sequence_padding(S)
    X = encoder.predict([X, S], verbose=True)
    Y = np.array(Y)
    return X, Y


train_x, train_y = convert(train_data)
valid_x, valid_y = convert(valid_data)
test_x, test_y = convert(test_data)


def compute_kernel_bias(vecs):
    """计算kernel和bias
    vecs.shape = [num_samples, embedding_size]，
    最后的变换：y = (x + bias).dot(kernel)
    """
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    W = u.dot(np.diag(1 / np.sqrt(s)))
    return W, -mu


# 数据白化
kernel, bias = compute_kernel_bias(train_x)
train_x = (train_x + bias).dot(kernel)
valid_x = (valid_x + bias).dot(kernel)
test_x = (test_x + bias).dot(kernel)

# ============== 通过梯度下降求解 ==============

x = Input(shape=(train_x.shape[1],))
y = Dense(1, activation='sigmoid')(x)

model = Model(x, y)
model.compile(
    loss='binary_crossentropy', optimizer=Adam(1e-3), metrics=['accuracy']
)
model.summary()

model.fit(
    train_x,
    train_y,
    epochs=10,
    batch_size=32,
    validation_data=(valid_x, valid_y)
)

train_y_pred = model.predict(train_x, verbose=True)
valid_y_pred = model.predict(valid_x, verbose=True)
test_y_pred = model.predict(test_x, verbose=True)
train_acc = np.mean((train_y == 1) == (train_y_pred > 0.5))
valid_acc = np.mean((valid_y == 1) == (valid_y_pred > 0.5))
test_acc = np.mean((test_y == 1) == (test_y_pred > 0.5))
print(train_acc, valid_acc, test_acc)

# ============== 通过解析解求解 ==============

p_1 = (train_y).mean()
p_0 = 1 - p_1
mu_0 = train_x[train_y[:, 0] == 0].mean(axis=0)
mu_1 = train_x[train_y[:, 0] == 1].mean(axis=0)
w = mu_1 - mu_0
b = (mu_0.dot(mu_0) - mu_1.dot(mu_1)) / 2 + np.log(p_1 / p_0)

train_y_pred = train_x.dot(w) + b
valid_y_pred = valid_x.dot(w) + b
test_y_pred = test_x.dot(w) + b
train_acc = np.mean((train_y[:, 0] == 1) == (train_y_pred > 0))
valid_acc = np.mean((valid_y[:, 0] == 1) == (valid_y_pred > 0))
test_acc = np.mean((test_y[:, 0] == 1) == (test_y_pred > 0))
print(train_acc, valid_acc, test_acc)
