#! -*- coding: utf-8 -*-
# 测试“全连接+Softmax”解析解的效果
# 博客：https://kexue.fm/archives/8578

import json
import numpy as np
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.snippets import sequence_padding
from bert4keras.snippets import open
from bert4keras.optimizers import Adam
from keras.layers import *
from keras.models import Model
from tqdm import tqdm

labels = [
    "100", "101", "102", "103", "104", "106", "107", "108", "109", "110", "112",
    "113", "114", "115", "116"
]
num_classes = len(labels)
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
    D = []
    with open(filename) as f:
        for i, l in enumerate(f):
            l = json.loads(l)
            text, label = l['sentence'], l['label']
            D.append((text, labels.index(label)))
    return D


# 加载数据集
train_data = load_data(
    '/root/CLUE-master/baselines/CLUEdataset/tnews/train.json'
)
valid_data = load_data('/root/CLUE-master/baselines/CLUEdataset/tnews/dev.json')


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

# ============== 通过梯度下降求解 ==============

x = Input(shape=(train_x.shape[1],))
y = Dense(num_classes, activation='softmax')(x)

model = Model(x, y)
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(1e-3),
    metrics=['accuracy']
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
train_acc = np.mean(train_y[:, 0] == train_y_pred.argmax(1))
valid_acc = np.mean(valid_y[:, 0] == valid_y_pred.argmax(1))
print(train_acc, valid_acc)

# ============== 通过解析解求解 ==============

ps = np.array([(train_y == i).mean() for i in range(num_classes)])
mus = np.array([
    train_x[train_y[:, 0] == i].mean(axis=0) for i in range(num_classes)
])
w = mus.T
b = np.log(ps) - (mus**2).sum(1) / 2

train_y_pred = train_x.dot(w) + b
valid_y_pred = valid_x.dot(w) + b
train_acc = np.mean(train_y[:, 0] == train_y_pred.argmax(1))
valid_acc = np.mean(valid_y[:, 0] == valid_y_pred.argmax(1))
print(train_acc, valid_acc)
