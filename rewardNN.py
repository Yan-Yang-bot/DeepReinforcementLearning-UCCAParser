import tensorflow as tf
import numpy as np
import json

data = json.load(open('envTrain.json','w'))

x = [d['obs'] + list(d['act'].values()) for d in data]
y = [d['r'] for d in data]

tf.placeholder(shape=(None, 38))
tf.layers.dense(64, activation="softmax")
