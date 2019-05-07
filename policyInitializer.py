
import json
import gzip
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

'''Load Data'''


def load_data(t):
    with gzip.GzipFile('env-{}.json'.format(t), 'r') as fin:
        json_bytes = fin.read()
    print('file read')
    json_str = json_bytes.decode('utf-8')
    print('string decoded')
    data = json.loads(json_str)
    print('json parsed')

    data.extend(json.loads(gzip.GzipFile('env-train-2.json').read().decode('utf-8')))
    print('second file')
    data.extend(json.loads(gzip.GzipFile('env-train-3.json').read().decode('utf-8')))
    print('third file')
    data.extend(json.loads(gzip.GzipFile('env-train-4.json').read().decode('utf-8')))
    print('fourth file')
    data.extend(json.loads(gzip.GzipFile('env-train-5.json').read().decode('utf-8')))
    print('all files loaded')

    # data = data[:3000000]
    np.random.shuffle(data)
    print('shuffled')

    # Prepare
    act_vectors = np.eye(88)
    j = 0
    _x = []
    _y = []
    # Populate data
    for d in data:
        if d['r'] == 0.5:
            _x.append(d['obs'])
            _y.append(act_vectors[d['act']])
            j += 1
            if j % 1000 == 0:
                sys.stdout.write(str(j/1000) + ' ')
                sys.stdout.flush()
    # Transform to numpy array
    _x = np.asarray(_x)
    _y = np.asarray(_y)
    # Calculate mean and std
    if t == 'train':
        _mean = _x.mean(axis=0)
        _std = _x.std(axis=0)
        return _x, _y, j, _mean, _std
    return _x, _y, j


x, y, total, mean, std = load_data('train')
print(total)
'''Load Data Finished'''

sess = tf.Session()

n_state = 25
n_action = 88

# Policy network start
state = tf.placeholder(shape=[None, n_state], dtype=tf.float32)
_state = tf.divide(tf.subtract(state, mean), std)
w1 = tf.get_variable("w1", shape=[n_state, 96])
w2 = tf.get_variable("w2", shape=[96, 96])
w3 = tf.get_variable("w3", shape=[96, n_action])
saver = tf.train.Saver({"w1": w1, "w2": w2, "w3": w3})
o1 = tf.matmul(_state, w1)
h1 = tf.math.softmax(o1)
o2 = tf.matmul(h1, w2)
h2 = tf.math.softmax(o2)
action_logits = tf.matmul(h2, w3)
action_prob = tf.math.softmax(action_logits)
# action_dist = tfp.distributions.Categorical(probs=action_prob[0])


action_one_hot = tf.placeholder(shape=[None, n_action], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(tf.subtract(action_one_hot, action_prob)), 1)
mean_loss, std_loss = tf.nn.moments(loss, 0)
rms = tf.train.RMSPropOptimizer(0.001)
opt = rms.minimize(mean_loss)

# Initialization & policy definition
sess.run(tf.global_variables_initializer())

to_plot = []
to_plot_std = []
i = 0
while i < total:
    end = min(i+128, total)
    feed_dict = {state: x[i:end], action_one_hot: y[i:end]}
    loss_m_v, loss_std_v, _ = sess.run([mean_loss, std_loss, opt], feed_dict=feed_dict)
    if i % 100 == 0:
        print(loss_m_v, loss_std_v)
        to_plot.append(loss_m_v)
        to_plot_std.append(loss_std_v)
    i += 128

plt.plot(to_plot)
plt.fill_between([m-std for m, std in zip(to_plot, to_plot_std)], [m+std for m, std in zip(to_plot, to_plot_std)])
plt.show()

saver.save(sess, "policy_model/policyinitial.ckpt")
