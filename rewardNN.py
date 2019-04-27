import tensorflow as tf
import numpy as np
import json, gzip

with gzip.GzipFile('env-train.json', 'r') as fin:
    json_bytes = fin.read()

json_str = json_bytes.decode('utf-8')

data = json.loads(json_str)
total = len(data)
print('file read', total)
np.random.shuffle(data)
print('shuffled')

x = np.asarray([d['obs'] + [d['act']] for d in data])
mean = x.mean(axis=0)
std = x.std(axis=0)
print('waiting for last step...')
y = np.asarray([d['r'] for d in data])


print("data prepared")

X = tf.placeholder(shape=(None, 26), dtype=tf.float32)
_X = tf.divide(tf.subtract(X, mean),std)
H1 = tf.layers.dense(_X, 64, activation="relu")
H2 = tf.layers.dense(H1, 64, activation="relu")
Y = tf.layers.dense(H2, 1)

L = tf.placeholder(shape=(None, ), dtype=tf.float32)
loss = tf.reduce_sum(tf.square(tf.subtract(Y,L)))
rms = tf.train.RMSPropOptimizer(0.001)
opt = rms.minimize(loss)


sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.save(sess, 'env_r_model', global_step=total)

# batch training, one pass
i = 0
while i < total:
    end = min(i+128, total)
    loss_value, _ = sess.run([loss, opt], feed_dict={X:x[i:end], L:y[i:end]})
    i += 128
    if i%200==0:
        print("loss: " + loss_value.astype('str'))

print('training finished')
