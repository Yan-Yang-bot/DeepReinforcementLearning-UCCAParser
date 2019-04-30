import tensorflow as tf
import numpy as np
import json, gzip, sys
import matplotlib.pyplot as plt

def loadData(t):
    with gzip.GzipFile('env-{}.json'.format(t), 'r') as fin:
        json_bytes = fin.read()
    print('file read')
    json_str = json_bytes.decode('utf-8')
    print('string decoded')
    data = json.loads(json_str)
    print('json parsed')
    np.random.shuffle(data)
    print('shuffled')

    act_vectors = np.eye(88)
    i = 0
    x = []
    y = []
    for d in data:
        x.append(np.append(d['obs'], act_vectors[d['act']]))
        y.append(d['r'])
        i += 1
        if i%100000==0:
            sys.stdout.write(str(i//100000)+" ")
            sys.stdout.flush()
    x = np.asarray(x)
    y = np.asarray(y)
    if t=='train':
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        return x, y, mean, std
    return x, y


x, y, mean, std = loadData('train')
total = len(x)
print(total)

xt, yt = loadData('dev')
xt = xt[:8000]
yt = yt[:8000]

print("data prepared")

X = tf.placeholder(shape=(None, 113), dtype=tf.float32)
_X = tf.divide(tf.subtract(X, mean),std)
H1 = tf.layers.dense(_X, 256, activation="relu")
H2 = tf.layers.dense(H1, 128, activation="relu")
Y = tf.layers.dense(H2, 1)

L = tf.placeholder(shape=(None, ), dtype=tf.float32)
loss = tf.reduce_mean(tf.square(tf.subtract(Y, L)))
rms = tf.train.RMSPropOptimizer(0.001)
opt = rms.minimize(loss)


sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.save(sess, 'env_r_model', global_step=total)

# batch training, one pass
loss_records = []
i = 0
while i < total:
    end = min(i+128, total)
    loss_value, _ = sess.run([loss, opt], feed_dict={X: x[i:end], L: y[i:end]})
    loss_test, = sess.run([loss], feed_dict={X: xt, L: yt})
    i += 128
    if i%3200==0:
        print("loss: " + loss_value.astype('str'))
        print("test loss: " + loss_test.astype('str'))
        loss_records.append((loss_value, loss_test))

plt.plot(loss_records)
plt.legend(['train mse','test mse'])
plt.savefig('rewardLoss.png')

print('training finished')
