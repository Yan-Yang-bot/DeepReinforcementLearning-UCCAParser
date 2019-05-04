import tensorflow as tf
import numpy as np
import json, gzip, sys
import matplotlib.pyplot as plt

def loadData(t):
    if t == "train":
        with gzip.GzipFile('env-{}-1.json'.format(t), 'r') as fin:
            json_bytes = fin.read()
    else:
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
    ncount = 0
    zcount = 0
    pcount = 0

    for index,d in enumerate(data[:min(10000000,len(data))]):
        if d['r'] == -0.5:
            
            if ncount <= 102101 or t == 'dev':
                ncount+= 1
                x.append(np.append(d['obs'], act_vectors[d['act']]))
                y.append(d['r'])
        else:
            if d['r'] == 0:
                
                if zcount <= 102101 or t == 'dev':
                    zcount+=  1
                    x.append(np.append(d['obs'], act_vectors[d['act']]))
                    y.append(d['r'])
            elif d['r'] == 0.5:
                pcount +=1        
                x.append(np.append(d['obs'], act_vectors[d['act']]))
                y.append(d['r'])
        i += 1
        if i%100000==0 and i>0:
            if t == 'dev' and i >=2*100000:
                break
            sys.stdout.write(str(i//100000)+" ")
            sys.stdout.flush()
    
    

    x = np.asarray(x)
    y = np.asarray(y)
    print('summary')
    print(ncount)
    print(zcount)
    print(pcount)
    if t=='train':
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        return x, y, mean, std
    return x, y


x, y, mean, std = loadData('train')
total = len(x)
print(total)

xt, yt = loadData('dev')
xt = xt[:20000]
yt = yt[:20000]

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
        saver.save(sess, 'env_r_model', global_step=i)        
plt.plot(loss_records)
plt.legend(['train mse','test mse'])
plt.savefig('rewardLoss.png')

print('training finished')
