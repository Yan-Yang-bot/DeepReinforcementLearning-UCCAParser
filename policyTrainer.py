import tensorflow as tf
import argparse
import gym
import drl_ucca
from glob import glob
from passage2oracles import load_passage

# Preparation for plotting
import matplotlib.pyplot as plt
ave_returns_plot = []

# Other globals
f_n = 0
filenames = glob("data/raw/train-xml/*")
batch_size = 40
sess = tf.Session()
train_data = []

def generate_samples(env):
    """
    :param env: environment handler
    """
    if 'state' not in globals():
        # Get params for action and state spaces
        global train_data, state, action_dist, f_n, filenames
        n_state = 25
        n_action = 88

        # Policy network start
        ''' for n1 '''
        state = tf.placeholder(shape=[None, n_state], dtype=tf.float32)
        w1 = tf.get_variable("w1", shape=[n_state, 96])
        w2 = tf.get_variable("w2", shape=[96, 96])
        w3 = tf.get_variable("w3", shape=[96, n_action])
        o1 = tf.matmul(state, w1)
        h1 = tf.math.softmax(o1)
        o2 = tf.matmul(h1, w2)
        h2 = tf.math.softmax(o2)
        action_logits = tf.matmul(h2, w3)
        action_prob = tf.math.softmax(action_logits)
        action_dist = tf.distributions.Categorical(probs=action_prob[0])

        # Initialization & policy definition
        sess.run(tf.global_variables_initializer())

    # Collect training data
    action_sample = action_dist.sample(sample_shape=())
    policy = lambda obs:sess.run([action_sample],feed_dict={state:[obs]})[0]
    train_data=[]
    print("Episode lengths: ")
    for _ in range(batch_size):
        #TODO: Stop when all files are exhausted. Check f_n >= len(filenames), and consider:
        #TODO:  1.  len(train_data) != batch_size
        #TODO:  2.  Not only end episode, but also jump out of the n_iter loop in main immediately to plot part
        obs = env.reset(load_passage(filenames[f_n]))
        f_n += 1
        traj = []
        t = 0
        while True:
            traj.append({'obs':obs})
            action = policy(obs)
            traj[t]['act']=action
            obs, r, done, info = env.step(action)
            traj[t]['R']=r
            t += 1
            if done:
                print(t, end=", ")
                break
        train_data.append(traj)


def estimate_return(lmbd):
    """
    :param lmbd: lambda value
    """
    # Calculate returns
    for j in range(batch_size):
        traj = train_data[j]
        l = len(traj)
        i = l-1
        while i>0:
            traj[i-1]['R'] += traj[i]['R']
            i-=1
        mult = lmbd
        for i in range(1,l):
            traj[i]['R'] *= mult
            mult *= lmbd


def update_policy(alpha):
    """
    :param alpha: learning rate
    """
    # Build dataflow for G * log probabilities of sampled actions
    act = tf.placeholder(shape=(), dtype=tf.int32)
    G = tf.placeholder(shape=(), dtype=tf.float32)
    obj = tf.multiply(G, action_dist.log_prob(act))
    # Optimizer pre-definition (minimizing negative object == maximizing object)
    neg_obj = tf.scalar_mul(-1,obj)
    adam = tf.train.AdamOptimizer(learning_rate=alpha)
    opt = adam.minimize(neg_obj)
    # Initialization for newly introduced variables
    adam_vars = adam.variables()
    sess.run(tf.variables_initializer(var_list=adam_vars))
    # Optimizing using training data collected
    return_sum = 0
    for j in range(batch_size):
        traj = train_data[j]
        for k in range(len(traj)):
            cur = traj[k]
            feed_dict={state: [cur['obs']],
                       act: cur['act'],  G: cur['R']}
            obj_value, _ = sess.run([obj, opt], feed_dict=feed_dict)
            print("object:", obj_value)
        return_sum += traj[0]['R']
    # Logging & Recording for later plot
    print(len(ave_returns_plot), return_sum/batch_size)
    ave_returns_plot.append(return_sum/batch_size)


def main(args):

    # Initialize environment
    env = gym.make('drlenv-v0')

    # Iterations
    for _ in range(args.n_iter):
        generate_samples(env)
        estimate_return(args.lmbd)
        update_policy(args.alpha)

    # Plot
    env.close()
    plt.plot(ave_returns_plot)
    plt.ylabel('mean return')
    plt.xlabel('# of iterations')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_iter", type=int, default=20)
    parser.add_argument("--lmbd", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.1)

    main(parser.parse_args())
