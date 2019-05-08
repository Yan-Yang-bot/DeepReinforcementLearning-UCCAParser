import os, sys
from glob import glob
from collections import OrderedDict

import tensorflow as tf
from ucca import ioutil

from tests.conftest import Settings
from tupa.action import Action
from tupa.states.state import State
from tupa.config import Config
from tupa.features.dense_features import DenseFeatureExtractor


def basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])


def load_passage(filename):
    passages = ioutil.read_files_and_dirs(filename, attempts=1, delay=0)
    try:
        return next(iter(passages))
    except StopIteration:
        return passages


def gen_actions(passage, feature_extractor, policy):

    simpleActions = ['SHIFT', 'REDUCE', 'SWAP', 'FINISH']
    allLabels = ['H', 'A', 'C', 'L', 'D', 'E', 'G', 'S', 'N', 'P', 'R', 'F', 'Terminal', 'U']
    complexActions = ['IMPLICIT', 'NODE', 'RIGHT-EDGE', 'LEFT-EDGE', 'RIGHT-REMOTE', 'LEFT-REMOTE']

    state = State(passage)

    while True:
        obs = feature_extractor.extract_features(state)['numeric']
        for index in [7, 9, 11, 14, 15, 16, 17, 17, 18, 22]:
            del obs[index]
        action = policy(obs)
        a_type = simpleActions[action] if action < 4 else complexActions[(action-4)//14]
        label = None if action < 4 else allLabels[(action-4) % 14]
        action = Action(a_type, tag=label)

        state.transition(action)
        yield str(action)
        if state.finished:
            printStruct(state.root)
            break

def printStruct(r, s=''):
    print(s+str(r))
    for o in r.outgoing:
        print(s+'  '+str(o))
        printStruct(o.child, s+'  ')


def produce_oracle(filename, feature_extractor, policy):
    passage = load_passage(filename)
    for i, action in enumerate(gen_actions(passage, feature_extractor, policy)):
        print(action)


if __name__ == "__main__":
    """define state feature extractor"""
    config = Config()
    setting = Settings('implicit')
    config.update(setting.dict())
    config.set_format("ucca")
    feature_extractor = DenseFeatureExtractor(OrderedDict(),
                                              indexed=config.args.classifier != 'mlp',
                                              hierarchical=False,
                                              node_dropout=config.args.node_dropout,
                                              omit_features=config.args.omit_features)
    """load network"""
    n_state = 25
    n_action = 88
    # Policy network start
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
    action_dist = tf.distributions.Categorical(probs=action_prob)
    
    saver = tf.train.Saver({"w1": w1, "w2": w2, "w3": w3})
    sess = tf.Session()
    saver.restore(sess, "policy_model/policymodel.ckpt")
    action_sample = action_dist.sample(sample_shape=())
    # this is to substitute the Oracle.get_action(state) function
    policy = lambda obs: sess.run([action_sample], feed_dict={state: [obs]})[0][0]
    """network ends"""

    if len(sys.argv) > 1:
        filename = sys.argv[1]
        produce_oracle(filename, feature_extractor, policy)
    else:
        filenames = glob("data/raw/test-xml/*")
        for filename in filenames[:100]:
            print()
            print(filename)
            produce_oracle(filename, feature_extractor, policy)
    sess.close()
