import os, sys, json, gzip
from glob import glob
from itertools import combinations
from collections import OrderedDict
from tupa.action import Action
from ucca import ioutil
import tensorflow as tf
from tupa.action import Actions
from tupa.oracle import Oracle
from tupa.states.state import State
from tupa.config import Config
from tupa.features.dense_features import DenseFeatureExtractor

def basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

def passage_files():
    return [f for dir in ['test-xml'] for f in glob("data/raw/{}/*".format(dir))]

def load_passage(filename):
    passages = ioutil.read_files_and_dirs(filename, attempts=1, delay=0)
    try:
        return next(iter(passages))
    except StopIteration:
        return passages

class Settings:
    SETTINGS = ("implicit", "linkage", "unlabeled")
    VALUES = {"unlabeled": (None, [])}
    INCOMPATIBLE = (("linkage", "unlabeled"),)

    def __init__(self, *args):
        for attr in self.SETTINGS:
            setattr(self, attr, attr in args)

    @classmethod
    def all(cls):
        return [Settings(*c) for n in range(len(cls.SETTINGS) + 1) for c in combinations(cls.SETTINGS, n)
                if not any(all(s in c for s in i) for i in cls.INCOMPATIBLE)]

    def dict(self):
        return {attr: self.VALUES.get(attr, (False, True))[getattr(self, attr)] for attr in self.SETTINGS}

    def list(self):
        return [attr for attr in self.SETTINGS if getattr(self, attr)]

    def suffix(self):
        return "_".join([""] + self.list())

    def __str__(self):
        return "-".join(self.list()) or "default"

envTrainingData = []
allLabels = ['H', 'A', 'C', 'L', 'D', 'E', 'G', 'S', 'N', 'P', 'R', 'F', 'Terminal', 'U']
allActions = [{'type':t, 'hasLabel': False, 'label':None} for t in ['SHIFT', 'REDUCE', 'SWAP', 'FINISH']]
allActions.extend([{'type':t, 'hasLabel':True, 'label':l} for l in allLabels for t in ['IMPLICIT', 'NODE', 'RIGHT-EDGE', 'LEFT-EDGE', 'RIGHT-REMOTE', 'LEFT-REMOTE']])

def gen_actions(passage, feature_extractor,policy):
    global envTrainingData, allLabels, allTypes, allActions

    simpleActions = ['SHIFT', 'REDUCE', 'SWAP', 'FINISH']
    allLabels = ['H', 'A', 'C', 'L', 'D', 'E', 'G', 'S', 'N', 'P', 'R', 'F', 'Terminal', 'U']
    complexActions = ['IMPLICIT', 'NODE', 'RIGHT-EDGE', 'LEFT-EDGE', 'RIGHT-REMOTE', 'LEFT-REMOTE']

    state = State(passage)
    actions = Actions()

    while True:
        obs = feature_extractor.extract_features(state)['numeric']
        for index in [7, 9, 11, 14, 15, 16, 17, 17, 18, 22]:
            del obs[index]
        action = policy(obs)
        type = simpleActions[action] if action < 4 else complexActions[(action-4)//14]
        label = None if action < 4 else allLabels[(action-4)%14]
        action  = Action(type,tag = label)

        state.transition(action)
        s = str(action)
        print(s)
        yield s
        if state.finished:
            break

def produce_oracle(filename, feature_extractor,policy):
    passage = load_passage(filename)
    sys.stdout.write('.')
    sys.stdout.flush()
    #store_sequence_to = "data/oracles/%s/%s.txt" % (cat, basename(filename))#, setting.suffix())
    #with open(store_sequence_to, "w", encoding="utf-8") as f:
    #    for i, action in enumerate(gen_actions(passage, feature_extractor)):
    #        pass#print(action, file=f)
    for _ in gen_actions(passage, feature_extractor,policy):
        pass



if __name__=="__main__":
    config = Config()
    setting = Settings(*('implicit'))
    config.update(setting.dict())
    config.set_format("ucca")
    feature_extractor = DenseFeatureExtractor(OrderedDict(),
                                              indexed = config.args.classifier!='mlp',
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
    saver = tf.train.Saver({"w1":w1, "w2":w2,"w3":w3})
    o1 = tf.matmul(state, w1)
    h1 = tf.math.softmax(o1)
    o2 = tf.matmul(h1, w2)
    h2 = tf.math.softmax(o2)
    action_logits = tf.matmul(h2, w3)
    action_prob = tf.math.softmax(action_logits)
    action_dist = tf.distributions.Categorical(probs=action_prob[0])
    
    saver = tf.train.Saver({"w1":w1, "w2":w2,"w3":w3})
    sess = tf.Session()
    saver.restore(sess,"policy_model/policymodel.ckpt")
    action_sample = action_dist.sample(sample_shape=())
    # this is to substitute the Oracle.get_action(state) function
    policy = lambda obs:sess.run([action_sample],feed_dict={state:[obs]})[0]
    """network ends"""
    filenames = passage_files()
    for filename in filenames[:100]: #TODO: solve the problem of "KILLED" while wring file. Use 100 files temporarily before solving this.
        
        produce_oracle(filename, feature_extractor, policy)
    sess.close()

    