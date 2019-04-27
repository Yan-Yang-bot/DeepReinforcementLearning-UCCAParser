from collections import OrderedDict
from glob import glob
import gym
import tensorflow as tf
from passage2oracles import Settings
from tupa.action import Action
from tupa.states.state import State
from tupa.config import Config
from tupa.features.dense_features import DenseFeatureExtractor

class UccaEnv(gym.Env):

    simpleActions = ['SHIFT', 'REDUCE', 'SWAP', 'FINISH']
    allLabels = ['H', 'A', 'C', 'L', 'D', 'E', 'G', 'S', 'N', 'P', 'R', 'F', 'Terminal', 'U']
    complexActions = ['IMPLICIT', 'NODE', 'RIGHT-EDGE', 'LEFT-EDGE', 'RIGHT-REMOTE', 'LEFT-REMOTE']

    metadata = {'render.modes':['human']}

    def __init__(self):
        config = Config()
        setting = Settings(*('implicit'))
        config.update(setting.dict())
        config.set_format("ucca")
        self.feature_extractor = DenseFeatureExtractor(OrderedDict(),
                                              indexed = config.args.classifier!='mlp',
                                              hierarchical=False,
                                              node_dropout=config.args.node_dropout,
                                              omit_features=config.args.omit_features)

        self.sess=tf.Session()
        saver = tf.train.import_meta_graph(glob('env_r_model-*.meta')[0])
        saver.restore(self.sess, tf.train.latest_checkpoint('./'))
        graph = tf.get_default_graph()
        self.x = graph.get_tensor_by_name("Placeholder:0")
        self.y = graph.get_tensor_by_name("dense_2/BiasAdd:0")

    def get_feature(self):
        f = self.feature_extractor.extract_features(self.state)[numeric]
        for index in [7, 9, 11, 14, 15, 16, 17, 17, 18, 22]:
            del f[index]
        return f

    def step(self, action):
        """

        Parameters
        ----------
        action :

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        # Get reward
        r = self._get_reward(action)
        # Take action
        type = self.simpleActions[action] if action < 4 else self.complexActions[(action-4)//14]
        label = None if action < 4 else self.allLabels[(action-4)%14]
        act = Action(type, tag=label)
        self.state.transition(act)
        # Get new state
        self.stateVec = self.get_feature()
        return self.stateVec, r, self.state.finished, ''

    def reset(self, passage):
        self.state = State(passage)
        self.stateVec = self.get_feature()
        return self.stateVec

    def _get_reward(self, actNum):
        input = self.stateVec + [actNum]
        return self.sess.run(self.y, feed_dict={self.x:[input]})[0][0]
