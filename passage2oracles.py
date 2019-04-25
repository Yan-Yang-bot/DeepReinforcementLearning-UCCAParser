import os
from glob import glob
from itertools import combinations

from ucca import ioutil

from tupa.action import Actions
from tupa.oracle import Oracle
from tupa.states.state import State

def basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

def passage_files():
    return [[f for f in glob("data/raw/{}/*".format(dir))] for dir in ['dev-xml','train-xml']]

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

def gen_actions(passage):
    oracle = Oracle(passage)
    state = State(passage)
    actions = Actions()
    while True:
        action = min(oracle.get_actions(state, actions).values(), key=str)
        trainingData = {}
        #TODO: transform state/action -> representation, store with reward=1
        trainingData['obs'] = #DenseFeatureExtractor
        trainingData['act'] = [action.type, action.type_id, action.tag]
        #TODO: for other actions, store with reward=0 and state=the same state
        #TODO: store all those to `envTrainingData`
        envTrainingData.append(trainingData)
        state.transition(action)
        s = str(action)
        if state.need_label:
            label, _ = oracle.get_label(state, action)
            state.label_node(label)
            s += " " + str(label)
        yield s
        if state.finished:
            break

def produce_oracle(cat, filename):
    setting = Settings(*('implicit', 'linkage'))
    passage = load_passage(filename)
    print(filename)
    store_sequence_to = "data/oracles/%s/%s%s.txt" % (cat, basename(filename), setting.suffix())
    with open(store_sequence_to, "w", encoding="utf-8") as f:
        print(passage, file=f)
        for i, action in enumerate(gen_actions(passage)):
            print(action, file=f)



if __name__=="__main__":
    filenames = passage_files()
    c = 'dev'
    for cat in filenames:
        for filename in cat:
            produce_oracle(c, filename)
        c = 'train'

    #TODO: dump envTrainingData to a file for further learning in rewardNN.py