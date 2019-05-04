import os, sys, json, gzip
from glob import glob
from itertools import combinations
from collections import OrderedDict

from ucca import ioutil

from tupa.action import Actions
from tupa.oracle import Oracle
from tupa.states.state import State
from tupa.config import Config
from tupa.features.dense_features import DenseFeatureExtractor
from tests.conftest import Settings



def basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

def passage_files(category):
    return glob("data/raw/{}-xml/*".format(category))
    #return [f for dir in ['dev-xml','train-xml'] for f in glob("data/raw/{}/*".format(dir))]

def load_passage(filename):
    passages = ioutil.read_files_and_dirs(filename, attempts=1, delay=0)
    try:
        return next(iter(passages))
    except StopIteration:
        return passages


envTrainingData = []
allLabels = ['H', 'A', 'C', 'L', 'D', 'E', 'G', 'S', 'N', 'P', 'R', 'F', 'Terminal', 'U']
allActions = [{'type':t, 'hasLabel': False, 'label':None} for t in ['SHIFT', 'REDUCE', 'SWAP', 'FINISH']]
allActions.extend([{'type':t, 'hasLabel':True, 'label':l} for l in allLabels for t in ['IMPLICIT', 'NODE', 'RIGHT-EDGE', 'LEFT-EDGE', 'RIGHT-REMOTE', 'LEFT-REMOTE']])

def gen_actions(passage, feature_extractor):
    global envTrainingData, allLabels, allTypes, allActions
    oracle = Oracle(passage)
    state = State(passage)
    actions = Actions()
    while True:
        acts = oracle.get_actions(state, actions).values()
        type_label_maps = {a.type:a.tag for a in acts} # There should be no duplicate types with different tags since there is only one golden tree
        obs = feature_extractor.extract_features(state)['numeric']
        for index in [7, 9, 11, 14, 15, 16, 17, 17, 18, 22]:
            del obs[index]
        for act in allActions:
            cur_type = act['type']
            cur_has_label = act['hasLabel']
            cur_label = act['label']
            # TODO: Double consider the reward mechanism.
            # Encourage the agent to produce less mistake VS encourage it to produce more correctness:
            # The latter will encourage an episode to go on endlessly, while the former encourage it to end as soon as possible.
            # For now, choose to be neutral: 100% correct = 0.5 reward; 100% wrong = -0.5 reward
            r = -0.5
            if cur_type in list(type_label_maps.keys()): # If action type matches
                r += 0.5
                if cur_has_label and cur_label == type_label_maps[cur_type] or not cur_has_label: # If action has no label or label matches
                    r += 0.5
            actNum = allActions.index(act)
            trainingData = {'obs':obs, 'act':actNum, 'r':r}
            envTrainingData.append(trainingData)
        action = min(acts, key=str)
        state.transition(action)
        s = str(action)
        # if state.need_label:
        #     label, _ = oracle.get_label(state, action)
        #     state.label_node(label)
        #     s += " " + str(label)
        yield s
        if state.finished:
            break

def produce_oracle(filename, feature_extractor):
    passage = load_passage(filename)
    sys.stdout.write('.')
    sys.stdout.flush()
    #store_sequence_to = "data/oracles/%s/%s.txt" % (cat, basename(filename))#, setting.suffix())
    #with open(store_sequence_to, "w", encoding="utf-8") as f:
    #    for i, action in enumerate(gen_actions(passage, feature_extractor)):
    #        pass#print(action, file=f)
    for _ in gen_actions(passage, feature_extractor):
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

    filenames = passage_files(sys.argv[1])
    for filename in filenames: #TODO: solve the problem of "KILLED" while wring file. Use 100 files temporarily before solving this.
        produce_oracle(filename, feature_extractor)

    # dump envTrainingData to a file for further learning in rewardNN.py
    json_str = json.dumps(envTrainingData) + "\n"
    json_bytes = json_str.encode('utf-8')
    with gzip.GzipFile('env-{}.json'.format(sys.argv[1]), 'w') as fout:
        fout.write(json_bytes)
    #with gzip.GzipFile('env-train-copy.json', 'w') as fout:
    #    fout.write(json_bytes)
