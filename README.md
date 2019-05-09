# DRL_UCCA
A Deep Reinforcement Learning Parser for UCCA.


## Preparations

1. Run `pip install -r requirement.txt`.

2. Run `python setup.py install`.

3. Check README instructions on [data](https://github.com/DIMPLY/DRL_UCCA/tree/master/data/raw) directory page.

## Test the result
Use the name of any xml file under `data/raw/test-xml` directory (including the path) to run:
```
python policyTester.py <filename>
```
***You can test the result of our ready trained model directly, or you can complete the following steps first to get you own trained model.***

## Train the reward function
Use the following to produce reward function training data:
```
python passage2oracles.py
```
It will produce and store a json binary file.
This will take a long time, and you can omit this step and test the rest of the code using `rwdFuncTrainData_smal_sample_set.json`, a small sample of the training data.

Train the reward function:
```
python rewardNN.py
```
This will store some files containing the final state of the model at the end of the running.

## The Reinforcement Learning part
The environment is set in `drl_ucca` folder, with the trained and stored reward function model plugged in.
The Reinforcement Learning part, `policyTrainer.py`, will use this environment as a black box.
Use:
```
python policyInitializer.py # This is to initialize the weights of our trainable parameters better than random
python policyTrainer.py # The silence mode
python policyTrainer.py -e # This will output all oracles predicted
```

