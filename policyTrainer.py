import gym
import drl_ucca
env = gym.make('drlenv-v0')
N = 4

# TODO: build and initialize a policy network
def policy(obs):
    return [0,0,0]




# TODO: build loss calculation graph and the optimizer





ave_returns = []
returns = []
epi = 0
while epi < args.nPlotPoints * N:
    obs = env.reset()
    done = False
    ret = 0
    while not done:
        action = policy(obs)     # TODO: use the policy network to predict an action
        obs, r, done, _ = obs.step(action)
        ret += r
    returns.append(ret)
    epi += 1
    if epi%N == 0:
        ave_returns.append(sum(returns)/N)
        returns = []

# TODO: update policy somewhere? I don't remember where should this plug in.
# TODO: Need to check homework.

import json
json.dump(returns, open('ave_ret_plot.json','w'))