from gym.envs.registration import register

register(
    id='drlenv-v0',
    entry_point='drl_ucca.envs:UccaEnv',
)