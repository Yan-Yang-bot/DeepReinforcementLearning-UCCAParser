from gym.envs.registration import register

register(
    id='drlUcca',
    entry_point='drlUcca.envs:uccaEnv',
)