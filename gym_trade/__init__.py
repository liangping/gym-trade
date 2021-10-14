from gym.envs.registration import register

register(
    id='trade-v1',
    entry_point='gym_trade.envs:TradeEnv',
)
