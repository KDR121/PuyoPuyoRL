# from gym_puyopuyo_master.gym_puyopuyo import register
# from gym import make

# register()
# small_env = make("PuyoPuyoEndlessSmall-v2")

# for i in range(10):
#     small_env.step(small_env.action_space.sample())
#     small_env.render()

# small_env.render()

from gym.envs.registration import make

from gym_puyopuyo_master.gym_puyopuyo.agent import TsuTreeSearchAgent
from gym_puyopuyo_master.gym_puyopuyo import register

register()

agent = TsuTreeSearchAgent()

env = make("PuyoPuyoEndlessTsu-v2")

env.reset()
state = env.get_root()

for i in range(100):
    action = agent.get_action(state)
    _, _, done, info = env.step(action)
    state = info["state"]
    env.render()
    if done:
        break

env.render()