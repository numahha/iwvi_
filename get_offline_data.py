import gym
import custom_gym
import numpy as np

env = gym.make('CustomPendulum-v0')


def random_rollout(e):
    transition_data = []
    o = env.reset()
    while True:
        a = e.action_space.sample()
        o_next, r, done, _ = env.step(a)
        #transition_data.append(np.hstack([o,a,o_next,r]))
        transition_data.append(np.hstack([o,a,o_next-o,r]))
        o = o_next
        if done:
            break
    return np.array(transition_data)


offline_data = []
for i in range(100):
    #print(i)
    offline_data.append(random_rollout(env))
offline_data = np.array(offline_data)
np.save('np_offline_data', offline_data)
