import numpy as np
import torch
import gym
import time
import torch.nn as nn
import torch.nn.functional as F
from gym_puyopuyo_master.gym_puyopuyo.agent import TsuTreeSearchAgent
from gym_puyopuyo_master.gym_puyopuyo import register
from icecream import ic



class NeuralNetwork(nn.Module):
    """
        NeuralNetwork関係のクラス
    """
    def __init__(self, acts_num):
        super(NeuralNetwork, self).__init__()
        self.fc2 = nn.Linear(24 + 4*13*6 , 128)
        self.fc3 = nn.Linear(128,128)
        self.fc4 = nn.Linear(128,64)
        self.fc5 = nn.Linear(64, acts_num)

        self.sf = nn.Softmax(dim=0)

    def forward(self, x):
        h = F.relu(self.fc2(x))
        h = F.relu(self.fc3(h))
        h = F.relu(self.fc4(h))
        h = self.fc5(h)
        return h

    def softmax(self, x):
        return self.sf(x)


def run(env):
    state = env.reset()
    policy = NeuralNetwork(env.action_space.n)
    model_PATH = 'model/' + '04 Feb 2022 10:45/model-10:45:26.pth'
    policy.load_state_dict(torch.load(model_PATH))

    for _ in range(100):
        state = env.reset()
        pstate_r0 = state[0].flatten()
        pstate_r1 = state[1].flatten()
        pstate_r = np.concatenate((pstate_r0, pstate_r1), axis = 0)
        pstate_t = torch.from_numpy(pstate_r.astype(np.float32)).clone()
        done = False
        ic("-------------------------------------")
        for _ in range(500):
            env.render()
            time.sleep(0.6)
            action = policy.forward(pstate_t)
            action2 = action.to('cpu').detach().numpy().copy()
            #index = softmax_selection(action2)
            index = torch.argmax(action)
            ic(index)
            state, _ , done , _ = env.step(index)
            pstate_r0 = state[0].flatten()
            pstate_r1 = state[1].flatten()
            pstate_r = np.concatenate((pstate_r0, pstate_r1), axis = 0)
            pstate_t = torch.from_numpy(pstate_r.astype(np.float32)).clone()
            if done:
                break

def softmax_selection(values):
    """
        softmax 行動選択
        @input : values : numpy.array型
    """
    tau = 0.4
    sum_exp_values = sum([np.exp(v/tau) for v in values])   # softmax選択の分母の計算
    p = [np.exp(v/tau)/sum_exp_values for v in values]      # 確率分布の生成
    action = np.random.choice(np.arange(len(values)), p=p)  # 確率分布pに従ってランダムで選択
    return action

if __name__ == '__main__':
    register()
    env = gym.make("PuyoPuyoEndlessTsu-v2")
    run(env)