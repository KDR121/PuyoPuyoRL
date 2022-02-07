# coding: utf-8
import copy
import time
import numpy as np
import os
import gym
from gym import wrappers
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from icecream import ic
from gym_puyopuyo_master.gym_puyopuyo import register
from gym.envs.registration import make
from  gym_puyopuyo_master.gym_puyopuyo.agent import TsuTreeSearchAgent


class ConNeuralNetwork(nn.Module):
    """
        ConNeuralNetwork関係のクラス
    """
    def __init__(self, acts_num):
        super(ConNeuralNetwork, self).__init__()
        self.fc1 = nn.Conv2d(  4,  8, kernel_size=2, stride=1)
        self.fc2 = nn.Conv2d(  8, 16, kernel_size=2, stride=1)
        self.fc3 = nn.Conv2d( 16, 32, kernel_size=2, stride=1)
        self.fc4 = nn.Flatten()
        self.fc5 = nn.Linear(960, acts_num)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        h = self.fc4(h)
        h = self.fc5(h)
        return h

class Agent_DeepQNetwork():
    """
        DeeQNetworkによるAgentを示すクラス
    """
    def __init__(self):
        register()
        agent = TsuTreeSearchAgent
        # 環境
        self.MONITOR = False
        self.env = gym.make("PuyoPuyoEndlessTsu-v2")
        if self.MONITOR:
            self.env = wrappers.Monitor(self.env, "./tmp", force=True)

        self.obs_num_next_puyo = 4 * 3 * 2
        self.obs_num_now_stage = 4 * 13 * 6
        self.obs_num_height = 13
        self.obs_num_weight = 6
        self.obs_num_dim = 4
        self.obs_num  =self.obs_num_next_puyo + self.obs_num_now_stage
        self.acts_num = self.env.action_space.n
        HIDDEN_SIZE = 100
        # 定数
        self.EPOCH_NUM = 10000000 # エポック数
        self.STEP_MAX = 500 # 最高ステップ数
        self.MEMORY_SIZE = 200 # メモリサイズいくつで学習を開始するか
        self.BATCH_SIZE = 50 # バッチサイズ
        self.EPSILON = 1.0 # ε-greedy法
        self.EPSILON_DECREASE = 0.000001 # εの減少値
        self.EPSILON_MIN = 0.1 # εの下限
        self.START_REDUCE_EPSILON = 200 # εを減少させるステップ数
        self.TRAIN_FREQ = 10 # Q関数の学習間隔
        self.UPDATE_TARGET_Q_FREQ = 20 # Q関数の更新間隔
        self.GAMMA = 0.97 # 割引率
        self.LOG_FREQ = 1000 # ログ出力の間隔

        # 出力用
        self.writer = SummaryWriter()
        return

    def main(self):

        #グラボの設定
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ic(self.device)
        # モデル
        NN = ConNeuralNetwork(acts_num= self.acts_num).to(self.device) # 近似Q関数
        Q_ast = copy.deepcopy(NN)

        optimizer = optim.RMSprop(NN.parameters(), lr=0.00015, alpha=0.95, eps=0.01)
        total_step = 0 # 総ステップ（行動）数
        memory = [] # メモリ
        total_rewards = [] # 累積報酬記録用リスト
        # 学習用ディレクトリ作成
        mkdir_PATH = './model/' + time.strftime("%d %b %Y %H:%M", time.gmtime())
        os.mkdir(mkdir_PATH)
        # 学習開始
        print("Train")
        print("\t".join(["epoch", "self.EPSILON", "reward", "total_step", "elapsed_time"]))
        start = time.time()
        for epoch in range(self.EPOCH_NUM):
            pobs = self.env.reset() # 環境初期化
            step = 0 # ステップ数
            done = False # ゲーム終了フラグ
            total_reward = 0 # 累積報酬

            while not done and step < self.STEP_MAX:
                if self.MONITOR:
                    self.env.render()

                # 行動選択
                pact = self.env.action_space.sample()
                pobs_r = pobs[1]
                # ε-greedy法
                if np.random.rand() > self.EPSILON:

                    # 最適な行動を予測
                    pobs_r = pobs[1]
                    pobs_ = np.array(pobs_r, dtype="float32").reshape((1, self.obs_num_dim, self.obs_num_height, self.obs_num_weight))
                    pobs_ = Variable(torch.from_numpy(pobs_)).to(self.device)
                    pact = NN.forward(pobs_)
                    maxs, indices = torch.max(pact.data, 1)
                    pact = indices.cpu().numpy()[0]

                # 行動
                obs, reward, done, _ = self.env.step(pact)
                reward /= 5000
                if done:
                    reward = -1
                obs_r = obs[1]
                # メモリに蓄積
                memory.append((pobs_r, pact, reward, obs_r, done)) # 状態、行動、報酬、行動後の状態、ゲーム終了フラグ
                if len(memory) > self.MEMORY_SIZE: # メモリサイズを超えていれば消していく
                    memory.pop(0)

                # 学習
                if len(memory) == self.MEMORY_SIZE: # メモリサイズ分溜まっていれば学習

                    # 経験リプレイ
                    if total_step % self.TRAIN_FREQ == 0:

                        memory_ = np.random.permutation(memory)
                        memory_idx = range(len(memory_))

                        for i in memory_idx[::self.BATCH_SIZE]:
                            batch = np.array(memory_[i:i+self.BATCH_SIZE]) # 経験ミニバッチ
                            pobss = np.array(batch[:,0].tolist(), dtype="float32").reshape((self.BATCH_SIZE, self.obs_num_dim, self.obs_num_height, self.obs_num_weight))
                            pacts = np.array(batch[:,1].tolist(), dtype="int32")
                            rewards = np.array(batch[:,2].tolist(), dtype="int32")
                            obss = np.array(batch[:,3].tolist(), dtype="float32").reshape((self.BATCH_SIZE, self.obs_num_dim, self.obs_num_height, self.obs_num_weight))
                            dones = np.array(batch[:,4].tolist(), dtype="bool")

                            pobss_ = Variable(torch.from_numpy(pobss)).to(self.device)
                            q = NN.forward(pobss_)
                            obss_ = Variable(torch.from_numpy(obss)).to(self.device)
                            maxs, indices = torch.max(Q_ast(obss_).data, 1)
                            maxq = maxs.cpu().numpy() # maxQ
                            target = copy.deepcopy(q.cpu().data.numpy())
                            for j in range(self.BATCH_SIZE):
                                target[j, pacts[j]] = rewards[j]+self.GAMMA*maxq[j]*(not dones[j]) # 教師信号

                            # Perform a gradient descent step
                            optimizer.zero_grad()
                            loss = nn.MSELoss()(q, Variable(torch.from_numpy(target)).to(self.device))
                            loss.backward()
                            optimizer.step()

                    # Q関数の更新
                    if total_step % self.UPDATE_TARGET_Q_FREQ == 0:
                        Q_ast = copy.deepcopy(NN)

                # εの減少
                if self.EPSILON > self.EPSILON_MIN and total_step > self.START_REDUCE_EPSILON:
                    self.EPSILON -= self.EPSILON_DECREASE

                # 次の行動へ
                total_reward += reward
                step += 1
                total_step += 1
                pobs = obs

            total_rewards.append(total_reward) # 累積報酬を記録
            if (epoch + 1) % self.LOG_FREQ == 0:
                r = sum(total_rewards[((epoch+1)-self.LOG_FREQ):(epoch+1)])/self.LOG_FREQ # ログ出力間隔での平均累積報酬
                elapsed_time = time.time()-start
                print("\t".join(map(str,[epoch+1, self.EPSILON, r, total_step, str(elapsed_time)+"[sec]"]))) # ログ出力
                start = time.time()
                self.writer.add_scalar('elapsed_time', elapsed_time, epoch)
            self.writer.add_scalar('reward', total_reward, epoch)
            self.writer.add_scalar('EPSILON', self.EPSILON, epoch)

            if (epoch + 1) % (self.LOG_FREQ * 10) == 0:
                #途中のモデルを保存する
                model_name = mkdir_PATH + '/model-' + time.strftime("%H:%M:%S", time.gmtime()) + '.pth'
                torch.save(NN.state_dict(), model_name)

        if self.MONITOR:
            self.env.render(close=True)

        #最終的なモデルを保存する
        model_name = 'model/' + mkdir_PATH + '/model-' + time.strftime("%H:%M:%S", time.gmtime()) + '.pth'
        torch.save(NN.state_dict(), model_name)
        return


if __name__ == '__main__':
    DQN = Agent_DeepQNetwork()
    DQN.main()