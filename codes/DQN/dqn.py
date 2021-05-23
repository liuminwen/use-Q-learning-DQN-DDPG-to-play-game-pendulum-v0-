import numpy as np
import pandas as pd
import time
import gym
import csv
import os
import pickle
from queue import Queue
import pickle
import random
from itertools import count 


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from tensorboardX import SummaryWriter


max_length_of_trajectory = 500
MEMORY_CAPACITY = 5000
BATCH_SIZE = 64
MAX_EPISODE = 20000
sample_frequency = 256
log_interval = 50
test_iteration=10

device = 'cuda' if torch.cuda.is_available() else 'cpu'
directory = './runs'

class Critic(nn.Module):
    """docstring for Critic"""
    def __init__(self,state_dim,action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim,64)
        self.l2 = nn.Linear(64,32)
        self.l3 = nn.Linear(32,action_dim)
        
    def forward(self,x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

class Replay_buffer():
    def __init__(self,max_size=MEMORY_CAPACITY):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self,data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr+1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self,batch_size):
        ind = np.random.randint(0,len(self.storage),size=batch_size)
        x,y,index,u,r,d = [],[],[],[],[],[]

        for i in ind:
            X,Y,I,U,R,D = self.storage[i]
            x.append(np.array(X,copy=False))
            y.append(np.array(Y,copy=False))
            index.append(np.array(I,copy=False))
            u.append(np.array(U,copy=False))
            r.append(np.array(R,copy=False))
            d.append(np.array(D,copy=False))
        return np.array(x),np.array(y),np.array(index),np.array(u),np.array(r),np.array(d)


class DQN:
    def __init__(self,tau=0.005,state_dim=3,test='False',learning_rate=1e-2, reward_decay=0.99, e_greedy=1,epsilon_decay=0.995,epsilon_min=0.01,):
        #self.target                     # 目标状态（终点）
        self.test=test
        self.lr = learning_rate         # 学习率
        self.gamma = reward_decay       # 回报衰减率
        self.epsilon = e_greedy         # 探索/利用 贪婪系数
        self.epsilon_decay=epsilon_decay    
        self.epsilon_min = epsilon_min     
        self.tau=tau
        self.replace_target_iter=3  #每隔多少步更新一次target参数
        # self.num_cos = 10               #分为多少份
        # self.num_sin = 10               
        # self.num_dot = 10 
        self.num_actions = 10 
        self.actions = self.toBins(-2.0, 2.0, self.num_actions)    # 可以选择的动作空间  离散化
        #########=============== 状态空间离散化，但实际上DQN的状态空间可以不离散，如果离散的话，取消注释digital、digitize_state相关的函数===========
        # self.cos_bins = self.toBins(-1.0, 1.0, self.num_cos)
        # self.sin_bins = self.toBins(-1.0, 1.0, self.num_sin)
        # self.dot_bins = self.toBins(-8.0, 8.0, self.num_dot)
        self.state_dim=state_dim
        #
        self.critic = Critic(self.state_dim,self.num_actions).to(device)
        self.critic_target = Critic(self.state_dim,self.num_actions).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(),self.lr)

        self.replay_buffer = Replay_buffer()
        self.writer = SummaryWriter(directory)
        self.num_critic_update_iteration = 0
        self.num_training = 0
        #



    # 根据本次的行动及其反馈（下一个时间步的状态），返回下一次的最佳行动
    

    # 分箱处理函数，把[clip_min,clip_max]区间平均分为num段，  如[1,10]分为5.5 
    def toBins(self,clip_min, clip_max, num):
        return np.linspace(clip_min, clip_max, num + 1)[1:-1] #第一项到倒数第一项

    # 分别对各个连续特征值进行离散化  如[1,10]分为5.5  小于5.5取0  大于取5.5取1
    def digit(self,x, bin): 
        n = np.digitize(x,bins = bin)
        return n

    # 将观测值observation离散化处理
    def digitize_state(self,observation):
        # 将矢量打散回3个连续特征值
        cart_sin, cart_cos , cart_dot = observation
        # 分别对各个连续特征值进行离散化（分箱处理）
        digitized = [self.digit(cart_sin,self.cos_bins),
                    self.digit(cart_cos,self.sin_bins),
                    self.digit(cart_dot,self.dot_bins),]
        # 将离散值再组合为一个离散值，作为最终结果
        return (digitized[1]*self.num_cos + digitized[0]) * self.num_dot + digitized[2]

    def act(self,state): 
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        epsilon = 0.01 if self.test else self.epsilon  # use epsilon = 0.01 when testing
        state = torch.FloatTensor(state.reshape(1,-1)).to(device)
        q_values = self.critic(state).cpu().data.numpy().flatten()
        #self.summaries['q_val'] = max(q_values)
        if np.random.random() < epsilon:
            action_index = np.random.choice(self.num_actions)# sample random action
        else:
            action_index = np.argmax(q_values)
        action = -2 + 4/(self.num_actions-1) * action_index  
        return action_index,action


    # update，主要是更新Q值
    def update(self):
        x,y,index,u,r,d = self.replay_buffer.sample(BATCH_SIZE)
        state = torch.FloatTensor(x).to(device)
        action = torch.FloatTensor(u).to(device)
        next_state = torch.FloatTensor(y).to(device)
        reward = torch.FloatTensor(r).to(device)
        done = torch.FloatTensor(d).to(device)

        # compute the target Q value
        target_Q = self.critic_target(next_state).max(axis=1)[0].reshape([64,1])
        target_Q=reward.reshape([64,1]) + ((((1-done)*self.gamma).reshape([64,1]))*target_Q)
        # get current Q estimate
        #current_Q = self.critic(state).max(axis=1)[0].reshape([64,1])
        current_Q = self.critic(state)[np.arange(64),index].reshape([64,1])

        # compute critic loss
        critic_loss = F.mse_loss(current_Q,target_Q)
        self.writer.add_scalar('Loss',critic_loss,global_step=self.num_critic_update_iteration)

        # optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
#       # update the frozen target models
        if self.num_critic_update_iteration % self.replace_target_iter == 0:
            for param,target_param in zip(self.critic.parameters(),self.critic_target.parameters()):
                target_param.data.copy_(param.data)

        self.num_critic_update_iteration += 1

    def save(self):
        torch.save(self.critic.state_dict(),directory+'critic.pth')
        print('model has been saved...')

    def load(self):
        self.critic.load_state_dict(torch.load(directory+'critic.pth'))
        print('model has been loaded...')

def train():
    ep_r=0
    env = gym.make('Pendulum-v0')   
    #print(env.action_space)
    agent = DQN(learning_rate=1e-2)
    # with open(os.getcwd()+'/tmp/Pendulum.model', 'rb') as f:
    #     agent = pickle.load(f)
    action = [0]    #输入格式要求 要是数组
    for i in range(MAX_EPISODE):  #训练次数
        state = env.reset()  #状态  cos(theta), sin(theta) , thetadot角速度
        #state = agent.digitize_state(state)  #状态标准化
        for t in count():   #一次训练最大运行次数
            action_index,action[0] = agent.act(state)  #动作 -2到2
            next_state, reward, done, info = env.step(action)   
            #next_state = agent.digitize_state(next_state)
            #下面这两个if语句可以加快训练
            # if done:
            #     reward-=200  #对于一些直接导致最终失败的错误行动，其报酬值要减200
            # if reward >= -0.5:  #竖直时时reward接近0  -10到0
            #     reward+=20   #给大一点
            #     #print('arrive')
            # # print(action,reward,done,state,next_state)
            ep_r+=reward
            agent.replay_buffer.push((state,next_state,action_index,action[0],reward,float(done)))
            #agent.learn(state,action[0],reward,next_state)
            state = next_state
            agent.update()
            if done or t>=max_length_of_trajectory:
                agent.writer.add_scalar('ep_r',ep_r,global_step=i)
                if i % 10 ==0:
                    print('Episode:{}, Return:{:0.2f}, Step:{}'.format(i,ep_r,t))
                ep_r = 0
                break
            # env.render()    # 更新并渲染画面
        if (i+1) % 100 == 0:
                print('Episode:{}, Memory size:{}'.format(i,len(agent.replay_buffer.storage)))

        if i % log_interval == 0:
            agent.save()

    
def test():
    ep_r=0
    agent=DQN()
    env = gym.make('Pendulum-v0')   
    agent.load()
    action=[0]
    for i in range(test_iteration):
        state = env.reset()  #状态  cos(theta), sin(theta) , thetadot角速度
        #state = agent.digitize_state(state)  #状态标准化
        for t in count():
            action_index,action[0]  = agent.act(state)
            next_state, reward, done, info = env.step(action)   
            #next_state = agent.digitize_state(next_state)
            ep_r += reward
            env.render()
            if done or t>=max_length_of_trajectory:
                print('Episode:{}, Return:{:0.2f}, Step:{}'.format(i,ep_r,t))
                ep_r = 0
                break
            state = next_state
            time.sleep(0.05)

def run_test():
    env = gym.make('Pendulum-v0')   
    action = [0]
    observation = env.reset()  #状态   
    # print(env.action_space)
    # print(observation)
    actions = np.linspace(-2, 2, 10)
    for t in range(100):   #
        # action[0] =  random.uniform(-2,2)   #力矩  -2到2 
        action[0] = 2
        observation, reward, done, info = env.step(action)   
        #print(action,reward,done)
       
        # print('observation:',observation)
        # print('theta:',env.state)
        env.render() 
        time.sleep(1)
    env.close()

if __name__ == '__main__':
    train()
    #test()
    #run_test()
    
