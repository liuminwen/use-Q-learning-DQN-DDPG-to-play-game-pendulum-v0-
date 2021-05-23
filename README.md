# use-Q-learning-DQN-DDPG-to-play-game-pendulum-v0-
使用了DDPG、DQN、Q-learning来玩gym游戏“pendulum-v0”，使用python版本3.7+，pytorch版本1.8.1
## DDPG
代码参考了https://www.jianshu.com/p/a8608c98adc0， 修改了其中出错的地方，大概在几千个episode之后可以得到较好结果。参数的设置在文件的开头，directory为模型参数储存的地址，训练模型是设置MODE=‘train’，测试训练后的模型结果则设置MODE='test',一些其它重要参数的说明：MAX_EPISODE：要训练的episode数量，max_length_of_trajectory：每个episode训练的最大step数量，exploration_noise：对于action加入的噪声，TAU：更新target网络时的tau，update_iteration：每个episode更新网络的次数
## DQN
因为DQN不能处理连续的动作空间，所以将动作空间离散化，我选择将【-2，2】的动作等距离散为10个动作，想要修改离散后的动作个数可以在DQN类的初始参数设置里更改self.num_actions。运行train()训练模型,运行test()测试训练出的模型，directory为模型储存的地址。
## Q-learning
Q-learning将状态和动作都离散化了，想要更改离散后的参数个数可以修改QLeaning类的初始参数设置。
