from n_puzzle import N_puzzle
from DQN import hidden_unit, Q_learning, RL_training

network = Q_learning(72, [250,250], 4, hidden_unit)
train = RL_training(3, 10, network, epsilon = 0.9, gamma = 0.9, lr = 1e-4)
train.train(10001,1000)
train.test()
train.optimizer.param_groups[0]['lr'] = 5e-5
train.epsilon = 0.5
train.train(10001,1000)
train.test()
train.optimizer.param_groups[0]['lr'] = 1e-5
train.epsilon = 0.5
train.train(10001,1000)
train.test()