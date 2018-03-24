# Reinforcement-Learning-Q-learning-8puzzle-Pytorch
This is a project using neural-network reinforcement learning to solve the 8 puzzle problem (or even N puzzle)

If you want to know what is the 8 puzzle problem, please look at:
http://coursera.cs.princeton.edu/algs4/assignments/8puzzle.html

https://en.wikipedia.org/wiki/15_puzzle

If you need some background on the Q learning and deep Q learning, here is a good references:
http://outlace.com/rlpart3.html

If you want to know how to use Pytorch to do reinforcement learning, please refer to the Pytorch tutorial:
http://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

This project contains two files: `n_puzzle.py` is for constructing a 8 puzzle game, and `DQN.py` is used to construct and train the neural network. Since I use the Pytorch to construct the network, I also use the tensor of Pytorch for constructing the game. If you are only interested in the 8 puzzle game, please download the Pytorch before implementing the `n_puzzle.py`. Of cause, you can always change the Pytorch Tensor into numpy Array.


Tutorial:

First, import both files:

`from n_puzzle import N_puzzle`

`from DQN import hidden_unit, Q_learning, RL_training`

Then, construct the network and traning system:

`network = Q_learning(72, [250,250], 4, hidden_unit)`

Since the board of the 8 puzzle is represented by one hot-key coding, the dimension is 3x3x8 = 72. Here, the input will be input into 2 hidden layers both of which have 250 hidden units, and then output as vector of 4 elements. This network can also deal with replay experience. The input becomes `Batch x 72`, and the output is `Batch x 4`.

`train = RL_training(3, 10, network, epsilon = 0.9, gamma = 0.9, lr = 1e-4)`

The 3 and 10 represent the dimension and difficulty of the puzzles to be solved. Dimension of 3 equals to 8 puzzle. The fucntion of gamma, epsilon, and learning rate (lr) can be found at the reinforcement leanrning tutorial.

The last step is to train and test the DQN:

`train.train(10001,1000)`

`train.test()`

The 10001 and 1000 mean traning epoches and the output will be printed every 1000 epoches.

Given the above parameters and training, the network can achieve 75% success rate to solve a 8 puzzle in 20 steps. The difficulty (manhattan distance of the starting puzzle) is 10. If you give such a puzzle to a human and ask him/her to solve in 20 steps, it is likely that they would fail. After fine tuning and more traning, the success rate can reach 85%, which is smart enough compared to me.


Here is an example how the Q learning solves the 8 puzzle in a prefect way:
```
Initial State:

 1  8  0
 4  3  2
 7  6  5
[torch.IntTensor of size 3x3]


-33.4408 -42.0156 -44.3450 -41.6370
[torch.FloatTensor of size 1x4]

distance:  10
Move #: 0; Taking action: 0

 1  8  2
 4  3  0
 7  6  5
[torch.IntTensor of size 3x3]


-43.7496 -46.2655 -42.9442 -45.4692
[torch.FloatTensor of size 1x4]

distance:  9
Move #: 1; Taking action: 2

 1  8  2
 4  0  3
 7  6  5
[torch.IntTensor of size 3x3]


-31.5128 -17.3894 -37.5328 -28.5738
[torch.FloatTensor of size 1x4]

distance:  8
Move #: 2; Taking action: 1

 1  0  2
 4  8  3
 7  6  5
[torch.IntTensor of size 3x3]


-27.4439 -28.7143 -33.4939 -12.0024
[torch.FloatTensor of size 1x4]

distance:  7
Move #: 3; Taking action: 3

 1  2  0
 4  8  3
 7  6  5
[torch.IntTensor of size 3x3]


 -6.5040 -13.7535 -13.2515  -7.2390
[torch.FloatTensor of size 1x4]

distance:  6
Move #: 4; Taking action: 0

 1  2  3
 4  8  0
 7  6  5
[torch.IntTensor of size 3x3]


-2.5222 -8.6801 -4.6879 -6.2299
[torch.FloatTensor of size 1x4]

distance:  5
Move #: 5; Taking action: 0

 1  2  3
 4  8  5
 7  6  0
[torch.IntTensor of size 3x3]


-9.5404 -9.4426  1.5755 -9.6195
[torch.FloatTensor of size 1x4]

distance:  4
Move #: 6; Taking action: 2

 1  2  3
 4  8  5
 7  0  6
[torch.IntTensor of size 3x3]


-4.7309  5.7080 -3.8861 -1.9533
[torch.FloatTensor of size 1x4]

distance:  3
Move #: 7; Taking action: 1

 1  2  3
 4  0  5
 7  8  6
[torch.IntTensor of size 3x3]


 0.3997 -5.7986  1.2636  7.6686
[torch.FloatTensor of size 1x4]

distance:  2
Move #: 8; Taking action: 3

 1  2  3
 4  5  0
 7  8  6
[torch.IntTensor of size 3x3]


 9.7120  3.6771  6.5884  6.5222
[torch.FloatTensor of size 1x4]

distance:  1
Move #: 9; Taking action: 0

 1  2  3
 4  5  6
 7  8  0
[torch.IntTensor of size 3x3]

Reward: 10
```
