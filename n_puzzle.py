import torch
import numpy as np
import copy

class N_puzzle():
    def __init__(self,n, difficulty = 5):
        #the n is the dimension of the puzzle
        #the difficulty is the manhattan distance of the starting puzzle
        assert type(n) is int and n > 1
        self.n = n
        self.N = n**2
        #the reason to use torch.tensor here is that it is convenient to input the model
        board = torch.zeros((self.N,self.N - 1)).int()
        for i in range(self.N - 1):
            board[i,i] = 1
        self.board = board.view(n,n,-1)
        #the final is the goal to be achieved
        self.final = copy.deepcopy(self.board)
        #a dict to convert string into direction represented by int
        self.way_dict = {"up":0, "down":1, "left":2, "right":3}
        #convert the direction into vector
        self.direction_list = [(1,0),(-1,0),(0,-1),(0,1)]
        #zero_loc traces the position of the zero or empty
        self.zero_loc = np.array([n-1, n-1])
        #the mark is used to convert the one hot-key puzzle to the visible puzzle
        self.mark = torch.arange(self.N - 1)+1
        self.mark = self.mark.int().view(1,1,-1)
        #initilize the puzzle
        self.random_walk(difficulty)
        
    # the move will return reward based on the movement:
    # achieve the goal: 10, hit the wall:-1, otherwise: 0
    def move(self, way):
        if way in self.way_dict:
            way = self.way_dict[way]
        assert type(way) is int
        assert way >= 0 and way <=3, "please input 0 to 3"
        direction = self.direction_list[way]   
        new_loc = self.zero_loc + direction
#         if new_loc.any() >= self.n or new_loc.any() < 0:
#             print("wrong")
#             return -5
        if np.sum(new_loc >= self.n) or np.sum(new_loc< 0):
            return -1
        self.swap(self.zero_loc,new_loc)
        self.zero_loc = new_loc
        if self.achieve_final():
            return 10
        else:
            return 0
        
    def swap(self, old, new):
        tmp = self.board[old[0],old[1]].clone()
        self.board[old[0],old[1]] = self.board[new[0],new[1]]
        self.board[new[0],new[1]] = tmp
    
    # This also acts as a reward to guide the puzzle
    def manhattan(self):
        table = self.display()
        row = (table-1) / self.n
        col = (table-1) % self.n
        std = torch.arange(self.n).int()
        d_row = (row - std.view(self.n,1)).abs().sum()
        d_col = (col - std.view(1,self.n)).abs().sum()
        zero_distance = ((table == 0).nonzero().int() - torch.IntTensor([0,self.n - 1])).abs().sum()
        return d_row + d_col - zero_distance
        
    def achieve_final(self):
        return torch.equal(self.board, self.final)
    
    def display(self):
        table = self.board.int() * self.mark
        table = table.sum(dim = 2)
        return table       
            
    def random_walk(self, n_step):
        distance = 0
        while distance < n_step:
            step = np.random.randint(4)
            self.move(step)
            distance = self.manhattan()