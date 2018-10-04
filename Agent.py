import numpy as np
from GenerateMatrix import GenerateMatrix
np.set_printoptions(threshold=np.inf)
from Arena import Arena
from threading import Thread
class Agent():
    def __init__(self,size=5,max_size=5):
        self._size=size
        self._max_size=max_size 
        # self._arena=Arena()       
        
        # self._arena.start()
        # self._arena.run()
    def reset(self):
        self._state_visited=np.zeros(self._max_size)
        self._state_visited[self._size:]=-1
        self._current_vertex=0
        self._current_reward=0
        _matrix_generator=GenerateMatrix(self._size,self._max_size)
        _matrix_generator.prepare()
        self._matrix=_matrix_generator.get_matrix()
        self._state=-1+np.zeros((1,self._max_size))
        self._state[0,:self._size]=0
        
        # self._arena.reset(self._matrix[:self._size,:self._size])
        return np.asarray([self._state])

    def print_states(self,verbose=[True,False]):
        if(verbose[0]):
            print("=============Current State============")
            print(self._state)
        if(verbose[1]):
            print("================Matrix================")
            print(self._matrix)
    
    def _get_reward(self,action):
        None
        # if already visited state then -1000
        if self._state_visited[action]==1:
            return -800

        # if visiting unknown then -1000
        elif self._state_visited[action]==-1:
            return -1000

        # if visiting unvisited edge -ve  of distance
        else:
            return -self._matrix[self._current_vertex][action]

    def _check_if_done(self):
        sum=np.sum(self._state_visited[:self._size])
        if sum==self._size:
            return True
        else:
            return False

    def step(self,action): 
        reward=self._get_reward(action)
        # print(reward,action)
        # current_state=self._state               
        if(action<self._size):
            # self._arena.add_edge(self._current_vertex,action)
            self._current_vertex=action
            if(self._state_visited[action]==0):
                self._state=np.vstack((self._state,self._state_visited))
            self._state_visited[action]=1
            self._matrix[action,action]=-1
        done=self._check_if_done()
        return np.asarray([np.copy(self._state)]),reward,np.copy(self._matrix),done
    
    
# if __name__=='__main__':
#     a=Agent(size=10,max_size=10)
#     a.__init_states__()
#     a.print_states(verbose=[True,True])

    
    
