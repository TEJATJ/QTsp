import numpy as np

class GenerateMatrix():
    def __init__(self,size=20,max_size=50):
        self.size=size
        self.max_size=max_size

    # prepare a random distance matrix with the give size
    def prepare(self):
        matrix=np.random.randint(1,100,size=(self.size,self.size))
        matrix=(matrix+matrix.T)/2
        matrix=matrix-(matrix*np.eye(self.size))
        arena=-1+np.zeros((self.max_size,self.max_size))
        arena[0:self.size,0:self.size]=matrix
        self._matrix=arena
    # getter function for the matrix
    def get_matrix(self):
        return self._matrix

# if __name__=='__main__':
#     m=GenerateMatrix(5)
#     m.prepare()
#     print(m.get_matrix())