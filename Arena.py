import tkinter as tk
import matplotlib.pyplot as plt
# top=tk.Tk()
# top.mainloop()
import networkx as nx
from GenerateMatrix import GenerateMatrix as gm
import string
import pylab
pylab.ion()
import random
from threading import Thread
# matrix_obj=gm(10,10)
# matrix_obj.prepare()
# A=matrix_obj.get_matrix()
# # print(A)

# G=nx.from_numpy_matrix(A)
# print(nx.node_link_data(G))
# nx.draw(G)
# plt.ion()
import time


class Arena():
    def __init__(self):        
        Thread.__init__(self)
        None

    def reset(self,matrix):
        # if matrix is None:
        # matrix=self.matrix
        self.G=nx.from_numpy_matrix(matrix)
        pylab.clf()
        print(matrix.shape)
        self.G=nx.create_empty_copy(self.G)
        self.pos=nx.random_layout(self.G)
        nx.draw_networkx_nodes(self.G,self.pos,edge_color='w')
        # plt.hold(False)
        # plt.ion()
        
    
    def add_edge(self,node1,node2):
        # edges=self.G.edges()
        # self.G.remove_edge(node1,node2)        
        self.G.add_edge(node1,node2,color='g',weight=5)
        # pos=nx.get_node_attributes(self.G,'pos')
        nx.draw_networkx_edges(self.G,self.pos)
        # self.G.draw()
        # self.G.updte
        # plt.clf()
        # nx.draw_random(self.G)
        # plt.show()

    def run(self):
        pylab.show()
        while(True):
            pylab.draw()
            plt.pause(2)
# if __name__=='__main__':
#     a=Arena()
#     a.reset()
#     pylab.show()
#     while(True):
#         node1=random.randint(0,5)
#         node2=random.randint(5,10)
#         a.add_edge(node1,node2)
#         # pylab.draw()
#         plt.pause(0.2)
    