# from keras.layers import Dense,LSTM,Conv1D,MaxPool1D,Flatten,BatchNormalization,concatenate,Input
# from keras.models import Sequential,Model
# input_lstm=Input(batch_shape=(None,None,10))
# lstm=LSTM(30,return_sequences=True)(input_lstm)
# lstm=LSTM(10,return_sequences=False)(lstm)
# input_conv=Input(batch_shape=(None,10,10))
# conv=Conv1D(100,6)(input_conv)
# conv=MaxPool1D(2)(conv)
# conv=BatchNormalization()(conv)
# conv=Conv1D(50,4)(input_conv)
# conv=MaxPool1D(2)(conv)
# conv=BatchNormalization()(conv)
# conv=Flatten()(conv)
# merge=concatenate([lstm,conv],axis=1)
# merge=Dense(150)(merge)
# model=Model([input_lstm,input_conv],outputs=merge)
# model.summary()
import random
import pylab
from matplotlib.pyplot import pause
import networkx as nx
pylab.ion()

graph = nx.Graph()
node_number = 0
graph.add_node(node_number, Position=(random.randrange(0, 100), random.randrange(0, 100)))

def get_fig():
    global node_number
    node_number += 1
    graph.add_node(node_number, Position=(random.randrange(0, 100), random.randrange(0, 100)))
    graph.add_edge(node_number, node_number-1)
    nx.draw(graph, pos=nx.get_node_attributes(graph,'Position'))

num_plots = 50;
pylab.show()

for i in range(num_plots):

    get_fig()
    pylab.draw()
    pause(2)