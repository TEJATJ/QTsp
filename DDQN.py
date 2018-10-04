# -*- coding: utf-8 -*-
import random
# import gym
import numpy as np
from collections import deque
from keras.models import Sequential,Model
from keras.layers import Dense,LSTM,Conv1D,MaxPool1D,Flatten,BatchNormalization,concatenate,Input,Dropout
from keras.optimizers import Adam
from keras import backend as K
from Agent import Agent
import tensorflow as tf
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
EPISODES = 5000

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.3
        self.epsilon_decay = 0.9998
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    """Huber loss for Q Learning
    References: https://en.wikipedia.org/wiki/Huber_loss
                https://www.tensorflow.org/api_docs/python/tf/losses/huber_loss
    """

    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond  = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        input_lstm=Input(batch_shape=(None,None,self.state_size))
        lstm=LSTM(80,return_sequences=True)(input_lstm)
        lstm=LSTM(40,return_sequences=False)(lstm)
        input_conv=Input(batch_shape=(None,self.state_size,self.state_size))
        conv=Conv1D(400,3)(input_conv)
        conv=MaxPool1D(2)(conv)
        conv=BatchNormalization()(conv)
        conv=Conv1D(300,2)(conv)
        conv=MaxPool1D(2)(conv)
        conv=BatchNormalization()(conv)
        conv=Conv1D(100,2)(conv)
        conv=MaxPool1D(2)(conv)
        conv=BatchNormalization()(conv)
        conv=Flatten()(conv)
        merge=concatenate([lstm,conv],axis=1)
        merge=Dense(200)(merge)
        merge=Dropout(0.5)(merge)
        merge=Dense(80)(merge)
        merge=Dropout(0.5)(merge)
        merge=Dense(self.action_size)(merge)

        model=Model([input_lstm,input_conv],outputs=merge)
        # model.summary()
        model.compile(Adam(self.learning_rate),loss=self._huber_loss)  
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state,matrix, done):
        self.memory.append((state, action, reward, next_state,matrix, done))

    def act(self, state,matrix):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict([state,matrix])
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state,matrix, done in minibatch:
            target = self.model.predict([state,matrix])
            if done:
                target[0][action] = reward
            else:
                # a = self.model.predict([next_state,matrix])[0]
                t = self.target_model.predict([next_state,matrix])[0]
                target[0][action] = reward + self.gamma * np.amax(t)
                # target[0][action] = reward + self.gamma * t[np.argmax(a)]
            self.model.fit([state,matrix], target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon*=self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    # env = gym.make('CartPole-v1')
    # state_size = env.observation_space.shape[0]
    # action_size = env.action_space.n
    state_size=action_size=20   
   
    agent = DQNAgent(state_size, action_size)
    # agent.load("./save/cartpole-ddqn.h5")
    done = False
    batch_size = 20

    for e in range(EPISODES):
        size=random.randint(6,action_size)
        print("Episode {} : Nodes : {} , Maximum Nodes : {} ".format(e,size,action_size))
        env=Agent(size=size,max_size=action_size)
        state = env.reset()
        # state = np.reshape(state, [1, state_size])
        matrix=np.asarray([env._matrix])
        global_reward=0
        time=0
        done=False
        for time in range(500):
            # time+=1
            # env.render()
            action = agent.act(state,matrix)
            next_state, reward,matrix, done = env.step(action)
            matrix=np.asarray([matrix])
            reward = reward if not done else 0
            global_reward+=reward
            # print(reward)
            # next_state = np.asarray([next_state])
            agent.remember(state, action, reward, next_state,matrix, done)
            state = next_state
            # print(reward,done)
            # print(env._state_visited[:state_size])
            

            if done:
                print("GameDone in {}: updating the model".format(time))
                agent.update_target_model()
                print("episode: {}/{}, score: {},reward:{}, e: {:.2}"
                      .format(e, EPISODES, time,global_reward, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        # if e % 10 == 0:
        #     agent.save("./save/cartpole-ddqn.h5")