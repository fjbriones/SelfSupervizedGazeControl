import gym
import numpy as np
from collections import deque

class Environment(object):
    """ Environment Helper Class (Multiple State Buffer) for Continuous Action Environments
    (MountainCarContinuous-v0, LunarLanderContinuous-v2, etc..), and MujuCo Environments
    """
    def __init__(self, gym_env, action_repeat):
        self.env = gym_env
        self.timespan = action_repeat
        self.latent_dim = 8
        self.gym_actions = range(gym_env.action_space.n)
        # self.state1_buffer = deque()
        # self.state2_buffer = deque()
        # self.state3_buffer = deque()
        # self.rng = np.random.RandomState(0)

    def get_action_size(self):
        return self.env.action_space.n

    def get_state_size(self):
        return self.env.observation_space.shape

    def reset(self):
        """ Resets the game, clears the state buffer.
        """
        # Clear the state buffer
        # self.state1_buffer = deque()
        # self.state2_buffer = deque()
        # self.state3_buffer = deque()
        x_t = self.env.reset()

        # x_t1 = x_t[0]
        # x_t2 = x_t[1]
        # # x_t3 = x_t[2]
        # # x_t3 = self.rng.normal(size=(1, self.latent_dim))

        # s_t1 = np.stack([x_t1 for i in range(self.timespan)], axis=0)
        # for i in range(self.timespan-1):
        #     self.state1_buffer.append(x_t1)

        # s_t2 = np.stack([x_t2 for i in range(self.timespan)], axis=0)
        # for i in range(self.timespan-1):
        #     self.state2_buffer.append(x_t2)

        # # s_t3 = np.stack([x_t3 for i in range(self.timespan)], axis=0)
        # # for i in range(self.timespan-1):
        # #     self.state3_buffer.append(x_t3)

        # # s_t3 = np.stack([x_t3 for i in range(self.timespan)], axis=0)
        # # for i in range(self.timespan-1):
        # #     self.state3_buffer.append(x_t3)

        # s_t = [s_t1, s_t2]
        return x_t

    def step(self, action):
        x_t, r_t, terminal, iou, cov, info = self.env.step(action)

        # x_t1 = x_t[0]
        # x_t2 = x_t[1]
        # # x_t3 = x_t[2]
        # # x_t3 = self.rng.normal(size=(1, self.latent_dim))

        # # print(self.state_buffer)

        # # previous_states = np.array(self.state_buffer)
        # # print(previous_states.shape)

        # previous_states1 = np.array(self.state1_buffer)
        # previous_states2 = np.array(self.state2_buffer)
        # # previous_states3 = np.array(self.state3_buffer)

        # s_t1 = np.empty((self.timespan, *self.env.observation_space.shape))
        # s_t1[:self.timespan-1, :] = previous_states1
        # s_t1[self.timespan-1] = x_t1

        # s_t2 = np.empty((self.timespan, *(2,)))
        # s_t2[:self.timespan-1, :] = previous_states2# np.expand_dims(previous_states2, axis=1)
        # s_t2[self.timespan-1] = x_t2

        # # s_t3 = np.empty((self.timespan, *(1,)))
        # # s_t3[:self.timespan-1, :] = np.expand_dims(previous_states3, axis=1)
        # # s_t3[self.timespan-1] = x_t3


        # # s_t3 = np.empty((self.timespan, *(1, 1, 8)))
        # # s_t3[:self.timespan-1, :] = np.expand_dims(previous_states3, axis=1)
        # # s_t3[self.timespan-1] = x_t3
        # # Pop the oldest frame, add the current frame to the queue
        
        # self.state1_buffer.popleft()
        # self.state1_buffer.append(x_t1)

        # self.state2_buffer.popleft()
        # self.state2_buffer.append(x_t2)

        # self.state3_buffer.popleft()
        # self.state3_buffer.append(x_t3)

        # s_t = [s_t1, s_t2]

        return x_t, r_t, terminal, iou, cov, info

    def render(self):
        return self.env.render()
