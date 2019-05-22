import sys
import random
import numpy as np
import csv

from tqdm import tqdm
from .agent import Agent
from random import random, randrange

from rl_utils.memory_buffer import MemoryBuffer
from rl_utils.networks import tfSummary
from rl_utils.stats import gather_stats

class DDQN:
    """ Deep Q-Learning Main Algorithm
    """

    def __init__(self, action_dim, state_dim, args, epsilon=0.9):
        """ Initialization
        """
        # Environment and DDQN parameters
        self.with_per = args.with_per
        self.action_dim = action_dim
        self.state_dim = state_dim
        #
        self.lr = 2.5e-4
        self.gamma = 0.95
        self.epsilon = epsilon
        self.epsilon_decay = 0.99
        self.buffer_size = 20000
        #
        if(len(state_dim) < 3):
            self.tau = 1e-2
        else:
            self.tau = 1.0
        # Create actor and critic networks
        self.agent = Agent(self.state_dim, action_dim, self.lr, self.tau, args.consecutive_frames, args.dueling)
        # Memory Buffer for Experience Replay
        self.buffer = MemoryBuffer(self.buffer_size, args.with_per)

    def policy_action(self, s):
        """ Apply an espilon-greedy policy to pick next action
        """
        if random() <= self.epsilon:
            return randrange(self.action_dim)
        else:
            return np.argmax(self.agent.predict(s)[0])

    def train_agent(self, batch_size):
        """ Train Q-network on batch sampled from the buffer
        """
        # Sample experience from memory buffer (optionally with PER)
        s, a, r, d, new_s, idx = self.buffer.sample_batch(batch_size)
        # print(s.shape)
        # Apply Bellman Equation on batch samples to train our DDQN

        # s = np.reshape(s, (-1, s.shape[2], s.shape[3], s.shape[4]))
        # new_s = np.reshape(new_s, (-1, new_s.shape[2], new_s.shape[3], new_s.shape[4]))
        # a = np.reshape(a, (-1, a.shape[2]))
        # print(a.shape)
        s1 = []
        s2 = []
        for i in range(s.shape[0]):
            s1.append(s[i,0])
            s2.append(s[i,1])

        s1_a = np.asarray(s1)
        s2_a = np.asarray(s2)

        s_e = [s1_a, s2_a]

        s1 = []
        s2 = []
        for i in range(new_s.shape[0]):
            s1.append(new_s[i,0])
            s2.append(new_s[i,1])

        s1_a = np.asarray(s1)
        s2_a = np.asarray(s2)

        new_s_e = [s1_a, s2_a]
        # s = np.squeeze(s)
        # print(s.shape)
        # s = np.reshape(s, (s[0], ))
        q = self.agent.predict(s_e)
        next_q = self.agent.predict(new_s_e)
        q_targ = self.agent.target_predict(new_s_e)

        # q = []
        # next_q = []
        # q_targ = []

        # for j in range(s.shape[0]):
        #     q.append(self.agent.predict(s[j]))
        #     next_q.append(self.agent.predict(new_s[j]))
        #     q_targ.append(self.agent.target_predict(new_s[j]))

        # q = np.asarray(q)
        # next_q = np.asarray(next_q)
        # q_targ = np.asarray(q_targ)

        # print(next_q.shape)
        # print(q_targ.shape)

        for i in range(s.shape[0]):
            old_q = q[i, a[i]]
            if d[i]:
                q[i, a[i]] = r[i]
            else:
                next_best_action = np.argmax(next_q[i,:])
                q[i, a[i]] = r[i] + self.gamma * q_targ[i, next_best_action]
            if(self.with_per):
                # Update PER Sum Tree
                self.buffer.update(idx[i], abs(old_q - q[i, a[i]]))
        # Train on batch
        self.agent.fit(s_e, q)
        # Decay epsilon
        self.epsilon *= self.epsilon_decay


    def train(self, env, args, summary_writer):
        """ Main DDQN Training Algorithm
        """

        results = []
        tqdm_e = tqdm(range(args.nb_episodes), desc='Score', leave=True, unit=" episodes")

        row = ['Episode', 'Cumul reward', 'Mean iou', 'Mean cov']
        with open(args.csv_save_path, 'w') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()

        # list_iou = []
        # list_cov = []
        # list_reward = []

        for e in tqdm_e:
            # Reset episode
            time, cumul_reward, done  = 0, 0, False
            old_state = env.reset()

            list_ep_iou = []
            list_ep_cov = []

            while not done:
                if args.render: env.render()
                # Actor picks an action (following the policy)
                a = self.policy_action(old_state)
                # Retrieve new state, reward, and whether the state is terminal
                new_state, r, done, iou, cov, _ = env.step(a)

                list_ep_iou.append(iou)
                list_ep_cov.append(cov)

                # Memorize for experience replay
                # print(old_state[0].shape)
                # print(new_state[0].shape)
                self.memorize(old_state, a, r, done, new_state)
                # Update current state
                old_state = new_state
                cumul_reward += r
                time += 1
                # Train DDQN and transfer weights to target network
                if(self.buffer.size() > args.batch_size):
                    self.train_agent(args.batch_size)
                    self.agent.transfer_weights()

            # Gather stats every episode for plotting
            if(args.gather_stats):
                mean, stdev = gather_stats(self, env)
                results.append([e, mean, stdev])

            # Export results for Tensorboard
            score = tfSummary('score', cumul_reward)
            summary_writer.add_summary(score, global_step=e)
            summary_writer.flush()

            # Display score
            tqdm_e.set_description("Score: " + str(cumul_reward))
            tqdm_e.refresh()

            #Write to csv
            mean_ep_iou = np.mean(np.asarray(list_ep_iou))
            mean_ep_cov = np.mean(np.asarray(list_ep_cov))
            row = [e, cumul_reward, mean_ep_iou, mean_ep_cov]
            with open(args.csv_save_path, 'a') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(row)
            csvFile.close()

            if e % 100 == 0:
                self.save_weights('DDQN/models/{:05d}'.format(e))

        return results

    def memorize(self, state, action, reward, done, new_state):
        """ Store experience in memory buffer
        """

        if(self.with_per):
            q_val = self.agent.predict(state)
            q_val_t = self.agent.target_predict(new_state)
            next_best_action = np.argmax(q_val)
            new_val = reward + self.gamma * q_val_t[0, next_best_action]
            td_error = abs(new_val - q_val)[0]
        else:
            td_error = 0
        self.buffer.memorize(state, action, reward, done, new_state, td_error)

    def save_weights(self, path):
        path += '_LR_{}'.format(self.lr)
        if(self.with_per):
            path += '_PER'
        self.agent.save(path)

    def load_weights(self, path):
        self.agent.load_weights(path)
