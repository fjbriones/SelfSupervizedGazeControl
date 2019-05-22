import sys
sys.path.append('/home/fjbriones/Projects')

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import cv2
import os
import glob
from Thesis.models import encoder, build_encoder
from Thesis.losses import kl_loss
from keras.optimizers import Adam
from tensorflow.python.eager.context import eager_mode, graph_mode
import tensorflow as tf

def video_generator(
	video_dir="data/kth_rl_test", 
	frame_height=64, 
	frame_width=64, 
	frame_channels=3, 
	batch_size=1,
	camera_fps=10):
	while(True):
		video_files = os.path.join(video_dir, '*.avi')
		videos = glob.glob(video_files)
		# print(videos)
		for video in videos:
			# print(video)
			video_key = os.path.basename(video).split(".")[0]
			cap = cv2.VideoCapture(video)
			# print(cap.get(cv2.CAP_PROP_FPS))
			
			video_fps = cap.get(cv2.CAP_PROP_FPS);
			# print(video_fps)
			frame_sample = int(video_fps/camera_fps)
			if (frame_sample) < 1:
				frame_sample = 1

			done = False
			frame_count = 0
			while(cap.isOpened()):
				ret, frame = cap.read()
				# print(video)
				if not ret:
					done = True
					break
				if frame_count % frame_sample == 0:
					yield frame, done

				frame_count = frame_count + 1
			yield None, done

class MovingScreenEnv(gym.Env):
	def __init__(self):
		#Action space is left, stay, or right
		self.action_space = spaces.Discrete(3)
		self.observation_space = spaces.Box(low=0, high=1, shape = (64,64,3), dtype=np.uint8)
		self.curr_step = -1
		self.curr_episode = 0
		self.reward = 0

		self.INIT_LOCATION = 50
		self.MAX_LOCATION = 100
		self.MIN_LOCATION = 0
		self.WIDTH_FOV = 60
		self.pix_movement = 4

		# self.location = self.INIT_LOCATION
		self.location = self._init_location()

		self.video_gen = video_generator()
		self.frame, self.done = next(self.video_gen)
		self.prev_ob = cv2.resize(self.frame[:, self.location:self.location+self.WIDTH_FOV, :], (64, 64))/255.

		self.cumul_r = 0

		# self.kl_ref = 0.04368501639357586
		# self.prev_ob = np.expand_dims(prev_obs, axis=0)
		# cv2.imshow('prev_ob', self.prev_ob)
		# cv2.waitKey(0)

		# with graph_mode():
		# 	assert not tf.executing_eagerly()
		self.enc = encoder(time=1, latent_dim=8)
		self.enc_rl = build_encoder(self.enc, time=1)
		self.enc_rl.compile(Adam(), kl_loss)
		self.enc_rl.load_weights('models/encoder.064.h5', by_name=True)
		self.enc_rl._make_predict_function()
		self.enc_rl._make_test_function()
		self.edge = 0
		self.z_p = np.empty(shape=(1, 1, 8 * 2))
		self.prev_r = 0


	def step(self, action):
		if self.done:
			raise RuntimeError("Episode is done")
		self.curr_step += 1

		movement, _, self.edge = self._get_movement(action, self.edge)
		self.location += movement

		penalty = 0

		self.frame, self.done = next(self.video_gen)

		# if self.edge>30:
		# 	# self.done = True
		# 	penalty = - 1.0

		if not self.done:
			self.curr_ob = cv2.resize(self.frame[:, self.location:self.location+self.WIDTH_FOV, :], (64, 64))/255.
			self.reward = self._generate_reward(self.curr_ob, self.prev_ob) + penalty
			self.cumul_r += self.reward
			self.prev_ob = self.curr_ob
		else:
			if self.edge>3:
				self.reward = penalty
			else:
				self.reward = 0

		return [self.curr_ob, self.location/self.MAX_LOCATION], self.reward, self.done, {}

	def _get_movement(self, action, edge):
		movement = 0
		done = False
		if action == 0:
			movement = -self.pix_movement
			if self.location == self.MIN_LOCATION:
				movement = 0
				done = True
				edge += 1
			else:
				edge = 0
		elif action == 1:
			movement = 0
			edge = 0
		else:
			movement = self.pix_movement
			if self.location == self.MAX_LOCATION:
				movement = 0
				done = True
				edge += 1
			else:
				edge = 0
		return movement, done, edge

	def _generate_reward(self, curr_ob, prev_ob):
		# with graph_mode():
		kl = self.enc_rl.evaluate([np.expand_dims(np.expand_dims(prev_ob, axis=0), axis=1), np.expand_dims(np.expand_dims(curr_ob, axis=0), axis=1)], self.z_p, batch_size=1, verbose=0)
		r = np.abs(kl - self.prev_r)
		# r = -(self.prev_r - kl)
		self.prev_r = kl
		# r = -kl
		return r

	def _init_location(self):
		return int(np.random.randint(low=self.MIN_LOCATION, high=self.MAX_LOCATION+1)/self.pix_movement)*self.pix_movement

	def reset(self):
		self.video_gen = video_generator()
		# self.location = self.INIT_LOCATION
		self.location = self._init_location()
		self.curr_step = -1
		self.curr_episode += 1
		self.prev_r = 0
		self.cumul_r = 0
		self.frame, self.done = next(self.video_gen)
		self.prev_ob = cv2.resize(self.frame[:, self.location:self.location+self.WIDTH_FOV, :], (64, 64))/255.
		
		return [self.prev_ob, self.location/self.MAX_LOCATION]

	def render(self, mode='human', close=False):
		cv2.rectangle(self.frame, (self.location,0), (self.location+self.WIDTH_FOV, 120), (0,0,255), 1)
		cv2.putText(self.frame, "Reward: {:.3f}".format(self.reward), (0,110), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255,0,0), 1, cv2.LINE_AA)
		cv2.imshow('env', self.frame)

		cv2.waitKey(5)
		return self.frame
