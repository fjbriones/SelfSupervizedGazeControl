import sys
sys.path.append('../')

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import cv2
import os
import glob
from models.vaegan_models import encoder, build_encoder, generator
# from Thesis.losses import kl_loss
from keras.optimizers import Adam
from keras.models import load_model
from tensorflow.python.eager.context import eager_mode, graph_mode
import tensorflow as tf

def video_generator(
	video_dir="../data/AVDIAR/*/Video/*_CAM1.mp4", 
	frame_height=64, 
	frame_width=64, 
	frame_channels=3, 
	batch_size=1,
	camera_fps=10):
	while(True):
		# video_files = os.path.join(video_dir, '*.mp4')
		videos = glob.glob(video_dir)
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
				# # print(video)
				# cv2.imshow('frame', frame)
				# cv2.waitKey(10)
				if not ret:
					done = True
					break
				if frame_count % frame_sample == 0:
					yield frame, done

				frame_count = frame_count + 1
			yield None, done

class MovingScreenEnvFace(gym.Env):
	def __init__(self):
		#Action space is stay, left, up, right, down
		self.action_space = spaces.Discrete(5)
		self.observation_space = spaces.Box(low=0, high=1, shape = (64,64,3), dtype=np.uint8)
		self.curr_step = -1
		self.curr_episode = 0
		self.reward = 0

		self.video_gen = video_generator()
		self.frame, self.done = next(self.video_gen)
		

		self.Y_LIMIT = self.frame.shape[0]
		self.X_LIMIT = self.frame.shape[1]

		x_offset = 0
		y_offset = 0

		self.X_MIN_LOCATION = x_offset
		self.X_FOV = 80
		self.X_MAX_LOCATION = self.X_LIMIT - x_offset - self.X_FOV

		self.Y_MIN_LOCATION = y_offset
		self.Y_FOV = 80
		self.Y_MAX_LOCATION = self.Y_LIMIT - y_offset - self.Y_FOV


		self.pix_movement = 5

		# self.location = self.INIT_LOCATION
		self.x_location = self._init_location(self.X_MIN_LOCATION, self.X_MAX_LOCATION)
		self.y_location = self._init_location(self.Y_MIN_LOCATION, self.Y_MAX_LOCATION)


		# self.prev_ob = self._mod_frame(self.frame)
		
		self.cumul_r = 0

		# self.kl_ref = 0.04368501639357586
		# self.prev_ob = np.expand_dims(prev_obs, axis=0)
		# cv2.imshow('prev_ob', self.prev_ob)
		# cv2.waitKey(0)

		# with graph_mode():
		# 	assert not tf.executing_eagerly()
		# enc_comp = encoder(batch_size=1, time=10, latent_dim=32, frame_width=128, frame_height=128, frame_channels=1)
		# gen = generator(batch_size=1, time=10, latent_dim=32, frame_width=128, frame_height=128, frame_channels=1)
		# dis_vae = discriminator(batch_size=1, time=10, frame_width=128, frame_height=128, frame_channels=1)
		# dis_gan = discriminator(batch_size=1, time=10, frame_width=128, frame_height=128, frame_channels=1, name='vae')

		# encoder_train, generator_train, discriminator_train, vaegan = build_graph(enc_comp, gen, dis_gan, dis_vae, time=10, latent_dim=32, frame_width=128, frame_height=128, frame_channels=1)
		
		# gen.trainable=False
		# encoder_train.compile(Adam(), [kl_loss, 'mean_absolute_error'])
		# encoder_train.summary()
		# encoder_train.load_weights('models/human_box_2.h5')

		# enc = load_model('models/enc.011.h5')
		# enc.summary()

		# self.enc = encoder(time=1, latent_dim=8, frame_width=64, frame_height=64, frame_channels=1)

		# for i in range(len(enc.layers)):
		# 	self.enc.layers[i].set_weights(enc.layers[i].get_weights())
		# 	print("Loading weights at layer {:03d}".format(i))

		# self.enc.summary()
		enc = encoder(batch_size=1)
		enc.load_weights('../weights.old/vaegan.enc.100.h5')
		enc.summary()
		self.enc_rl = build_encoder(enc, batch_size=1)
		self.enc_rl.summary()
		# self.enc_rl= build_encoder(self.enc, time=1, frame_width=64, frame_height=64, frame_channels=1, latent_dim=8)
		
		# self.enc_rl.compile(Adam(), kl_loss)
		# self.enc_rl._make_predict_function()
		# self.enc_rl._make_test_function()
		self.edge = 0
		# self.z_p = np.empty(shape=(1, 1, 8 * 2))
		self.prev_r = 0
		self.action = 0


	def step(self, action):
		if self.done:
			raise RuntimeError("Episode is done")
		self.curr_step += 1

		x_movement, y_movement= self._get_movement(action)

		self.x_location += x_movement
		self.y_location += y_movement

		penalty = 0

		self.frame, self.done = next(self.video_gen)

		self.action = action
		# if self.edge>30:
		# 	# self.done = True
		# 	penalty = - 1.0

		if not self.done:
			self.curr_ob = self._mod_frame(self.frame)
			self.reward = self._generate_reward(self.curr_ob) + penalty
			self.cumul_r += self.reward
			self.prev_ob = self.curr_ob
		else:
			if self.edge>3:
				self.reward = penalty
			else:
				self.reward = 0

		return [self.curr_ob, np.array([self.x_location/self.X_MAX_LOCATION, self.y_location/self.Y_MAX_LOCATION])], self.reward, self.done, {}

	def _get_movement(self, action):
		x_movement = 0
		y_movement =0
		if action == 0:
			#stay
			x_movement = 0
			y_movement = 0
		elif action == 1:
			#left
			x_movement = -self.pix_movement
			if self.x_location <= self.X_MIN_LOCATION - x_movement:
				x_movement = 0
		elif action == 2:
			#up
			y_movement = -self.pix_movement
			if self.y_location <= self.Y_MIN_LOCATION - y_movement:
				y_movement = 0
		elif action == 3:
			#right
			x_movement = self.pix_movement
			if self.x_location >= self.X_MAX_LOCATION - x_movement:
				x_movement = 0
		else:
			#down
			y_movement = self.pix_movement
			if self.y_location >= self.Y_MAX_LOCATION - y_movement:
				y_movement = 0
		return x_movement, y_movement

	def _generate_reward(self, curr_ob):
		# with graph_mode():
		# kl = self.enc_rl.evaluate([np.expand_dims(np.expand_dims(prev_ob, axis=0), axis=1), np.expand_dims(np.expand_dims(curr_ob, axis=0), axis=1)], self.z_p, batch_size=1, verbose=0)
		# r = np.abs(kl - self.prev_r)
		# r = (self.prev_r - kl)
		# self.prev_r = kl
		# r = -np.log(kl)
		# r = -kl
		r = np.squeeze(self.enc_rl.predict_on_batch(np.expand_dims(curr_ob, axis=0)))/1000.
		# print(r)
		# print(r)
		# r = 1/kl
		return r

	def _init_location(self, min_loc, max_loc):
		# return int(((max_loc-min_loc)/2)/self.pix_movement)*self.pix_movement
		return int(np.random.randint(low=min_loc, high=max_loc+1)/self.pix_movement)*self.pix_movement

	def _mod_frame(self, frame):
		image = cv2.resize(frame[self.y_location:self.y_location+self.Y_FOV, self.x_location:self.x_location+self.X_FOV, :], (64, 64))
		# print(image.shape)
		# image = self._brighten_image(image)/255.
		image = image/127.5 - 1.
		# image = np.expand_dims(image, axis=0)
		return image

	# def _brighten_image(self, grey):
	# 	value = 50
	# 	grey_new = np.where((255 - grey) < value, 255, grey+value)
	# 	return grey_new

	def reset(self):
		self.video_gen = video_generator()
		# self.location = self.INIT_LOCATION
		self.x_location = self._init_location(self.X_MIN_LOCATION, self.X_MAX_LOCATION)
		self.y_location = self._init_location(self.Y_MIN_LOCATION, self.Y_MAX_LOCATION)
		self.curr_step = -1
		self.curr_episode += 1
		self.prev_r = 0
		self.cumul_r = 0
		self.frame, self.done = next(self.video_gen)
		self.curr_ob = self._mod_frame(self.frame)
		# print(self.prev_ob.shape)
		
		return [self.curr_ob, np.array([self.x_location/self.X_MAX_LOCATION, self.y_location/self.Y_MAX_LOCATION])]

	def render(self, mode='human', close=False):
		fov = cv2.resize(self.frame[self.y_location:self.y_location+self.Y_FOV, self.x_location:self.x_location+self.X_FOV, :], (64, 64))
		cv2.rectangle(self.frame, (self.x_location,self.y_location), (self.x_location+self.X_FOV, self.y_location+self.Y_FOV), (0,0,255), 2)
		# cv2.rectangle(self.frame, (self.X_MIN_LOCATION,self.Y_MIN_LOCATION), (self.X_MAX_LOCATION+self.X_FOV, self.Y_MAX_LOCATION + self.Y_FOV), (0,255,0), 1)
		cv2.putText(self.frame, "Reward: {:.04f}".format(self.reward), (0,self.Y_LIMIT-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 1, cv2.LINE_AA)
		cv2.putText(self.frame, "Action: {:01d}".format(self.action), (220,self.Y_LIMIT-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 1, cv2.LINE_AA)
		cv2.imshow('env', self.frame)

		
		cv2.imshow('fov', fov)

		cv2.waitKey(5)
		return self.frame, fov
