import sys
sys.path.append('../')

from generator.savp_generator import video_generator, discriminator_loader, generator_loader, encoder_loader
from generator.savp_generator import discriminator_data, generator_data, encoder_data
from models.savp_models import trial_predictor, encoder, generator, build_graph
from models.savp_models import discriminator
from keras.models import Model
from keras.layers import Input
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.optimizers import RMSprop, Adam
from keras.utils import plot_model
import keras.backend as K
from utils.utils import count_images, count_frames, set_trainable
from losses import kl_loss
import argparse
import numpy as np
import math
import cv2
import os
import glob

def main(args):
	optimizer = Adam()

	enc = encoder(batch_size=args.batch_size, time=2*args.time-1, latent_dim=args.latent_dim, frame_height=args.frame_height, frame_width=args.frame_width, frame_channels=args.frame_channels)
	gen = generator(batch_size=args.batch_size, time=args.time, latent_dim=args.latent_dim, frame_height=args.frame_height, frame_width=args.frame_width, frame_channels=args.frame_channels)
	dis_gan = discriminator(batch_size=args.batch_size, time=2*args.time-1, name='gan', frame_height=args.frame_height, frame_width=args.frame_width, frame_channels=args.frame_channels)
	dis_vae = discriminator(batch_size=args.batch_size, time=2*args.time-1, name='vae', frame_height=args.frame_height, frame_width=args.frame_width, frame_channels=args.frame_channels)

	gen.compile(optimizer, ['mean_absolute_error'])

	print("\n\n\n-------------------------------- Encoder Summary --------------------------------\n")
	enc.summary()

	print("\n\n\n-------------------------------- Generator Summary --------------------------------\n")
	gen.summary()

	print("\n\n\n-------------------------------- Discriminator Summary --------------------------------\n")
	dis_gan.summary()

	plot_model(enc, to_file='../models/savp_encoder.png', show_shapes=True)
	plot_model(gen, to_file='../models/savp_generator.png', show_shapes=True)
	plot_model(dis_gan, to_file='../models/savp_discriminator.png', show_shapes=True)
	
	encoder_train, generator_train, discriminator_train, vaegan= build_graph(
		encoder=enc, 
		generator=gen, 
		discriminator_gan=dis_gan, 
		discriminator_vae=dis_vae,
		batch_size=args.batch_size, 
		time=args.time, 
		latent_dim=args.latent_dim, 
		frame_height=args.frame_height, 
		frame_width=args.frame_width, 
		frame_channels=args.frame_channels)

	set_trainable(enc, False)
	set_trainable(gen, False)
	discriminator_train.compile(optimizer, 
		['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], 
		loss_weights=[1., 1., 1., 1.])
	print("\n\n\n-------------------------------- Discriminator Train Summary --------------------------------\n")
	discriminator_train.summary()

	plot_model(discriminator_train, to_file='../models/savp_discriminator_train.png', show_shapes=True)

	set_trainable(dis_vae, False)
	set_trainable(dis_gan, False)
	set_trainable(gen, True)
	generator_train.compile(optimizer, 
		['binary_crossentropy', 'binary_crossentropy', 'mean_absolute_error'], 
		loss_weights=[1., 1., 1.])
	print("\n\n\n-------------------------------- Generator Train Summary --------------------------------\n")
	generator_train.summary()

	plot_model(generator_train, to_file='../models/savp_generator_train.png', show_shapes=True)

	set_trainable(gen, False)
	set_trainable(enc, True)
	encoder_train.compile(optimizer, 
		[kl_loss, 'mean_absolute_error'],
		loss_weights=[0.1, 100.])
	print("\n\n\n-------------------------------- Encoder Train Summary --------------------------------\n")
	encoder_train.summary()

	plot_model(encoder_train, to_file='../models/savp_encoder_train.png', show_shapes=True)

	#Determine the steps per epoch for training
	train_video_dir = os.path.join(args.train_directory, '*_act_14_*/*.jpg')
	steps_per_epoch = count_images(train_video_dir, batch_size=args.batch_size, time=args.time)

	print("Steps per epoch: {}".format(steps_per_epoch))

	#Determine the steps per epoch for evaluating
	val_video_dir = os.path.join(args.val_directory, '*_act_14_*/*.jpg')
	val_steps = count_images(val_video_dir, batch_size=args.batch_size, time=args.time)

	# #Determine the steps per epoch for training
	# train_video_dir = os.path.join(args.train_directory, '*.avi')
	# steps_per_epoch = count_frames(train_video_dir, batch_size=args.batch_size, time=args.time, camera_fps=args.camera_fps)

	# print("Steps per epoch: {}".format(steps_per_epoch))

	# #Determine the steps per epoch for evaluating
	# val_video_dir = os.path.join(args.val_directory, '*.avi')
	# val_steps = count_frames(val_video_dir, batch_size=args.batch_size, time=args.time, camera_fps=args.camera_fps)

	initial_epoch = args.load_epoch + 1

	l = 0

	if args.load_epoch>0:
		discriminator_train.load_weights('../weights/savp.discriminator.{:03d}.h5'.format(args.load_epoch))
		generator_train.load_weights('../weights/savp.generator.{:03d}.h5'.format(args.load_epoch))
		encoder_train.load_weights('../weights/savp.encoder.{:03d}.h5'.format(args.load_epoch))
		print('Loaded weights {:03d}'.format(args.load_epoch))
		l = args.load_epoch * steps_per_epoch * (args.time - 1)*args.batch_size

	for i in range(args.load_epoch + 1, args.epochs + 1):
		video_loader = video_generator(
			video_dir=args.train_directory,  
			frame_height=args.frame_height,
			frame_width=args.frame_width,
			frame_channels=args.frame_channels,
			batch_size = args.batch_size, 
			time = 2*args.time-1,
			camera_fps = args.camera_fps,
			json_filename="kinetics_train.json")

		dis_losses_avg = np.zeros((5))
		gen_losses_avg = np.zeros((4))
		enc_losses_avg = np.zeros((3))
		rec_losses_avg = 0

		seed = 0
		rng = np.random.RandomState(seed)

		print("\nEpoch {:03d} \n".format(i))

		for j in range(steps_per_epoch):
			
			x, y = next(video_loader)

			dis_inputs, dis_outputs = discriminator_data(x, y, latent_dim=args.latent_dim, seed=seed, time_init=args.time)
			gen_inputs, gen_outputs = generator_data(x, y, latent_dim=args.latent_dim, seed=seed, time_init=args.time)
			enc_inputs, enc_outputs = encoder_data(x, y, latent_dim=args.latent_dim, seed=seed, time_init=args.time)
			# print(enc_outputs[0].shape)

			# if j == 0:
				# previous_inputs = np.zeros_like(gen_outputs[2])
			# current_data = x[0]
			# output_data = y[0]
			# print(current_data.shape)
			# for m in range(args.time):
			# 	cv2.imshow("Input", current_data[m])
			# 	cv2.imshow("Output", output_data[m])
			# 	cv2.waitKey(25)
			# cv2.destroyAllWindows()

			# Use model inference
			z_p = rng.normal(size=(args.batch_size, args.time, args.latent_dim))
			gen_sample_input = [gen_inputs[0], gen_inputs[1], z_p]
			for k in range (args.time -1):
				l = l + 1*args.batch_size
				epsilon = -1./float(args.k) * float(l) + 1.
				if epsilon < np.random.random_sample():
					# print('\nUsing model inference at ', epsilon)
					gen_sample_output = gen.predict_on_batch(gen_sample_input)
					gen_inputs[3][:, k+args.time-1, :, :, :] = gen_sample_output[:, args.time - 1, :, :, :]
					gen_sample_input[1][:, 0:args.time-1, :, :, :] = gen_sample_input[1][:, 1:args.time, :, :, :]
					gen_sample_input[1][:, args.time-1, :, :, :] = gen_sample_output[:, args.time - 1, :, :, :]
					gen_sample_input[2] = rng.normal(size=(args.batch_size, args.time, args.latent_dim))					
				else:
					gen_sample_input[1][:, 0:args.time, :, :, :] = gen_inputs[3][:, k+1:args.time+k+1, :, :, :]
			
			dis_losses = discriminator_train.train_on_batch(dis_inputs, dis_outputs)
			gen_losses = generator_train.train_on_batch(gen_inputs, gen_outputs)
			enc_losses = encoder_train.train_on_batch(enc_inputs, enc_outputs)

			m = 0
			for loss in dis_losses:
				dis_losses_avg[m] = (dis_losses_avg[m] * j + loss)/(j + 1)
				m = m + 1

			m = 0
			for loss in gen_losses:
				gen_losses_avg[m] = (gen_losses_avg[m] * j + loss)/(j + 1)
				m = m + 1

			m = 0
			for loss in enc_losses:
				enc_losses_avg[m] = (enc_losses_avg[m] * j + loss)/(j + 1)
				m = m + 1

			print('\rEpoch {:03d} Step {:04d} of {:04d}. D_t_vae: {:.4f} D_t_gan: {:.4f} D_f_vae: {:.4f} D_f_gan: {:.4f} G_t_vae: {:.4f} G_t_gan: {:.4f} E_kl: {:.4f} Rec: {:.4f}'.format(i, j, steps_per_epoch, dis_losses_avg[1], dis_losses_avg[2], dis_losses_avg[3], dis_losses_avg[4], gen_losses_avg[1], gen_losses_avg[2], enc_losses_avg[1], enc_losses_avg[2]), end="", flush=True)

		if i%10 == 0 and i > 0:
			encoder_train.optimizer.lr = encoder_train.optimizer.lr/10
			discriminator_train.optimizer.lr = discriminator_train.optimizer.lr/10
			generator_train.optimizer.lr = generator_train.optimizer.lr/10
			print("\nCurrent learning rate is now: {}".format(K.eval(encoder_train.optimizer.lr)))

		print('\nSaving models')
		discriminator_train.save_weights('../weights/savp.discriminator.{:03d}.h5'.format(i))
		generator_train.save_weights('../weights/savp.generator.{:03d}.h5'.format(i))
		encoder_train.save_weights('../weights/savp.encoder.{:03d}.h5'.format(i))
		enc.save('../weights/savp.enc.{:03d}.h5'.format(i))

		dis_losses_avg = np.zeros((5))
		gen_losses_avg = np.zeros((4))
		enc_losses_avg = np.zeros((3))
		rec_losses_avg = 0

		seed = 0
		rng = np.random.RandomState(seed)

		video_loader_val = video_generator(
			video_dir = args.val_directory, 
			frame_height=args.frame_height,
			frame_width=args.frame_width,
			frame_channels=args.frame_channels,
			batch_size = args.batch_size, 
			time = 2*args.time-1,
			camera_fps = args.camera_fps,
			json_filename = "kinetics_val.json")

		print('Evaluating the model')
		for j in range(val_steps):
			
			x, y = next(video_loader_val)

			dis_inputs, dis_outputs = discriminator_data(x, y, latent_dim=args.latent_dim, seed=seed, time_init=args.time)
			gen_inputs, gen_outputs = generator_data(x, y, latent_dim=args.latent_dim, seed=seed, time_init=args.time)
			enc_inputs, enc_outputs = encoder_data(x, y, latent_dim=args.latent_dim, seed=seed, time_init=args.time)

			dis_losses = discriminator_train.test_on_batch(dis_inputs, dis_outputs)
			gen_losses = generator_train.test_on_batch(gen_inputs, gen_outputs)
			enc_losses = encoder_train.test_on_batch(enc_inputs, enc_outputs)

			m = 0
			for loss in dis_losses:
				dis_losses_avg[m] = (dis_losses_avg[m] * j + loss)/(j + 1)
				m = m + 1

			m = 0
			for loss in gen_losses:
				gen_losses_avg[m] = (gen_losses_avg[m] * j + loss)/(j + 1)
				m = m + 1

			m = 0
			for loss in enc_losses:
				enc_losses_avg[m] = (enc_losses_avg[m] * j + loss)/(j + 1)
				m = m + 1

			print('\rEpoch {:03d} Evaluation Step {:04d} of {:04d}. D_t_vae: {:.4f} D_t_gan: {:.4f} D_f_vae: {:.4f} D_f_gan: {:.4f} G_t_vae: {:.4f} G_t_gan: {:.4f} E_kl: {:.4f} Rec: {:.4f}'.format(i, j, val_steps, dis_losses_avg[1], dis_losses_avg[2], dis_losses_avg[3], dis_losses_avg[4], gen_losses_avg[1], gen_losses_avg[2], enc_losses_avg[1], enc_losses_avg[2]), end="", flush=True)

		# video_loader_val_dis = video_generator(
		# 	video_dir = args.val_directory, 
		# 	frame_height=args.frame_height,
		# 	frame_width=args.frame_width,
		# 	frame_channels=args.frame_channels,
		# 	batch_size = args.batch_size, 
		# 	time = 2*args.time-1,
		# 	camera_fps = args.camera_fps,
		# 	json_filename = "kinetics_val.json")

		# video_loader_val_gen = video_generator(
		# 	video_dir = args.val_directory,  
		# 	frame_height=args.frame_height,
		# 	frame_width=args.frame_width,
		# 	frame_channels=args.frame_channels,
		# 	batch_size=args.batch_size, 
		# 	time = 2*args.time-1,
		# 	camera_fps = args.camera_fps,
		# 	json_filename = "kinetics_val.json")

		# video_loader_val_enc = video_generator(
		# 	video_dir = args.val_directory,  
		# 	frame_height=args.frame_height,
		# 	frame_width=args.frame_width,
		# 	frame_channels=args.frame_channels,
		# 	batch_size = args.batch_size, 
		# 	time = 2*args.time-1,
		# 	camera_fps = args.camera_fps,
		# 	json_filename = "kinetics_val.json")

		# dis_loader_val = discriminator_loader(video_loader_val_dis, latent_dim=args.latent_dim, seed=seed, time_init=2*args.time-1)
		# gen_loader_val = generator_loader(video_loader_val_gen, latent_dim=args.latent_dim, seed=seed, eval=True, time_init=2*args.time-1)
		# enc_loader_val = encoder_loader(video_loader_val_enc, latent_dim=args.latent_dim, seed=seed, time_init=2*args.time-1)

		# print('\nEvaluating the model')
		# dis_losses_val = discriminator_train.evaluate_generator(dis_loader_val, steps=val_steps)
		# print('Discriminator: ')
		# print(discriminator_train.metrics_names)
		# print(dis_losses_val)

		# gen_losses_val = generator_train.evaluate_generator(gen_loader_val, steps=val_steps)
		# print('Generator: ')
		# print(generator_train.metrics_names)
		# print(gen_losses_val)

		# enc_losses_val = encoder_train.evaluate_generator(enc_loader_val, steps=val_steps)
		# print('Encoder: ')
		# print(encoder_train.metrics_names)
		# print(enc_losses_val)
		
def main_trial(args):
	train_gen = video_generator(args.train_directory, args.batch_size, args.time)
	val_gen = video_generator(args.val_directory, args.batch_size, args.time)

	inputs = Input(shape=(args.time, args.frame_height, args.frame_width, args.frame_channels))
	outputs = trial_predictor(inputs, args.time)

	checkpoint = ModelCheckpoint(filepath='models/weights.{epoch:02d}.h5')
	scheduler = LearningRateScheduler(lr_schedule, verbose=1)

	model = Model(inputs=inputs, outputs=outputs)
	model.compile(optimizer='rmsprop',
		loss='mean_squared_error',
		metrics=['accuracy'])
	model.summary()

	if args.load_epoch > 0:
		model.load_weights('models/weights.{:02d}.h5'.format(args.load_epoch))

	model.fit_generator(train_gen, 
		steps_per_epoch=797, 
		epochs=args.epochs, 
		validation_data=val_gen, 
		validation_steps=47,
		callbacks=[checkpoint, scheduler],
		initial_epoch=args.load_epoch)

	model.save('models/final_weights.h5')

if __name__=="__main__":
	parser = argparse.ArgumentParser(description='Train the network')
	parser.add_argument('-t', '--train_directory', type=str, default='../data/Human3.6M/train')
	parser.add_argument('-v', '--val_directory', type=str, default='../data/Human3.6M/val')
	parser.add_argument('-b', '--batch_size', type=int, default=1)
	parser.add_argument('-e', '--epochs', type=int, default=50)
	parser.add_argument('-i', '--time', type=int, default=5)
	parser.add_argument('-l', '--latent_dim', type=int, default=8)
	parser.add_argument('-r', '--trial', action='store_true')
	parser.add_argument('-le', '--load_epoch', type=int, default=0)
	parser.add_argument('-k', type=float, default=500000.)
	parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
	parser.add_argument('-c', '--camera_fps', type=int, default=10)
	parser.add_argument('-fh', '--frame_height', type=int, default=64)
	parser.add_argument('-fw', '--frame_width', type=int, default=64)
	parser.add_argument('-fc', '--frame_channels', type=int, default=3)
	args = parser.parse_args()

	if args.trial:
		main_trial(args)
	else:
		main(args)
	

