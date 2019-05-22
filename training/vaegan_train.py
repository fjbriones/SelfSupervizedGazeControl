import sys
sys.path.append('../')

from generator.vaegan_generator import celeba_image_generator, humanm_image_generator, discriminator_data, generator_data, encoder_data
from models.vaegan_models import discriminator, generator, encoder, build_vaegan_graph
from keras.optimizers import Adam, RMSprop
from keras.utils import plot_model
from training.losses import kl_loss, mean_gaussian_negative_log_likelihood
from utils.utils import count_celeba_data, count_humanm_data
import numpy as np
import keras.backend as K
import argparse
import cv2

def train(args):
	# optimizer = Adam(lr=args.learning_rate)
	optimizer = RMSprop(lr=args.learning_rate)
	image_dimension = (args.image_height, args.image_width, args.image_channels)
	rec_loss = mean_gaussian_negative_log_likelihood

	enc = encoder(batch_size=args.batch_size,
		latent_dimension=args.latent_dimension,
		image_dimension=image_dimension)

	gen = generator(batch_size=args.batch_size,
		latent_dimension=args.latent_dimension)

	dis = discriminator(batch_size=args.batch_size,
		image_dimension=image_dimension)

	print("\n\n\n-------------------------------- Encoder Summary --------------------------------\n")
	enc.summary()

	print("\n\n\n-------------------------------- Generator Summary --------------------------------\n")
	gen.summary()

	print("\n\n\n-------------------------------- Discriminator Summary --------------------------------\n")
	dis.summary()

	plot_model(enc, to_file='../models/vaegan_encoder.png', show_shapes=True)
	plot_model(gen, to_file='../models/vaegan_generator.png', show_shapes=True)
	plot_model(dis, to_file='../models/vaegan_discriminator.png', show_shapes=True)

	encoder_train, generator_train, discriminator_train = build_vaegan_graph(
		encoder=enc,
		generator=gen,
		discriminator=dis,
		batch_size=args.batch_size,
		latent_dimension=args.latent_dimension,
		image_dimension=image_dimension)

	enc.trainable = False
	gen.trainable = False
	dis.trainable = True
	discriminator_train.compile(optimizer,
		['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'],
		loss_weights=[1., 1., 1.])
	print("\n\n\n-------------------------------- Discriminator Train Summary --------------------------------\n")
	discriminator_train.summary()
	plot_model(discriminator_train, to_file='../models/vaegan_discriminator_train.png', show_shapes=True)

	enc.trainable = False
	gen.trainable = True
	dis.trainable = False
	generator_train.compile(optimizer, 
		['binary_crossentropy', 'binary_crossentropy'], 
		loss_weights=[1., 1.])
	print("\n\n\n-------------------------------- Generator Train Summary --------------------------------\n")
	generator_train.summary()
	plot_model(generator_train, to_file='../models/vaegan_generator_train.png', show_shapes=True)

	enc.trainable = True
	gen.trainable = False
	dis.trainable = False
	encoder_train.compile(optimizer)#, 
		# [kl_loss],
		# loss_weights=[1.])
	print("\n\n\n-------------------------------- Encoder Train Summary --------------------------------\n")
	encoder_train.summary()
	plot_model(encoder_train, to_file='../models/encoder_train.png', show_shapes=True)

	

	initial_epoch = args.load_epoch + 1

	if args.load_epoch>0:
		discriminator_train.load_weights('../weights/vaegan.discriminator.{:03d}.h5'.format(args.load_epoch))
		# generator_train.load_weights('../weights/vaegan.generator.{:03d}.h5'.format(args.load_epoch))
		# encoder_train.load_weights('../weights/vaegan.encoder.{:03d}.h5'.format(args.load_epoch))
		gen.load_weights('../weights/vaegan.gen.{:03d}.h5'.format(args.load_epoch))
		enc.load_weights('../weights/vaegan.enc.{:03d}.h5'.format(args.load_epoch))
		print('Loaded weights {:03d}'.format(args.load_epoch))

	seed = 0
	rng = np.random.RandomState(seed)

	# images_list_train = images_list()
	# images_list_val = images_list(mode=1)

	if (args.dataset == 'celeba'):
		#Count training data
		train_steps = count_celeba_data(mode=0, batch_size=args.batch_size)
		val_steps = count_celeba_data(mode=1, batch_size=args.batch_size)
		#loaders
		images_loader_train = celeba_image_generator(mode=0, rng=rng, batch_size=args.batch_size)
		images_loader_val = celeba_image_generator(mode=1, rng=rng, batch_size=args.batch_size)
	elif(args.dataset == 'humanm'):
		#Count training data
		train_steps = count_humanm_data(video_dir='../data/Human3.6M/train', batch_size=args.batch_size)
		val_steps = count_humanm_data(video_dir='../data/Human3.6M/val', batch_size=args.batch_size)
		# print(train_steps)
		#loaders
		images_loader_train = humanm_image_generator(video_dir='../data/Human3.6M/train', batch_size=args.batch_size)
		images_loader_val = humanm_image_generator(video_dir='../data/Human3.6M/val', batch_size=args.batch_size)


	
	for i in range(args.load_epoch+1, args.epoch+1):
		enc_losses_avg = 0#np.zeros((3))
		gen_losses_avg = np.zeros((3))
		dis_losses_avg = np.zeros((4))

		# print("\nEpoch {:03d}".format(i))

		for j in range(train_steps):
			images_batch = next(images_loader_train)

			enc_inputs, enc_outputs = encoder_data(
				images_batch=images_batch,
				rng=rng,
				latent_dimension=args.latent_dimension)

			gen_inputs, gen_outputs = generator_data(
				images_batch=images_batch,
				rng=rng,
				latent_dimension=args.latent_dimension)

			dis_inputs, dis_outputs = discriminator_data(
				images_batch=images_batch,
				rng=rng,
				latent_dimension=args.latent_dimension)

			dis_losses = discriminator_train.train_on_batch(dis_inputs, dis_outputs)
			gen_losses = generator_train.train_on_batch(gen_inputs, gen_outputs)
			enc_losses = encoder_train.train_on_batch(enc_inputs, enc_outputs)

			# print(dis_losses)
			# print(gen_losses)
			# print(enc_losses)
			
			m = 0
			for loss in dis_losses:
				dis_losses_avg[m] = (dis_losses_avg[m] * j + loss)/(j + 1)
				m = m + 1

			m = 0
			for loss in gen_losses:
				gen_losses_avg[m] = (gen_losses_avg[m] * j + loss)/(j + 1)
				m = m + 1

			enc_losses_avg = (enc_losses_avg * j + enc_losses)/(j + 1)
			# m = 0
			# for loss in enc_losses:
				# enc_losses_avg[m] = (enc_losses_avg[m] * j + loss)/(j + 1)
				# m = m + 1

			print('\rEpoch {:03d} Step {:04d} of {:04d}. D_real: {:.4f} D_z_vae: {:.4f} D_z_p {:.4f} G_z_vae_gan: {:.4f} G_z_p_gan: {:.4f} Dis_loss: {:.4f} Gen_loss: {:.4f} Enc_loss: {:.4f}'.format(i, j, train_steps, dis_losses_avg[1], dis_losses_avg[2], dis_losses_avg[3], gen_losses_avg[1], gen_losses_avg[2], dis_losses_avg[0], gen_losses_avg[0], enc_losses_avg), end="", flush=True)

		encoder_train.optimizer.lr = encoder_train.optimizer.lr * 0.95
		print('\nLearning rate is now {:.6f}'.format(K.eval(encoder_train.optimizer.lr)))

		print('Saving models')
		discriminator_train.save_weights('../weights/vaegan.discriminator.{:03d}.h5'.format(i))
		generator_train.save_weights('../weights/vaegan.generator.{:03d}.h5'.format(i))
		encoder_train.save_weights('../weights/vaegan.encoder.{:03d}.h5'.format(i))
		enc.save('../weights/vaegan.enc.{:03d}.h5'.format(i))
		gen.save('../weights/vaegan.gen.{:03d}.h5'.format(i))

		enc_losses_avg = 0#np.zeros_like(enc_losses_avg)
		gen_losses_avg = np.zeros_like(gen_losses_avg)
		dis_losses_avg = np.zeros_like(dis_losses_avg)

		z = rng.normal(size=(args.batch_size, args.latent_dimension))
		generator_output = ((np.squeeze(gen.predict_on_batch(z)) + 1.0)*127.5).astype(np.uint8)
		display_tiles = int(np.sqrt(args.batch_size))
		display_box = np.zeros((display_tiles*generator_output.shape[1], display_tiles*generator_output.shape[2], generator_output.shape[3]))
		for m in range(display_tiles):
			for n in range(display_tiles):
				# cv2.imshow('result', generator_output[i*display_tiles+j])
				# cv2.waitKey(0)
				# cv2.destroyAllWindows()
				display_box[m*generator_output.shape[1]:(m+1)*generator_output.shape[1], n*generator_output.shape[2]:(n+1)*generator_output.shape[2],:] = generator_output[m*display_tiles+n]

		# cv2.imshow('Result {:03d}'.format(i), display_box)
		# cv2.waitKey(5)
		cv2.imwrite('../results/vaegan/epoch.{:03d}.jpg'.format(i), display_box)
		
		print('Evaluating the model')
		for j in range(val_steps):
			images_batch = next(images_loader_val)

			enc_inputs, enc_outputs = encoder_data(
				images_batch=images_batch,
				rng=rng,
				latent_dimension=args.latent_dimension)

			gen_inputs, gen_outputs = generator_data(
				images_batch=images_batch,
				rng=rng,
				latent_dimension=args.latent_dimension)

			dis_inputs, dis_outputs = discriminator_data(
				images_batch=images_batch,
				rng=rng,
				latent_dimension=args.latent_dimension)

			enc_losses = encoder_train.test_on_batch(enc_inputs, enc_outputs)
			gen_losses = generator_train.test_on_batch(gen_inputs, gen_outputs)
			dis_losses = discriminator_train.test_on_batch(dis_inputs, dis_outputs)

			m = 0
			for loss in dis_losses:
				dis_losses_avg[m] = (dis_losses_avg[m] * j + loss)/(j + 1)
				m = m + 1

			m = 0
			for loss in gen_losses:
				gen_losses_avg[m] = (gen_losses_avg[m] * j + loss)/(j + 1)
				m = m + 1

			enc_losses_avg = (enc_losses_avg * j + enc_losses)/(j + 1)
			# m = 0
			# for loss in enc_losses:
			# 	enc_losses_avg[m] = (enc_losses_avg[m] * j + loss)/(j + 1)
			# 	m = m + 1

			print('\rEpoch {:03d} Eval Step {:04d} of {:04d}. D_real: {:.4f} D_z_vae: {:.4f} D_z_p {:.4f} G_z_vae_gan: {:.4f} G_z_p_gan: {:.4f} Dis_loss: {:.4f} Gen_loss: {:.4f} Enc_loss: {:.4f}'.format(i, j, val_steps, dis_losses_avg[1], dis_losses_avg[2], dis_losses_avg[3], gen_losses_avg[1], gen_losses_avg[2], dis_losses_avg[0], gen_losses_avg[0], enc_losses_avg), end="", flush=True)
		print('\n')


if __name__=="__main__":
	parser = argparse.ArgumentParser(description='Train the vaegan network')
	parser.add_argument('-b', '--batch_size', type=int, default=64)
	parser.add_argument('-l', '--latent_dimension', type=int, default=128)
	parser.add_argument('-lr', '--learning_rate', type=float, default=0.0003)
	parser.add_argument('-fh', '--image_height', type=int, default=64)
	parser.add_argument('-fw', '--image_width', type=int, default=64)
	parser.add_argument('-fc', '--image_channels', type=int, default=3)
	parser.add_argument('-e', '--epoch', type=int, default=1000)
	parser.add_argument('-le', '--load_epoch', type=int, default=0)
	parser.add_argument('-d', '--dataset', choices=['celeba', 'humanm'], default='humanm')
	args = parser.parse_args()

	train(args)
