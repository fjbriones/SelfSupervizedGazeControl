import sys
sys.path.append('../')

from generator.vaegan_generator import image_generator, discriminator_data, generator_data, encoder_data
from models.vaegan_models import discriminator, generator, encoder, build_vaegan_graph
from keras.optimizers import Adam, RMSprop
from keras.utils import plot_model
from losses import kl_loss, mean_gaussian_negative_log_likelihood
from utils.utils import count_celeba_data
import numpy as np
import argparse

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

	plot_model(enc, to_file='../models/vaegan_encoder.png')
	plot_model(gen, to_file='../models/vaegan_generator.png')
	plot_model(dis, to_file='../models/vaegan_discriminator.png')

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
	plot_model(discriminator_train, to_file='../models/vaegan_discriminator_train.png')

	enc.trainable = False
	gen.trainable = True
	dis.trainable = False
	generator_train.compile(optimizer, 
		['binary_crossentropy', 'binary_crossentropy'], 
		loss_weights=[1., 1.])
	print("\n\n\n-------------------------------- Generator Train Summary --------------------------------\n")
	generator_train.summary()
	plot_model(generator_train, to_file='../models/vaegan_generator_train.png')

	enc.trainable = True
	gen.trainable = False
	dis.trainable = False
	encoder_train.compile(optimizer, 
		[kl_loss],
		loss_weights=[1.])
	print("\n\n\n-------------------------------- Encoder Train Summary --------------------------------\n")
	encoder_train.summary()
	plot_model(encoder_train, to_file='../models/encoder_train.png')

	#Count training data
	train_steps = count_celeba_data(mode=0, batch_size=args.batch_size)
	val_steps = count_celeba_data(mode=1, batch_size=args.batch_size)

	initial_epoch = args.load_epoch + 1

	if args.load_epoch>0:
		discriminator_train.load_weights('models/discriminator.{:03d}.h5'.format(args.load_epoch))
		generator_train.load_weights('models/generator.{:03d}.h5'.format(args.load_epoch))
		encoder_train.load_weights('models/encoder.{:03d}.h5'.format(args.load_epoch))
		print('Loaded weights {:03d}'.format(args.load_epoch))

	images_loader_train = image_generator(mode=0)
	images_loader_val = image_generator(mode=1)

	seed = 0
	rng = np.random.RandomState(seed)

	for i in range(args.load_epoch+1, args.epoch+1):
		enc_losses_avg = np.zeros((3))
		gen_losses_avg = np.zeros((3))
		dis_losses_avg = np.zeros((4))

		# print("\nEpoch {:03d} \n".format(i))

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

			enc_losses = encoder_train.train_on_batch(enc_inputs, enc_outputs)
			gen_losses = generator_train.train_on_batch(gen_inputs, gen_outputs)
			dis_losses = discriminator_train.train_on_batch(dis_inputs, dis_outputs)
		
			m = 0
			for loss in dis_losses:
				dis_losses_avg[m] = (dis_losses_avg[m] * j + loss)/(j + 1)
				m = m + 1

			m = 0
			for loss in gen_losses:
				gen_losses_avg[m] = (gen_losses_avg[m] * j + loss)/(j + 1)
				m = m + 1

			# m = 0
			# for loss in enc_losses:
				# enc_losses_avg[m] = (enc_losses_avg[m] * j + loss)/(j + 1)
				# m = m + 1

			print('\rEpoch {:03d} Step {:04d} of {:04d}. D_real: {:.4f} D_z_vae: {:.4f} D_z_p {:.4f} G_z_vae_gan: {:.4f} G_z_p_gan: {:.4f}'.format(i, j, train_steps, dis_losses_avg[1], dis_losses_avg[2], dis_losses_avg[3], gen_losses_avg[1], gen_losses_avg[2]), end="", flush=True)

		print('\nSaving models')
		discriminator_train.save_weights('../weights/vaegan.discriminator.{:03d}.h5'.format(i))
		generator_train.save_weights('../weights/vaegan.generator.{:03d}.h5'.format(i))
		encoder_train.save_weights('../weights/vaegan.encoder.{:03d}.h5'.format(i))
		enc.save('../weights/vaegan.enc.{:03d}.h5'.format(i))
		gen.save('../weights/vaegan.gen.{:03d}.h5'.format(i))

		enc_losses_avg = np.zeros_like(enc_losses_avg)
		gen_losses_avg = np.zeros_like(gen_losses_avg)
		dis_losses_avg = np.zeros_like(dis_losses_avg)

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

			# m = 0
			# for loss in enc_losses:
			# 	enc_losses_avg[m] = (enc_losses_avg[m] * j + loss)/(j + 1)
			# 	m = m + 1

			print('\rEpoch {:03d} Eval Step {:04d} of {:04d}. D_real: {:.4f} D_z_vae: {:.4f} D_z_p {:.4f} G_z_vae_gan: {:.4f} G_z_p_gan: {:.4f}'.format(i, j, val_steps, dis_losses_avg[1], dis_losses_avg[2], dis_losses_avg[3], gen_losses_avg[1], gen_losses_avg[2]), end="", flush=True)
		print('\n')


if __name__=="__main__":
	parser = argparse.ArgumentParser(description='Train the network')
	parser.add_argument('-b', '--batch_size', type=int, default=64)
	parser.add_argument('-l', '--latent_dimension', type=int, default=128)
	parser.add_argument('-lr', '--learning_rate', type=float, default=0.0003)
	parser.add_argument('-fh', '--image_height', type=int, default=64)
	parser.add_argument('-fw', '--image_width', type=int, default=64)
	parser.add_argument('-fc', '--image_channels', type=int, default=3)
	parser.add_argument('-e', '--epoch', type=int, default=1000)
	parser.add_argument('-le', '--load_epoch', type=int, default=0)
	args = parser.parse_args()

	train(args)

	# im_gen = image_generator(mode=0)
	# while(True):
	# 	next(im_gen)