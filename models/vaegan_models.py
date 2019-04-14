from keras.layers import Conv2D, Activation, BatchNormalization, MaxPooling2D, Dense, Flatten, Conv2DTranspose, UpSampling2D, LeakyReLU, Reshape, Lambda, Input, Concatenate
from models.SpectralNormalizationKeras import DenseSN, ConvSN2D
from keras.models import Model
from models.misc_models import sampling
from losses import mean_gaussian_negative_log_likelihood
import numpy as np
import keras.backend as K

def discriminator(
	batch_size=64,
	image_dimension=(64,64,3)):

	image_input_shape = (batch_size,) + image_dimension

	discriminator_input = Input(batch_shape=image_input_shape, name='discriminator_image_input')

	x = ConvSN2D(filters=32,
		kernel_size=5,
		strides=1,
		padding='same',
		name='discriminator_conv_1')(discriminator_input)
	x = LeakyReLU()(x)
	
	x = ConvSN2D(filters=128,
		kernel_size=5,
		strides=2,
		padding='same',
		name='discriminator_conv_2')(x)
	# x = MaxPooling2D()(x)
	# x = BatchNormalization()(x)
	x = LeakyReLU()(x)

	x = ConvSN2D(filters=256,
		kernel_size=5,
		strides=2,
		padding='same',
		name='discriminator_conv_3')(x)
	# x = MaxPooling2D()(x)
	# x = BatchNormalization()(x)
	x = LeakyReLU()(x)
	
	x = ConvSN2D(filters=256,
		kernel_size=5,
		strides=2,
		padding='same',
		name='discriminator_conv_4')(x)
	# x = MaxPooling2D()(x)
	# x = BatchNormalization()(x)
	x = LeakyReLU()(x)
	feature_output = x

	x = Flatten()(x)
	x = DenseSN(512, name='discriminator_dense_1')(x)
	# x = BatchNormalization()(x)
	x = LeakyReLU()(x)

	x = DenseSN(1, name='discriminator_dense_2')(x)
	discriminator_output = Activation('sigmoid')(x)

	return Model(inputs=discriminator_input, outputs=[discriminator_output, feature_output], name='discriminator')

def generator(
	batch_size=64,
	latent_dimension=128,
	decode_shape=(8,8,256)):

	latent_input_shape = (batch_size, latent_dimension)

	generator_input = Input(batch_shape=latent_input_shape, name='generator_latent_input')

	x = Dense(np.prod(decode_shape), name='generator_dense_1')(generator_input)
	x = BatchNormalization()(x)
	x = LeakyReLU()(x)
	x = Reshape(decode_shape)(x)

	x = Conv2DTranspose(filters=256,
		kernel_size=5,
		strides=2,
		padding='same',
		name='generator_conv_1')(x)
	# x = UpSampling2D()(x)
	x = BatchNormalization()(x)
	x = LeakyReLU()(x)

	x = Conv2DTranspose(filters=128,
		kernel_size=5,
		strides=2,
		padding='same',
		name='generator_conv_2')(x)
	# x = UpSampling2D()(x)
	x = BatchNormalization()(x)
	x = LeakyReLU()(x)

	x = Conv2DTranspose(filters=32,
		kernel_size=5,
		strides=2,
		padding='same',
		name='generator_conv_3')(x)
	# x = UpSampling2D()(x)
	x = BatchNormalization()(x)
	x = LeakyReLU()(x)

	x = Conv2DTranspose(filters=3,
		kernel_size=5,
		strides=1,
		padding='same',
		name='generator_conv_4')(x)
	generator_output = Activation('tanh')(x)

	return Model(inputs=generator_input, outputs=generator_output, name='generator')

def encoder(
	batch_size=64,
	latent_dimension=128,
	image_dimension=(64,64,3)):

	image_input_shape = (batch_size,) + image_dimension

	encoder_input = Input(batch_shape=image_input_shape, name='encoder_image_input')

	x = Conv2D(filters=64,
		kernel_size=5,
		strides=2,
		padding='same',
		name='encoder_conv_1')(encoder_input)
	# x = MaxPooling2D()(x)
	x = BatchNormalization()(x)
	x = LeakyReLU()(x)

	x = Conv2D(filters=128,
		kernel_size=5,
		strides=2,
		padding='same',
		name='encoder_conv_2')(x)
	# x = MaxPooling2D()(x)
	x = BatchNormalization()(x)
	x = LeakyReLU()(x)

	x = Conv2D(filters=256,
		kernel_size=5,
		strides=2,
		padding='same',
		name='encoder_conv_3')(x)
	# x = MaxPooling2D()(x)
	x = BatchNormalization()(x)
	x = LeakyReLU()(x)

	x = Flatten()(x)	
	x = Dense(1024, name='encoder_dense_1')(x)
	x = BatchNormalization()(x)
	x = LeakyReLU()(x)

	encoder_output_z_mean = Dense(latent_dimension, name='encoder_dense_z_mean')(x)
	encoder_output_z_log_var = Dense(latent_dimension, name='encoder_dense_z_log_var')(x)

	return Model(inputs=encoder_input, outputs=[encoder_output_z_mean, encoder_output_z_log_var], name='encoder')

def build_vaegan_graph(
	encoder,
	generator,
	discriminator,
	recon_vs_gan_weight=1e-6,
	batch_size=64,
	latent_dimension=128,
	image_dimension=(64,64,3)):

	sampler = Lambda(sampling, output_shape=(latent_dimension,), name='sampler')

	image_input_shape = (batch_size,) + image_dimension
	latent_input_shape = (batch_size, latent_dimension)

	image_input = Input(batch_shape=image_input_shape, name='image_input')
	z_p = Input(batch_shape=latent_input_shape, name='z_p_input')

	z_mean, z_log_var = encoder(image_input)

	z_vae = sampler([z_mean, z_log_var])

	encoder_output_z = Concatenate(axis=-1)([z_mean, z_log_var])

	generator_output_z_vae = generator(z_vae)
	generator_output_z_p = generator(z_p)

	discriminator_output_real, feature_output_real = discriminator(image_input)
	discriminator_output_z_vae, feature_output_z_vae = discriminator(generator_output_z_vae)
	discriminator_output_z_p, feature_output_z_p = discriminator(generator_output_z_p)

	nll_loss = mean_gaussian_negative_log_likelihood(feature_output_real, feature_output_z_vae)
	normalized_weight = recon_vs_gan_weight / (1. - recon_vs_gan_weight)

	encoder_train = Model(inputs=image_input, outputs=[encoder_output_z])
	encoder_train.add_loss(nll_loss)

	generator_train = Model(inputs=[image_input, z_p], outputs=[discriminator_output_z_vae, discriminator_output_z_p])
	generator_train.add_loss(normalized_weight * nll_loss)

	discriminator_train = Model(inputs=[image_input, z_p], outputs=[discriminator_output_real, discriminator_output_z_vae, discriminator_output_z_p])

	return encoder_train, generator_train, discriminator_train
