from keras.layers import Conv2D, Activation, BatchNormalization, MaxPooling2D, Dense, Flatten, Conv2DTranspose, UpSampling2D, LeakyReLU, Reshape, Lambda, Input, Concatenate
from models.SpectralNormalizationKeras import DenseSN, ConvSN2D
from keras.models import Model
from keras.regularizers import l2
from models.misc_models import sampling, eval_kl
from training.losses import mean_gaussian_negative_log_likelihood
import numpy as np
import keras.backend as K

def discriminator(
	batch_size=64,
	image_dimension=(64,64,3),
	weight_decay=1e-5,
	batch_norm_momentum=0.9,
	batch_norm_epsilon=1e-6,
	leaky_relu_alpha=0.2):

	image_input_shape = (batch_size,) + image_dimension

	discriminator_input = Input(batch_shape=image_input_shape, name='discriminator_image_input')

	x = ConvSN2D(filters=32,
		kernel_size=5,
		strides=1,
		padding='same',
		name='discriminator_conv_1',
		kernel_regularizer=l2(weight_decay),
		kernel_initializer='he_uniform')(discriminator_input)
	x = LeakyReLU(alpha=leaky_relu_alpha)(x)
	
	x = ConvSN2D(filters=128,
		kernel_size=5,
		strides=2,
		padding='same',
		name='discriminator_conv_2',
		kernel_regularizer=l2(weight_decay),
		kernel_initializer='he_uniform')(x)
	# x = BatchNormalization(momentum=batch_norm_momentum,
	# 	epsilon=batch_norm_epsilon)(x)
	x = LeakyReLU(alpha=leaky_relu_alpha)(x)

	x = ConvSN2D(filters=256,
		kernel_size=5,
		strides=2,
		padding='same',
		name='discriminator_conv_3',
		kernel_regularizer=l2(weight_decay),
		kernel_initializer='he_uniform')(x)
	# x = BatchNormalization(momentum=batch_norm_momentum,
	# 	epsilon=batch_norm_epsilon)(x)
	x = LeakyReLU(alpha=leaky_relu_alpha)(x)
	
	x = ConvSN2D(filters=256,
		kernel_size=5,
		strides=2,
		padding='same',
		name='discriminator_conv_4',
		kernel_regularizer=l2(weight_decay),
		kernel_initializer='he_uniform')(x)
	feature_output = x
	# x = BatchNormalization(momentum=batch_norm_momentum,
	# 	epsilon=batch_norm_epsilon)(x)
	x = LeakyReLU(alpha=leaky_relu_alpha)(x)
	

	x = Flatten()(x)
	x = DenseSN(512, name='discriminator_dense_1',
		kernel_regularizer=l2(weight_decay),
		kernel_initializer='he_uniform')(x)
	# x = BatchNormalization(momentum=batch_norm_momentum,
	# 	epsilon=batch_norm_epsilon)(x)
	x = LeakyReLU(alpha=leaky_relu_alpha)(x)

	x = DenseSN(1, name='discriminator_dense_2',
		kernel_regularizer=l2(weight_decay),
		kernel_initializer='he_uniform')(x)
	discriminator_output = Activation('sigmoid')(x)

	return Model(inputs=discriminator_input, outputs=[discriminator_output, feature_output], name='discriminator')

def generator(
	batch_size=64,
	latent_dimension=128,
	decode_shape=(8,8,256),
	weight_decay=1e-5,
	batch_norm_momentum=0.9,
	batch_norm_epsilon=1e-6,
	leaky_relu_alpha=0.2):

	latent_input_shape = (batch_size, latent_dimension)

	generator_input = Input(batch_shape=latent_input_shape, name='generator_latent_input')

	x = Dense(np.prod(decode_shape), name='generator_dense_1')(generator_input)
	x = BatchNormalization()(x)
	x = LeakyReLU(alpha=leaky_relu_alpha)(x)
	x = Reshape(decode_shape)(x)

	x = Conv2DTranspose(filters=256,
		kernel_size=5,
		strides=2,
		padding='same',
		name='generator_conv_1',
		kernel_regularizer=l2(weight_decay),
		kernel_initializer='he_uniform')(x)
	# x = UpSampling2D()(x)
	x = BatchNormalization(momentum=batch_norm_momentum,
		epsilon=batch_norm_epsilon)(x)
	x = LeakyReLU(alpha=leaky_relu_alpha)(x)

	x = Conv2DTranspose(filters=128,
		kernel_size=5,
		strides=2,
		padding='same',
		name='generator_conv_2',
		kernel_regularizer=l2(weight_decay),
		kernel_initializer='he_uniform')(x)
	# x = UpSampling2D()(x)
	x = BatchNormalization(momentum=batch_norm_momentum,
		epsilon=batch_norm_epsilon)(x)
	x = LeakyReLU(alpha=leaky_relu_alpha)(x)

	x = Conv2DTranspose(filters=32,
		kernel_size=5,
		strides=2,
		padding='same',
		name='generator_conv_3',
		kernel_regularizer=l2(weight_decay),
		kernel_initializer='he_uniform')(x)
	# x = UpSampling2D()(x)
	x = BatchNormalization(momentum=batch_norm_momentum,
		epsilon=batch_norm_epsilon)(x)
	x = LeakyReLU(alpha=leaky_relu_alpha)(x)

	x = Conv2DTranspose(filters=3,
		kernel_size=5,
		strides=1,
		padding='same',
		name='generator_conv_4',
		kernel_regularizer=l2(weight_decay),
		kernel_initializer='he_uniform')(x)
	generator_output = Activation('tanh')(x)

	return Model(inputs=generator_input, outputs=generator_output, name='generator')

def encoder(
	batch_size=64,
	latent_dimension=128,
	image_dimension=(64,64,3),
	weight_decay=1e-5,
	batch_norm_momentum=0.9,
	batch_norm_epsilon=1e-6,
	leaky_relu_alpha=0.2):

	image_input_shape = (batch_size,) + image_dimension

	encoder_input = Input(batch_shape=image_input_shape, name='encoder_image_input')

	x = Conv2D(filters=64,
		kernel_size=5,
		strides=2,
		padding='same',
		name='encoder_conv_1',
		kernel_regularizer=l2(weight_decay),
		kernel_initializer='he_uniform')(encoder_input)
	# x = MaxPooling2D()(x)
	x = BatchNormalization(momentum=batch_norm_momentum,
		epsilon=batch_norm_epsilon)(x)
	x = LeakyReLU(alpha=leaky_relu_alpha)(x)

	x = Conv2D(filters=128,
		kernel_size=5,
		strides=2,
		padding='same',
		name='encoder_conv_2',
		kernel_regularizer=l2(weight_decay),
		kernel_initializer='he_uniform')(x)
	# x = MaxPooling2D()(x)
	x = BatchNormalization(momentum=batch_norm_momentum,
		epsilon=batch_norm_epsilon)(x)
	x = LeakyReLU(alpha=leaky_relu_alpha)(x)

	x = Conv2D(filters=256,
		kernel_size=5,
		strides=2,
		padding='same',
		name='encoder_conv_3',
		kernel_regularizer=l2(weight_decay),
		kernel_initializer='he_uniform')(x)
	# x = MaxPooling2D()(x)
	x = BatchNormalization(momentum=batch_norm_momentum,
		epsilon=batch_norm_epsilon)(x)
	x = LeakyReLU(alpha=leaky_relu_alpha)(x)

	x = Flatten()(x)	
	x = Dense(1024, name='encoder_dense_1',
		kernel_regularizer=l2(weight_decay),
		kernel_initializer='he_uniform')(x)
	x = BatchNormalization()(x)
	x = LeakyReLU(alpha=leaky_relu_alpha)(x)

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

	# encoder_output_z = Concatenate(axis=-1)([z_mean, z_log_var])

	generator_output_z_vae = generator(z_vae)
	generator_output_z_p = generator(z_p)

	discriminator_output_real, feature_output_real = discriminator(image_input)
	discriminator_output_z_vae, feature_output_z_vae = discriminator(generator_output_z_vae)
	discriminator_output_z_p, feature_output_z_p = discriminator(generator_output_z_p)

	kl_loss = K.mean(-0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1))

	nll_loss = mean_gaussian_negative_log_likelihood(feature_output_real, feature_output_z_vae)
	normalized_weight = recon_vs_gan_weight / (1. - recon_vs_gan_weight)

	encoder_train = Model(inputs=image_input, outputs=[feature_output_z_vae])
	encoder_train.add_loss(nll_loss)
	encoder_train.add_loss(kl_loss)

	generator_train = Model(inputs=[image_input, z_p], outputs=[discriminator_output_z_vae, discriminator_output_z_p])
	generator_train.add_loss(normalized_weight * nll_loss)

	discriminator_train = Model(inputs=[image_input, z_p], outputs=[discriminator_output_real, discriminator_output_z_vae, discriminator_output_z_p])

	return encoder_train, generator_train, discriminator_train

def build_test_encoder(encoder,
	latent_dimension=128):

	sampler = Lambda(sampling, output_shape=(latent_dimension,), name='sampler')

	z_vae = sampler([encoder.outputs[0], encoder.outputs[1]])

	encoder_z_vae = Model(inputs=encoder.inputs, outputs=z_vae, name='encoder_with_z')

	return encoder_z_vae

def build_encoder(encoder,
	batch_size=2048,
	latent_dimension=128,
	image_dimension=(64,64,3)):

	input_shape = (batch_size,) + image_dimension

	enc_input = Input(batch_shape=input_shape)

	z_mean, z_log_var = encoder(enc_input)

	eval_loss = Lambda(eval_kl, name='calculate_kl')
	z = eval_loss([z_mean, z_log_var])

	encoder_loss = Model(inputs=enc_input, outputs=z, name="encoder_rl")

	return encoder_loss
