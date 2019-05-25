from keras.layers import Conv2D, Activation, BatchNormalization, MaxPooling2D, Dense, Flatten, Conv2DTranspose, UpSampling2D, LeakyReLU, Reshape, Lambda, Input, Concatenate, DepthwiseConv2D, Add, GlobalAveragePooling2D
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

def _conv_block(inputs, filters, kernel, strides):
	"""Convolution Block
	This function defines a 2D convolution operation with BN and relu6.
	# Arguments
	inputs: Tensor, input tensor of conv layer.
	filters: Integer, the dimensionality of the output space.
	kernel: An integer or tuple/list of 2 integers, specifying the
	width and height of the 2D convolution window.
	strides: An integer or tuple/list of 2 integers,
	specifying the strides of the convolution along the width and height.
	Can be a single integer to specify the same value for
	all spatial dimensions.
	# Returns
	Output tensor.
	"""

	channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

	x = Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
	x = BatchNormalization(axis=channel_axis)(x)
	return Activation('relu')(x)


def _bottleneck(inputs, filters, kernel, t, s, r=False):
	"""Bottleneck
	This function defines a basic bottleneck structure.
	# Arguments
	inputs: Tensor, input tensor of conv layer.
	filters: Integer, the dimensionality of the output space.
	kernel: An integer or tuple/list of 2 integers, specifying the
	width and height of the 2D convolution window.
	t: Integer, expansion factor.
	t is always applied to the input size.
	s: An integer or tuple/list of 2 integers,specifying the strides
	of the convolution along the width and height.Can be a single
	integer to specify the same value for all spatial dimensions.
	r: Boolean, Whether to use the residuals.
	# Returns
	Output tensor.
	"""

	channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
	tchannel = K.int_shape(inputs)[channel_axis] * t

	x = _conv_block(inputs, tchannel, (1, 1), (1, 1))

	x = DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
	x = BatchNormalization(axis=channel_axis)(x)
	x = Activation('relu')(x)

	x = Conv2D(filters, (1, 1), strides=(1, 1), padding='same')(x)
	x = BatchNormalization(axis=channel_axis)(x)

	if r:
		x = Add()([x, inputs])
	return x


def _inverted_residual_block(inputs, filters, kernel, t, strides, n):
	"""Inverted Residual Block
	This function defines a sequence of 1 or more identical layers.
	# Arguments
	inputs: Tensor, input tensor of conv layer.
	filters: Integer, the dimensionality of the output space.
	kernel: An integer or tuple/list of 2 integers, specifying the
	width and height of the 2D convolution window.
	t: Integer, expansion factor.
	t is always applied to the input size.
	s: An integer or tuple/list of 2 integers,specifying the strides
	of the convolution along the width and height.Can be a single
	integer to specify the same value for all spatial dimensions.
	n: Integer, layer repeat times.
	# Returns
	Output tensor.
	"""

	x = _bottleneck(inputs, filters, kernel, t, strides)

	for i in range(1, n):
		x = _bottleneck(x, filters, kernel, t, 1, True)

	return x

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

	x = _conv_block(encoder_input, 32, (3, 3), strides=(2, 2))

	x = _inverted_residual_block(x, 16, (3, 3), t=1, strides=1, n=1)
	x = _inverted_residual_block(x, 24, (3, 3), t=6, strides=2, n=2)
	x = _inverted_residual_block(x, 32, (3, 3), t=6, strides=2, n=3)
	x = _inverted_residual_block(x, 64, (3, 3), t=6, strides=2, n=4)
	x = _inverted_residual_block(x, 96, (3, 3), t=6, strides=1, n=3)
	x = _inverted_residual_block(x, 160, (3, 3), t=6, strides=2, n=3)
	x = _inverted_residual_block(x, 320, (3, 3), t=6, strides=1, n=1)

	x = _conv_block(x, 1280, (1, 1), strides=(1, 1))
	x = GlobalAveragePooling2D()(x)

	# x = Conv2D(filters=64,
	# 	kernel_size=5,
	# 	strides=2,
	# 	padding='same',
	# 	name='encoder_conv_1',
	# 	kernel_regularizer=l2(weight_decay),
	# 	kernel_initializer='he_uniform')(encoder_input)
	# # x = MaxPooling2D()(x)
	# x = BatchNormalization(momentum=batch_norm_momentum,
	# 	epsilon=batch_norm_epsilon)(x)
	# x = LeakyReLU(alpha=leaky_relu_alpha)(x)

	# x = Conv2D(filters=128,
	# 	kernel_size=5,
	# 	strides=2,
	# 	padding='same',
	# 	name='encoder_conv_2',
	# 	kernel_regularizer=l2(weight_decay),
	# 	kernel_initializer='he_uniform')(x)
	# # x = MaxPooling2D()(x)
	# x = BatchNormalization(momentum=batch_norm_momentum,
	# 	epsilon=batch_norm_epsilon)(x)
	# x = LeakyReLU(alpha=leaky_relu_alpha)(x)

	# x = Conv2D(filters=256,
	# 	kernel_size=5,
	# 	strides=2,
	# 	padding='same',
	# 	name='encoder_conv_3',
	# 	kernel_regularizer=l2(weight_decay),
	# 	kernel_initializer='he_uniform')(x)
	# # x = MaxPooling2D()(x)
	# x = BatchNormalization(momentum=batch_norm_momentum,
	# 	epsilon=batch_norm_epsilon)(x)
	# x = LeakyReLU(alpha=leaky_relu_alpha)(x)

	# x = Flatten()(x)	
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
