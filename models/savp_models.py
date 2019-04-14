from keras.layers import Conv2D, Add, Activation, Reshape, DepthwiseConv2D, BatchNormalization, Flatten, UpSampling3D, ConvLSTM2D, GlobalAveragePooling2D, Dropout
from keras.layers import Conv3D, RepeatVector, LeakyReLU, UpSampling2D, AveragePooling2D, GlobalAveragePooling3D, Concatenate, Input, TimeDistributed, Dense, LSTM, Lambda
from keras.models import Model
from keras.losses import mean_absolute_error
from keras.layers import Layer
# from keras.applications.mobilenet import 
from SpectralNormalizationKeras import DenseSN, ConvSN3D
import keras.backend as K
import tensorflow as tf

def discriminator(batch_size=1, time=10, name='gan', frame_height=64, frame_width=64, frame_channels=3):
	video_input_shape=(batch_size, time, frame_height, frame_width, frame_channels)
	video_input = Input(batch_shape=video_input_shape)
	paddings = [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]]

	x = ConvSN3D(filters=64,
		kernel_size=3,
		strides=1,
		padding='valid')(video_input)
	x = Lambda(lambda x: tf.pad(x, paddings))(x)
	x = LeakyReLU(0.1)(x)

	x = ConvSN3D(filters=128,
		kernel_size=4,
		strides=(1,2,2),
		padding='valid')(x)
	x = Lambda(lambda x: tf.pad(x, paddings))(x)
	x = LeakyReLU(0.1)(x)

	x = ConvSN3D(filters=128,
		kernel_size=3,
		strides=1,
		padding='valid')(x)
	x = Lambda(lambda x: tf.pad(x, paddings))(x)
	x = LeakyReLU(0.1)(x)

	x = ConvSN3D(filters=256,
		kernel_size=4,
		strides=(1,2,2),
		padding='valid')(x)
	x = Lambda(lambda x: tf.pad(x, paddings))(x)
	x = LeakyReLU(0.1)(x)	

	x = ConvSN3D(filters=256,
		kernel_size=3,
		strides=1,
		padding='valid')(x)
	x = Lambda(lambda x: tf.pad(x, paddings))(x)
	x = LeakyReLU(0.1)(x)

	x = ConvSN3D(filters=512,
		kernel_size=4,
		strides=2,
		padding='valid')(x)
	x = Lambda(lambda x: tf.pad(x, paddings))(x)
	x = LeakyReLU(0.1)(x)

	x = ConvSN3D(filters=512,
		kernel_size=3,
		strides=1,
		padding='valid')(x)
	x = Lambda(lambda x: tf.pad(x, paddings))(x)
	x = LeakyReLU(0.1)(x)

	x = GlobalAveragePooling3D()(x)

	x = DenseSN(1)(x)
	x = Activation('sigmoid')(x)

	return Model(inputs=video_input, outputs=x, name='discriminator_'+name)

def convolve_previous(tensors):
	prev_frame = tensors[0]
	kernels = tensors[1]
	batch_size, time, height, width, color_channels = list(K.int_shape(prev_frame))
	batch_size, time, kernel_height, kernel_width, num_transformed_images = list(K.int_shape(kernels))
	kernel_size = [kernel_height, kernel_width]

	kernels = K.permute_dimensions(kernels, (1,2,3,0,4))
	kernels = K.reshape(kernels, [time, kernel_size[0], kernel_size[1], batch_size, num_transformed_images])

	prev_frame_transposed = K.permute_dimensions(prev_frame, (1,4,2,3,0))
	
	for j in range(time):
		out_frames = K.depthwise_conv2d(prev_frame_transposed[j], kernels[j], padding='same')
		
		out_frames = K.reshape(out_frames, [color_channels, height, width, batch_size, num_transformed_images])
		out_frames = K.permute_dimensions(out_frames, (3,4,1,2,0))
		
		out_frames = K.expand_dims(out_frames, axis=1)
		if (j == 0):
			out_frames_times = out_frames
		else:
			out_frames_times = K.concatenate([out_frames_times, out_frames], axis=1)

	return out_frames_times

def mask(tensors):
	masks = tensors[0]
	first_frame = tensors[1]
	prev_frames = tensors[2]
	sp_frames = tensors[3]
	wp_frames = tensors[4]

	time = K.int_shape(prev_frames)[1]
	channels = K.int_shape(prev_frames)[4]

	rep_first_frame = K.repeat_elements(K.expand_dims(first_frame, axis=1), rep=time, axis=1)
	

	# out_frames_times = rep_first_frame * K.expand_dims(masks[:,:,:,:,0], axis=-1)
	# out_frames_times += prev_frames * K.expand_dims(masks[:,:,:,:,1], axis=-1)
	# out_frames_times += sp_frames * K.expand_dims(masks[:,:,:,:,2], axis=-1)
	# out_frames_times += wp_frames[:,:,0,:,:,:] * K.expand_dims(masks[:,:,:,:,3], axis=-1)
	# out_frames_times += wp_frames[:,:,1,:,:,:] * K.expand_dims(masks[:,:,:,:,4], axis=-1)
	# out_frames_times += wp_frames[:,:,2,:,:,:] * K.expand_dims(masks[:,:,:,:,5], axis=-1)
	# out_frames_times += wp_frames[:,:,3,:,:,:] * K.expand_dims(masks[:,:,:,:,6], axis=-1)

	chan_masks = K.repeat_elements(K.expand_dims(masks, axis=-1), rep=channels, axis=-1)

	out_frames_times = rep_first_frame * chan_masks[:,:,:,:,0,:]
	out_frames_times += prev_frames * chan_masks[:,:,:,:,1,:]
	out_frames_times += sp_frames * chan_masks[:,:,:,:,2,:]
	out_frames_times += wp_frames[:,:,0,:,:,:] * chan_masks[:,:,:,:,3,:]
	out_frames_times += wp_frames[:,:,1,:,:,:] * chan_masks[:,:,:,:,4,:]
	out_frames_times += wp_frames[:,:,2,:,:,:] * chan_masks[:,:,:,:,5,:]
	out_frames_times += wp_frames[:,:,3,:,:,:] * chan_masks[:,:,:,:,6,:]

	# # print(K.int_shape(prev_frames)[1])
	# for k in range(time):
	# 	# print(K.int_shape(prev_frames[:,0,:,:,0]))
	# 	# print(K.int_shape(masks[:,k,:,:,1]))
	# 	for j in range(channels):
	# 		# print(K.int_shape(first_frame[:,:,:,j]))
	# 		first_frame_mask = first_frame[:,:,:,j] * masks[:,k,:,:,0]
	# 		prev_frames_mask = prev_frames[:,k,:,:,j] * masks[:,k,:,:,1]
			
	# 		sp_frames_mask = sp_frames[:,k,:,:,j] * masks[:,k,:,:,2]
			
	# 		wp_frames_mask_1 = wp_frames[:,k,0,:,:,j] * masks[:,k,:,:,3]
	# 		wp_frames_mask_2 = wp_frames[:,k,1,:,:,j] * masks[:,k,:,:,4]
	# 		wp_frames_mask_3 = wp_frames[:,k,2,:,:,j] * masks[:,k,:,:,5]
	# 		wp_frames_mask_4 = wp_frames[:,k,3,:,:,j] * masks[:,k,:,:,6]
			
	# 		out_frame = first_frame_mask + prev_frames_mask + sp_frames_mask + wp_frames_mask_1 + wp_frames_mask_2 + wp_frames_mask_3 + wp_frames_mask_4
			
	# 		out_frame = K.expand_dims(out_frame)
	# 		if j == 0:
	# 			out_frames = out_frame
	# 		else:
	# 			out_frames = K.concatenate([out_frames, out_frame])
	# 	out_frames_time = K.expand_dims(out_frames, axis=1)
	# 	if k == 0:
	# 		out_frames_times = out_frames_time
	# 	else:
	# 		out_frames_times = K.concatenate([out_frames_times, out_frames_time], axis=1)

	return out_frames_times

def generator(batch_size=1, time=10, latent_dim=8., frame_height=64, frame_width=64, frame_channels=3):
	video_input_shape = (batch_size, time, frame_height, frame_width, frame_channels)
	latent_input_shape = (batch_size, time, latent_dim)
	first_input_shape = (batch_size, frame_height, frame_width, frame_channels)
	# latent_input_shape = (latent_dim,)

	first_input = Input(batch_shape=first_input_shape)
	video_input = Input(batch_shape=video_input_shape)
	latent_input = Input(batch_shape=latent_input_shape)
	# y = Reshape((1,K.int_shape(latent_input)[1]))(latent_input)
	y = Reshape((time, latent_dim))(latent_input)
	y = LSTM(latent_dim, return_sequences=True)(y)

	x = TimeDistributed(Conv2D(filters=32,
		kernel_size=5,
		strides=1,
		padding='same'))(video_input)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	y1 = TimeDistributed(RepeatVector(K.int_shape(x)[2]*K.int_shape(x)[3]))(y)
	y1 = Reshape((time, K.int_shape(x)[2], K.int_shape(x)[3],-1))(y1)
	x = Concatenate()([x,y1])
	
	x1 = ConvLSTM2D(filters=32,
		kernel_size=5,
		strides=1,
		padding='same',
		return_sequences=True)(x)
	x = BatchNormalization()(x1)
	x = Activation('relu')(x)
	y1 = TimeDistributed(RepeatVector(K.int_shape(x)[2]*K.int_shape(x)[3]))(y)
	y1 = Reshape((time, K.int_shape(x)[2], K.int_shape(x)[3],-1))(y1)
	x = Concatenate()([x,y1])

	x1 = TimeDistributed(AveragePooling2D())(x)

	x = TimeDistributed(Conv2D(filters=64,
		kernel_size=5,
		strides=1,
		padding='same'))(x)
	x = BatchNormalization()(x1)
	x = Activation('relu')(x)
	y1 = TimeDistributed(RepeatVector(K.int_shape(x)[2]*K.int_shape(x)[3]))(y)
	y1 = Reshape((time, K.int_shape(x)[2], K.int_shape(x)[3],-1))(y1)
	x = Concatenate()([x,y1])

	x2 = ConvLSTM2D(filters=64,
		kernel_size=5,
		strides=1,
		padding='same',
		return_sequences=True)(x)
	x = BatchNormalization()(x2)
	x = Activation('relu')(x)
	y1 = TimeDistributed(RepeatVector(K.int_shape(x)[2]*K.int_shape(x)[3]))(y)
	y1 = Reshape((time, K.int_shape(x)[2], K.int_shape(x)[3],-1))(y1)
	x = Concatenate()([x,y1])

	x2 = TimeDistributed(AveragePooling2D())(x)

	x = TimeDistributed(Conv2D(filters=128,
		kernel_size=5,
		strides=1,
		padding='same'))(x)
	x = BatchNormalization()(x2)
	x = Activation('relu')(x)
	y1 = TimeDistributed(RepeatVector(K.int_shape(x)[2]*K.int_shape(x)[3]))(y)
	y1 = Reshape((time, K.int_shape(x)[2], K.int_shape(x)[3],-1))(y1)
	x = Concatenate()([x,y1])
	
	x3 = ConvLSTM2D(filters=128,
		kernel_size=5,
		strides=2,
		padding='same',
		return_sequences=True)(x)
	x = BatchNormalization()(x3)
	x = Activation('relu')(x)
	y1 = TimeDistributed(RepeatVector(K.int_shape(x)[2]*K.int_shape(x)[3]))(y)
	y1 = Reshape((time, K.int_shape(x)[2], K.int_shape(x)[3],-1))(y1)
	x = Concatenate()([x,y1])
	
	x = TimeDistributed(UpSampling2D(size=(2,2), interpolation='bilinear'))(x)

	x = Concatenate()([x2, x])

	x = TimeDistributed(Conv2D(filters=64,
		kernel_size=5,
		strides=1,
		padding='same'))(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	y1 = TimeDistributed(RepeatVector(K.int_shape(x)[2]*K.int_shape(x)[3]))(y)
	y1 = Reshape((time, K.int_shape(x)[2], K.int_shape(x)[3],-1))(y1)
	x = Concatenate()([x,y1])

	x4 = ConvLSTM2D(filters=64,
		kernel_size=5,
		strides=1,
		padding='same',
		return_sequences=True)(x)
	x = BatchNormalization()(x4)
	x = Activation('relu')(x)
	y1 = TimeDistributed(RepeatVector(K.int_shape(x)[2]*K.int_shape(x)[3]))(y)
	y1 = Reshape((time, K.int_shape(x)[2], K.int_shape(x)[3],-1))(y1)
	x = Concatenate()([x, y1])
	
	x = TimeDistributed(UpSampling2D(size=(2,2), interpolation='bilinear'))(x)

	x = Concatenate()([x1, x])

	x = TimeDistributed(Conv2D(filters=32,
		kernel_size=5,
		strides=1,
		padding='same'))(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	y1 = TimeDistributed(RepeatVector(K.int_shape(x)[2]*K.int_shape(x)[3]))(y)
	y1 = Reshape((time, K.int_shape(x)[2], K.int_shape(x)[3],-1))(y1)
	x = Concatenate()([x,y1])

	x5 = ConvLSTM2D(filters=32,
		kernel_size=5,
		strides=1,
		padding='same',
		return_sequences=True)(x)
	x = BatchNormalization()(x5)
	x = Activation('relu')(x)	
	y1 = TimeDistributed(RepeatVector(K.int_shape(x)[2]*K.int_shape(x)[3]))(y)
	y1 = Reshape((time, K.int_shape(x)[2], K.int_shape(x)[3],-1))(y1)
	x = Concatenate()([x,y1])
	
	x = TimeDistributed(UpSampling2D(size=(2,2), interpolation='bilinear'))(x)

	x = TimeDistributed(Conv2D(filters=32,
		kernel_size=5,
		strides=1,
		padding='same'))(x)
	x = BatchNormalization()(x)
	main_output = Activation('relu')(x)

	sp = TimeDistributed(Conv2D(filters=32,
		kernel_size=5,
		strides=1,
		padding='same'))(main_output)
	sp = BatchNormalization()(sp)
	sp = Activation('relu')(sp)

	sp = TimeDistributed(Conv2D(filters=frame_channels,
		kernel_size=1,
		strides=1,
		padding='same'))(sp)
	sp = BatchNormalization()(sp)
	sp_output = Activation('sigmoid')(sp)

	cdna = TimeDistributed(Flatten())(x3)
	cdna = TimeDistributed(Dense(100))(cdna)
	cdna_kernel = TimeDistributed(Reshape((5,5,4)))(cdna)

	wp_layer = Lambda(convolve_previous)
	wp = wp_layer([video_input, cdna_kernel])
	# wp = TimeDistributed(ConvolveKernel())([video_input, cdna_kernel])
	wp_output = TimeDistributed(Activation('sigmoid'))(wp)

	cm = TimeDistributed(Conv2D(filters=32,
		kernel_size=5,
		strides=1,
		padding='same'))(main_output)
	cm = BatchNormalization()(cm)
	cm = TimeDistributed(Activation('relu'))(cm)

	cm = TimeDistributed(Conv2D(filters=7,
		kernel_size=5,
		strides=1,
		padding='same'))(cm)
	cm = BatchNormalization()(cm)
	cm_output = TimeDistributed(Activation('softmax'))(cm)

	mask_layer = Lambda(mask)
	mask_output = mask_layer([cm_output, first_input, video_input, sp_output, wp_output])

	return Model(inputs=[first_input, video_input, latent_input], outputs=mask_output, name="generator")

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

	x = TimeDistributed(Conv2D(filters, kernel, padding='same', strides=strides))(inputs)
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

	x = TimeDistributed(DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same'))(x)
	x = BatchNormalization(axis=channel_axis)(x)
	x = Activation('relu')(x)

	x = TimeDistributed(Conv2D(filters, (1, 1), strides=(1, 1), padding='same'))(x)
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


def encoder(batch_size=1, time=32, latent_dim=8, frame_height=64, frame_width=64, frame_channels=3):
	video_input_shape = (batch_size, time, frame_height, frame_width, frame_channels*2)
	latent_input_shape = (batch_size, time, latent_dim)

	images_input = Input(batch_shape=video_input_shape)
	
	# x = Concatenate()([image_input_t0, image_input_t1])

	    
	x = _conv_block(images_input, 32, (3, 3), strides=(2, 2))

	x = _inverted_residual_block(x, 16, (3, 3), t=1, strides=1, n=1)
	x = _inverted_residual_block(x, 24, (3, 3), t=6, strides=2, n=2)
	x = _inverted_residual_block(x, 32, (3, 3), t=6, strides=2, n=3)
	x = _inverted_residual_block(x, 64, (3, 3), t=6, strides=2, n=4)
	x = _inverted_residual_block(x, 96, (3, 3), t=6, strides=1, n=3)
	x = _inverted_residual_block(x, 160, (3, 3), t=6, strides=2, n=3)
	x = _inverted_residual_block(x, 320, (3, 3), t=6, strides=1, n=1)

	x = _conv_block(x, 1280, (1, 1), strides=(1, 1))
	x = TimeDistributed(GlobalAveragePooling2D())(x)

	# x = TimeDistributed(Conv2D(filters=64,
	# 	kernel_size=3,
	# 	strides=1,
	# 	padding="same"))(images_input)
	# x = BatchNormalization()(x)
	# x = Activation('relu')(x)

	# x = TimeDistributed(Conv2D(filters=64,
	# 	kernel_size=3,
	# 	strides=2,
	# 	padding="same"))(images_input)
	# x = BatchNormalization()(x)
	# x = Activation('relu')(x)

	# x = TimeDistributed(Conv2D(filters=128,
	# 	kernel_size=3,
	# 	strides=1,
	# 	padding="same"))(x)
	# x = BatchNormalization()(x)
	# x = Activation('relu')(x)

	# x = TimeDistributed(Conv2D(filters=128,
	# 	kernel_size=3,
	# 	strides=2,
	# 	padding="same"))(x)
	# x = BatchNormalization()(x)
	# x = Activation('relu')(x)

	# x = TimeDistributed(Conv2D(filters=256,
	# 	kernel_size=3,
	# 	strides=1,
	# 	padding="same"))(x)
	# x = BatchNormalization()(x)
	# x = Activation('relu')(x)

	# x = TimeDistributed(Conv2D(filters=256,
	# 	kernel_size=3,
	# 	strides=2,
	# 	padding="same"))(x)
	# x = BatchNormalization()(x)
	# x = Activation('relu')(x)
	# x = TimeDistributed(Flatten())(x)

	x = TimeDistributed(Dense(1024))(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	x = TimeDistributed(Dense(128))(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	mean = TimeDistributed(Dense(latent_dim))(x)
	#Clip from -10 to 10
	sd = TimeDistributed(Dense(latent_dim))(x)

	return Model(inputs=images_input, outputs=[mean, sd], name='encoder')

def sampling(args):
	z_mean, z_log_var = args
	batch = K.shape(z_mean)[0]
	dim = K.int_shape(z_mean)[1:]

	epsilon = K.random_normal(shape=K.shape(z_mean))
	return z_mean + K.exp(0.5*K.clip(z_log_var, -10, 10))*epsilon

def trial_predictor(image_input,
	time_pred = 32):
	x1 = ConvLSTM2D(filters=32,
		kernel_size=5,
		strides=2,
		padding='same',
		return_sequences=True)(image_input)
	x = BatchNormalization()(x1)
	x = Activation('relu')(x)

	x2 = ConvLSTM2D(filters=64,
		kernel_size=5,
		strides=2,
		padding='same',
		return_sequences=True)(x)
	x = BatchNormalization()(x2)
	x = Activation('relu')(x)
	
	x3 = ConvLSTM2D(filters=128,
		kernel_size=5,
		strides=2,
		padding='same',
		return_sequences=True)(x)
	x = BatchNormalization()(x3)
	x = Activation('relu')(x)
	
	x = UpSampling3D(size=(1,2,2))(x)
	x = Concatenate()([x2, x])
	x4 = ConvLSTM2D(filters=64,
		kernel_size=5,
		strides=1,
		padding='same',
		return_sequences=True)(x)
	x = BatchNormalization()(x4)
	x = Activation('relu')(x)
	
	x = UpSampling3D(size=(1,2,2))(x)
	x = Concatenate()([x1, x])
	x5 = ConvLSTM2D(filters=32,
		kernel_size=5,
		strides=1,
		padding='same',
		return_sequences=True)(x)
	x = BatchNormalization()(x5)
	x = Activation('relu')(x)
	
	x = UpSampling3D(size=(1,2,2))(x)
	x = ConvLSTM2D(filters=3,
		kernel_size=3,
		strides=1,
		padding='same',
		return_sequences=True)(x)
	x = Reshape(target_shape=(K.int_shape(image_input)[1:]))(x)
	x = BatchNormalization()(x)
	x = Activation('sigmoid')(x)
	
	return x

def build_graph(encoder, 
	generator, 
	discriminator_gan, 
	discriminator_vae,
	batch_size=1,
	time=10, 
	latent_dim=8, 
	frame_height=64, 
	frame_width=64, 
	frame_channels=3):
	sampler = Lambda(sampling, output_shape=(2*time-1,latent_dim), name='sampler')

	x = Input(batch_shape=(batch_size, frame_height, frame_width, frame_channels), name='input_first_frame_video')
	x1 = Input(batch_shape=(batch_size, time, frame_height, frame_width, frame_channels), name='input_current_frame_video_generator')
	
	x_current = Input(batch_shape=(batch_size, 2*time-1, frame_height, frame_width, frame_channels), name='input_curr_frame_video_enc')
	x_next = Input(batch_shape=(batch_size, 2*time-1, frame_height, frame_width, frame_channels), name='input_next_frame_video_enc')
	
	z_p = Input(batch_shape=(batch_size, 2*time-1, latent_dim))

	enc_input = Concatenate(axis=-1)([x_current, x_next])

	z_mean, z_log_var = encoder(enc_input)

	z_vae = sampler([z_mean, z_log_var])

	# x_vae = generator([x, x_current, z_vae])
	# x_gan = generator([x, x_current, z_p])

	for i in range(time):
		z_vae_time = Lambda(lambda z: z[:, i:i+time, :], output_shape=(time, latent_dim))(z_vae)
		z_p_time = Lambda(lambda z: z[:, i:i+time, :], output_shape=(time, latent_dim))(z_p)
		x_current_time = Lambda(lambda z: z[:, i:i+time, :, :, :], output_shape=(time,K.int_shape(x_current)[2], K.int_shape(x_current)[3], K.int_shape(x_current)[4]))(x_current)
		
		x_vae_out = generator([x, x_current_time, z_vae_time])
		x_gan_out = generator([x, x_current_time, z_p_time])

		if i == 0:
			x_vae = x_vae_out
			x_gan = x_gan_out
		else:
			x_vae_out_last = Lambda(lambda z: K.expand_dims(z[:, time-1, :, :, :], axis=1), output_shape=(1,K.int_shape(x_vae_out)[2], K.int_shape(x_vae_out)[3], K.int_shape(x_vae_out)[4]))(x_vae_out)
			x_gan_out_last = Lambda(lambda z: K.expand_dims(z[:, time-1, :, :, :], axis=1), output_shape=(1,K.int_shape(x_gan_out)[2], K.int_shape(x_gan_out)[3], K.int_shape(x_gan_out)[4]))(x_gan_out)
			
			x_vae = Concatenate(axis=1)([x_vae, x_vae_out_last])
			x_gan = Concatenate(axis=1)([x_gan, x_gan_out_last])

	dis_true_vae = discriminator_vae(x_next)
	dis_true_gan = discriminator_gan(x_next)
	dis_vae = discriminator_vae(x_vae)
	dis_gan = discriminator_gan(x_gan)

	generator_train = Model(inputs=[x, x1, x_next, x_current, z_p], outputs=[dis_vae, dis_gan, x_vae], name='generator_train')

	discriminator_train = Model(inputs=[x, x1, x_next, x_current, z_p], outputs=[dis_true_vae, dis_true_gan, dis_vae, dis_gan], name='discriminator_train')
	
	z_out = Concatenate(axis=-1)([z_mean, z_log_var])

	vaegan = Model(inputs=[x, x1, x_next, x_current, z_p], outputs=x_gan)

	# kl_loss = K.mean(-0.5 * K.sum(1 + K.clip(z_log_var, -10, 10) - K.square(z_mean) - K.exp(K.clip(z_log_var, -10, 10)), axis=-1))
	encoder_train = Model(inputs=[x, x1, x_current, x_next], outputs=[z_out, x_vae], name='encoder_train')


	return encoder_train, generator_train, discriminator_train, vaegan

def eval_kl(tensors):
	z_mean = tensors[0];
	z_log_var = tensors[1];

	kl_loss = -0.5 * K.sum(1 + K.clip(z_log_var, -10, 10) - K.square(z_mean) - K.exp(K.clip(z_log_var, -10, 10)), axis=-1)

	return kl_loss

def build_encoder(encoder,
	time=1,
	latent_dim=8,
	frame_height=64, 
	frame_width=64, 
	frame_channels=3,
	batch_size=1):

	x_current = Input(batch_shape=(batch_size, 2*time - 1, frame_height, frame_width, frame_channels), name='input_curr_frame_video_enc')
	x_next = Input(batch_shape=(batch_size, 2*time - 1, frame_height, frame_width, frame_channels), name='input_next_frame_video_enc')

	enc_input = Concatenate(axis=-1)([x_current, x_next])

	z_mean, z_log_var = encoder(enc_input)

	# z_out = Concatenate(axis=-1)([z_mean, z_log_var])
	eval_loss = Lambda(eval_kl, name='calculate_kl')
	z = eval_loss([z_mean, z_log_var])

	enc_rl = Model(inputs= [x_current, x_next], outputs=z, name="encoder_rl")

	return enc_rl
