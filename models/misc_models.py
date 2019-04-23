import keras.backend as K

def eval_kl(tensors):
	z_mean = tensors[0];
	z_log_var = tensors[1];

	kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)

	return kl_loss

def sampling(tensors):
	z_mean, z_log_var = tensors

	epsilon = K.random_normal(shape=K.shape(z_mean))
	
	return z_mean + K.exp(0.5*z_log_var)*epsilon