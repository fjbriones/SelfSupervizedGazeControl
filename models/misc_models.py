import keras.backend as K

def sampling(tensors):
	z_mean, z_log_var = tensors

	epsilon = K.random_normal(shape=K.shape(z_mean))
	
	return z_mean + K.exp(0.5*K.clip(z_log_var, -10, 10))*epsilon