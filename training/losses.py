import keras.backend as K
import numpy as np


def kl_loss(y_true, y_pred):
	length = int(K.int_shape(y_pred)[2]/2.)

	z_mean = y_pred[:,0:length]
	z_log_var = y_pred[:,length:]

	# loss = K.mean(-0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1))
	loss = K.mean(-0.5 * K.sum(1 + K.clip(z_log_var, -10, 10) - K.square(z_mean) - K.exp(K.clip(z_log_var, -10, 10)), axis=-1))
	return loss

def mean_gaussian_negative_log_likelihood(y_true, y_pred):
	nll = 0.5 * np.log(2 * np.pi) + 0.5 * K.square(y_pred - y_true)
	axis = tuple(range(1, len(K.int_shape(y_true))))
	return K.mean(K.sum(nll, axis=axis), axis=-1)