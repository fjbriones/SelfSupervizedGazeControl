import sys
sys.path.append('../')

from generator.vaegan_generator import image_generator
from keras.models import load_model
from models.vaegan_models import build_test_encoder
import numpy as np
import argparse
import cv2

def test(args):
	seed = 0
	rng = np.random.RandomState(seed)
	gen = load_model(args.generator_model)
	enc = load_model(args.encoder_model)
	enc_z = build_test_encoder(enc)
	images_loader_test = image_generator(mode=2)
	while(True):
		if(args.use_encoder):
			images_batch = next(images_loader_test)
			z = np.squeeze(enc_z.predict_on_batch(images_batch))
		else:
			z = rng.normal(size=(args.batch_size, args.latent_dimension))

		generator_output = np.squeeze(gen.predict_on_batch(z))

		display_tiles = int(np.sqrt(args.batch_size))
		display_box = np.empty((display_tiles*generator_output.shape[1], display_tiles*generator_output.shape[2], generator_output.shape[3]))

		for i in range(display_tiles):
			for j in range(display_tiles):
				display_box[i*generator_output.shape[1]:(i+1)*generator_output.shape[1], j*generator_output.shape[2]:(j+1)*generator_output.shape[2],:] = generator_output[i*display_tiles+j]

		cv2.imshow('results', display_box)
		cv2.waitKey(0)
		cv2.destroyAllWindows()


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Test the vaegan network')
	parser.add_argument('-b', '--batch_size', type=int, default=64)
	parser.add_argument('-l', '--latent_dimension', type=int, default=128)
	parser.add_argument('-gm', '--generator_model', type=str, default='../weights/vaegan.gen.020.h5')
	parser.add_argument('-em', '--encoder_model', type=str, default='../weights/vaegan.enc.050.h5')
	parser.add_argument('-ue', '--use_encoder', action='store_true')
	args = parser.parse_args()

	test(args)
