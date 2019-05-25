import sys
sys.path.append('../')

from generator.vaegan_generator import celeba_image_generator, humanm_image_generator
from keras.models import load_model
from models.vaegan_models import build_test_encoder, build_encoder
from utils.utils import count_celeba_data, count_humanm_data
import numpy as np
import argparse
import cv2

def predict(args):
	seed = 0
	rng = np.random.RandomState(seed)
	gen = load_model(args.generator_model)
	enc = load_model(args.encoder_model)
	enc_z = build_test_encoder(enc)

	if (args.dataset == 'celeba'):
		images_loader_test = celeba_image_generator(mode=2, rng=rng, batch_size=args.batch_size)
	elif (args.dataset == 'humanm'):
		images_loader_test = humanm_image_generator(video_dir='../data/Human3.6M/test', batch_size=args.batch_size)

	count = 0;

	while(True):
		if(args.use_encoder):
			images_batch = next(images_loader_test)
			z = np.squeeze(enc_z.predict_on_batch(images_batch))
		else:
			z = rng.normal(size=(args.batch_size, args.latent_dimension))
			# z = np.zeros_like(z)

		generator_output = ((np.squeeze(gen.predict_on_batch(z)) + 1.0)/2.0)#.astype(np.uint8)

		display_tiles = int(np.sqrt(args.batch_size))

		display_box = np.zeros((display_tiles*generator_output.shape[1], display_tiles*generator_output.shape[2], generator_output.shape[3]))
		display_gt = np.zeros((display_tiles*generator_output.shape[1], display_tiles*generator_output.shape[2], generator_output.shape[3]))

		for i in range(display_tiles):
			for j in range(display_tiles):

				# cv2.imshow('result', generator_output[i*display_tiles+j])
				# cv2.waitKey(0)
				# cv2.destroyAllWindows()
				display_box[i*generator_output.shape[1]:(i+1)*generator_output.shape[1], j*generator_output.shape[2]:(j+1)*generator_output.shape[2],:] = generator_output[i*display_tiles+j]
				display_gt[i*generator_output.shape[1]:(i+1)*generator_output.shape[1], j*generator_output.shape[2]:(j+1)*generator_output.shape[2],:] = (images_batch[i*display_tiles+j] + 1.0)/2.0

		cv2.imshow('results', display_box)
		cv2.imshow('ground truth', display_gt)
		cv2.imwrite('../results/vaegan.{}/test.{:03d}.jpg'.format(args.dataset, count), display_box*255.)
		cv2.imwrite('../results/vaegan.{}/gt.{:03d}.jpg'.format(args.dataset, count), display_gt*255.)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

		count += 1

def test(args):
	seed = 0
	rng = np.random.RandomState(seed)
	enc = load_model(args.encoder_model)
	gen = load_model(args.generator_model)
	enc_z = build_test_encoder(enc)
	enc_kl = build_encoder(enc, batch_size=args.batch_size, latent_dimension=args.latent_dimension)

	if (args.dataset == 'celeba'):
		test_steps = count_celeba_data(mode=2, batch_size=args.batch_size)
		images_loader_test = celeba_image_generator(mode=2, rng=rng, batch_size=args.batch_size)
	elif (args.dataset == 'humanm'):
		test_steps = count_humanm_data(video_dir='../data/Human3.6M/test', batch_size=args.batch_size)
		images_loader_test = humanm_image_generator(video_dir='../data/Human3.6M/test', batch_size=args.batch_size)

	list_mse = []
	list_kl = []

	for i in range(test_steps):
		images_batch = next(images_loader_test)
		z = np.squeeze(enc_z.predict_on_batch(images_batch))
		kl = np.mean(np.squeeze(enc_kl.predict_on_batch(images_batch)))

		#Generate image pixels from 0 to 1
		generator_output = ((np.squeeze(gen.predict_on_batch(z)) + 1.0)/2.0)
		images_normalized = (images_batch + 1.0)/2.0

		mse = np.mean((generator_output - images_normalized) ** 2)

		print('MSE: {:.4f} KL: {:.4f}'.format(mse, kl))

		list_mse.append(mse)
		list_kl.append(kl)

	mean_mse = np.mean(np.asarray(list_mse))
	mean_kl = np.mean(np.asarray(list_kl))

	print('Mean mse: {:.4f}'.format(mean_mse))
	print('Mean kl: {:.4f}'.format(mean_kl))

	# enc_loss_avg = 0.

	# for step in range(test_steps):
	# 	images_batch = next(images_loader_test)
	# 	enc_loss = np.squeeze(enc.predict_on_batch(images_batch))
	# 	print(enc_loss)

	# 	# print("Encoder batch {:03d} loss: {:.4f}".format(step, enc_loss))

	# 	# enc_loss_avg = ((enc_loss_avg * step) + enc_loss)/(step + 1)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Test the vaegan network')
	parser.add_argument('-b', '--batch_size', type=int, default=64)
	parser.add_argument('-l', '--latent_dimension', type=int, default=128)
	parser.add_argument('-gm', '--generator_model', type=str, default='../weights/vaegan.gen.050.h5')
	parser.add_argument('-em', '--encoder_model', type=str, default='../weights/vaegan.enc.050.h5')
	parser.add_argument('-ue', '--use_encoder', action='store_true')
	parser.add_argument('-p', '--predict', action='store_true')
	parser.add_argument('-t', '--test', action='store_true')
	parser.add_argument('-d', '--dataset', choices=['celeba', 'humanm'], default='humanm')
	args = parser.parse_args()

	if (args.predict):
		predict(args)

	if (args.test):
		test(args)


