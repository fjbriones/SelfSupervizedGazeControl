import glob
import os
import csv
import cv2
import numpy as np

def image_generator(
	mode,
	images_dir='../data/celeba-dataset/img_align_celeba',
	images_partition_file='../data/celeba-dataset/list_eval_partition.csv',
	batch_size=64,
	image_height=64,
	image_width=64,
	image_channels=3,
	bbox=(40,218-30,15,178-15)):
	#Ensure that the bbox is square
	assert ((bbox[1] - bbox[0]) == (bbox[3] - bbox[2])), "Bounding box is not square"

	while(True):
		images_batch_list = []
		with open(images_partition_file) as csv_file:
			csv_reader = csv.reader(csv_file, delimiter=',')
			line_count = 0
			for row in csv_reader:
				if ((line_count > 0) and (mode == int(row[1]))):
					image_dir = os.path.join(images_dir, row[0])
					image = cv2.imread(image_dir)
					resized_image = cv2.resize(image[bbox[0]:bbox[1], bbox[2]:bbox[3]], (image_height, image_width))
					# cv2.imshow('Data {}'.format(row[0]), resized_image)
					# cv2.waitKey(0)
					# cv2.destroyAllWindows()
					images_batch_list.append(resized_image)
				line_count += 1
				if (len(images_batch_list) == batch_size):
					images_batch_array = (np.array(images_batch_list) - 127.5)/127.5
					images_batch_list = []
					yield images_batch_array

def discriminator_data(
	images_batch,
	rng,
	latent_dimension=128):

	batch_size = images_batch.shape[0]

	z_p = rng.normal(size=(batch_size, latent_dimension))

	y_real = np.ones((batch_size), dtype='float32')
	y_fake = np.zeros((batch_size), dtype='float32')

	return [images_batch, z_p], [y_real, y_fake, y_fake]

def generator_data(
	images_batch,
	rng,
	latent_dimension=128):

	batch_size = images_batch.shape[0]

	z_p = rng.normal(size=(batch_size, latent_dimension))

	y_real = np.ones((batch_size), dtype='float32')
	y_fake = np.zeros((batch_size), dtype='float32')

	return [images_batch, z_p], [y_real, y_real]

def encoder_data(
	images_batch,
	rng,
	latent_dimension=128):

	batch_size = images_batch.shape[0]

	z_p = rng.normal(size=(batch_size, latent_dimension))

	return [images_batch], [z_p]
					



