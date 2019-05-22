import glob
import os
import csv
import cv2
import numpy as np
import random

# def images_list(
# 	mode,
# 	images_partition_file='../data/celeba-dataset/list_eval_partition.csv'):

# 	image_list = []
# 	with open(images_partition_file) as csv_file:
# 		csv_reader = csv.reader(csv_file, delimiter=',')
# 		line_count = 0
# 		for row in csv_reader:
# 			if ((line_count > 0) and (mode == int(row[1]))):
# 				image_list.append([row[0], 0])
# 				# if (mode == 0):
# 				# 	image_list.append([row[0], 1])
# 			line_count += 1
				
# 	# random.shuffle(image_list)

	# return image_list

def celeba_image_generator(
	rng,
	mode,
	images_partition_file='../data/celeba-dataset/list_eval_partition.csv',
	images_dir='../data/celeba-dataset/img_align_celeba',
	batch_size=64,
	image_height=64,
	image_width=64,
	image_channels=3,
	bbox=(40,218-30,15,178-15)):
	#Ensure that the bbox is square
	assert ((bbox[1] - bbox[0]) == (bbox[3] - bbox[2])), "Bounding box is not square"
	while(True):
		with open(images_partition_file) as csv_file:
			csv_reader = csv.reader(csv_file, delimiter=',')
			line_count = 0
			images_batch_list= []
			for row in csv_reader:
				if ((line_count > 0) and (mode == int(row[1]))):
					image_dir = os.path.join(images_dir, row[0])
					image = cv2.imread(image_dir)
					image = cv2.resize(image[bbox[0]:bbox[1], bbox[2]:bbox[3]], (image_height, image_width))
					
					# cv2.imshow('Data {:01d}'.format(item[1]), image)
					# cv2.waitKey(0)
					# cv2.destroyAllWindows()
					images_batch_list.append(image)

				line_count += 1
		
				if (len(images_batch_list) == batch_size):
					images_batch_array = (np.array(images_batch_list)/127.5) - 1.0
					images_batch_array = np.clip(images_batch_array, -1., 1.)
					# rng.shuffle(images_batch_array)
					# cv2.imshow('Data', (images_batch_array[0] + 1.0)/2.0)
					# cv2.waitKey(0)
					# cv2.destroyAllWindows()
					images_batch_list = []
					yield images_batch_array

def humanm_image_generator(
	video_dir,
	batch_size=64,
	image_height=64,
	image_width=64,
	image_channels=3,
	crop_height=224,
	crop_width=112):
	while(True):
		video_frames_dir = os.path.join(video_dir, '*_act_14_*/*.jpg')
		video_frames = glob.glob(video_frames_dir)

		images_batch_list= []

		for frame in video_frames:
			
			img = cv2.imread(frame, cv2.IMREAD_COLOR)

			height = img.shape[0]
			width = img.shape[1]

			#Height start at 0, cut image in half crosswise
			height_start = 0
			#Width start is half the crop width
			width_start = int((width - crop_width)/2)

			cropped_frame = img[height_start:height_start + crop_height, width_start:width_start + crop_width]
			# cv2.imshow('img', cropped_frame)
			# cv2.waitKey(0)
			resized_frame = (cv2.resize(cropped_frame, (image_width, image_height))/127.5) - 1.0

			images_batch_list.append(resized_frame)

			if (len(images_batch_list) == batch_size):
				images_batch_array = np.array(images_batch_list)
				# print(images_batch_array.shape)
				images_batch_array = np.clip(images_batch_array, -1., 1.)
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

	# z_p = rng.normal(size=(batch_size, latent_dimension))

	return [images_batch], None#[z_p]
					



