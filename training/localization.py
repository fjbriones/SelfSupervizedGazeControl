import sys
sys.path.append('../')

from models.savp_models import build_encoder as build_encoder_savp
from models.savp_models import encoder as encoder_savp
from models.vaegan_models import build_encoder as build_encoder_vaegan
from keras.models import load_model
from keras.optimizers import Adam
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.models import Model
from keras.layers import Dense, BatchNormalization, Activation, Flatten, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras.optimizers import Adam
from utils.utils import write_to_video, count_frames
from utils.metrics import calculate_iou, get_iou
from yolo.yolo_functions import get_yolo_indices, draw_prediction
import numpy as np
import time
import cv2
import os
import glob
import argparse
import random
import csv
import copy


def lr_scheduler(epoch, cur_lr):
	new_lr = cur_lr
	if (epoch > 0) and (epoch%10==0):
		new_lr = new_lr / 10
	return new_lr

def get_ssd_indices(ssd, frame, frame_width, frame_height):

	ssd.setInput(cv2.dnn.blobFromImage(frame, size=(300,300), swapRB=True, crop=False))
	ssd_out = ssd.forward()

	boxes = []
	for detection in ssd_out[0,0,:,:]:
		score = float(detection[2])
		class_id = int(detection[1])
		if (score > 0.3) and (class_id==1):
			x = int(detection[3] * frame_width)
			y = int(detection[4] * frame_height)
			w = int(detection[5] * frame_width) - x
			h = int(detection[6] * frame_height) - y
			boxes.append([x, y, w, h])

	return boxes

def get_vaegan_indices(enc, frame, coords, enc_width=64, enc_height=64):

	imgs = np.array([cv2.resize(frame[c[1]:c[1]+c[3], c[0]:c[0]+c[2]], (enc_width, enc_height)) for c in coords])/127.5 - 1.
				
	losses = np.squeeze(enc.predict_on_batch(imgs))
	# print(losses)
	indices = np.argsort(np.squeeze(losses))[:4]
	# print(losses[indices])
	min_loss = np.argmin(np.squeeze(losses))

	return indices, min_loss

def get_savp_indices(enc, frame, prev_frame, coords, enc_width=64, enc_height=64):
	# prev_imgs = np.array([cv2.cvtColor(cv2.resize(prev_frame[c[1]:c[1]+c[3], c[0]:c[0]+c[2]], (enc_width, enc_height)), cv2.COLOR_BGR2GRAY) for c in coords])
	# curr_imgs = np.array([cv2.cvtColor(cv2.resize(frame[c[1]:c[1]+c[3], c[0]:c[0]+c[2]], (enc_width, enc_height)), cv2.COLOR_BGR2GRAY) for c in coords])

	# prev_imgs = np.expand_dims(np.expand_dims(prev_imgs, axis=1), axis=-1)/255.
	# curr_imgs = np.expand_dims(np.expand_dims(curr_imgs, axis=1), axis=-1)/255.

	prev_imgs = np.array([cv2.resize(prev_frame[c[1]:c[1]+c[3], c[0]:c[0]+c[2]], (enc_width, enc_height)) for c in coords])
	curr_imgs = np.array([cv2.resize(frame[c[1]:c[1]+c[3], c[0]:c[0]+c[2]], (enc_width, enc_height)) for c in coords])

	prev_imgs = np.expand_dims(prev_imgs, axis=1)/255.
	curr_imgs = np.expand_dims(curr_imgs, axis=1)/255.

	losses = np.squeeze(enc.predict_on_batch([prev_imgs, curr_imgs]))
	indices = np.argsort(np.squeeze(losses))[:4]
	min_loss = np.argmin(np.squeeze(losses))

	return indices, min_loss

def image_humaneva_generator(video_dir,
	enc_vaegan,
	enc_savp,
	rng,
	save_dir,
	args,
	sample_fps = 10,
	batch_size=2048,
	enc_height=64,
	enc_width=64):

	summary_save_path = os.path.join(save_dir, 'humaneva_summary.csv')

	row = ['Subject', 'Video', 'Mean vaegan min iou', 'Mean vaegan all iou', 'Mean savp min iou', 'Mean savp all iou', 'Mean ssd iou', 'Mean vaegan min cov', 'Mean vaegan all cov', 'Mean savp min cov', 'Mean savp all cov', 'Mean ssd cov']
	with open(summary_save_path, 'w') as csvFile:
		writer = csv.writer(csvFile)
		writer.writerow(row)
	csvFile.close()

	#YOLO initialization
	yolo = cv2.dnn.readNet(args.yolo_weights, args.yolo_config)

	#SSD initialization
	ssd = cv2.dnn.readNet(args.ssd_weights, args.ssd_config)

	subjects = glob.glob(os.path.join(video_dir, 'S*'))
	
	for subject in subjects:
		subject_key = os.path.basename(subject)
		# print(subject_key)
		subject_videos = glob.glob(os.path.join(subject, 'Image_Data/*_(C2).avi'))
		subject_videos.extend(glob.glob(os.path.join(subject, 'Image_Data/*_(C3).avi')))

		for video in subject_videos:
			video_name = os.path.basename(video).split(".")[0]
			# mocap_name = os.path.join(subject, 'Mocap_Data', video_name.split('_(')[0]+'.mat')
			# try:
			# 	mocap = sio.loadmat(mocap_name, squeeze_me=True)
			# except:
			# 	print("No mat file, moving on")
			# 	continue
			# print(sio.whosmat(mocap_name))
			# print(mocap['Markers'])
			# print(mocap['ParameterGroup'])
			cap = cv2.VideoCapture(video)

			frames = []
			list_vaegan_min_iou = []
			list_vaegan_all_iou = []
			list_savp_min_iou = []
			list_savp_all_iou = []
			list_ssd_iou = []
			list_vaegan_min_cov = []
			list_vaegan_all_cov = []
			list_savp_min_cov = []
			list_savp_all_cov = []
			list_ssd_cov = []

			frame_current = 0
			while(cap.isOpened()):
				ret, frame = cap.read()

				if (frame_current == 0):
					prev_frame = copy.deepcopy(frame)
					frame_height = frame.shape[0]
					frame_width = frame.shape[1]
					# print('Dimensions: ({},{})'.format(frame_height, frame_width))
					img_height_max = 380
					img_width_max = int(img_height_max / 2.)
					# img_width_max = 320
					# img_height_min = 80
					# img_width_min= 200
					# img_width_min = int(3 * img_height_min / 5)
					y_max = frame_height - img_height_max 
					x_max = frame_width - img_width_max - 80

					y_min = 0
					x_min = 80
					#frame_current = frame_current + 1
					size = batch_size
					camera_fps = cap.get(cv2.CAP_PROP_FPS)
					frame_sample = int(camera_fps/sample_fps)
					frame_saved = 0

					subject_save_path = os.path.join(save_dir, subject_key)
					if not os.path.exists(subject_save_path):
						os.makedirs(subject_save_path)

					csv_save_path = os.path.join(subject_save_path, video_name + '.csv')
					row = ['Frame', 'Vaegan min iou', 'Vaegan all iou', 'Savp min iou', 'Savp all iou', 'Ssd iou', 'Vaegan min cov', 'Vaegan all cov', 'Savp min cov', 'Savp all cov', 'Ssd cov', 'yolo_x', 'yolo_y', 'yolo_w', 'yolo_h', 'ssd_x', 'ssd_y', 'ssd_w', 'ssd_h', 'vaegan_x', 'vaegan_y', 'vaegan_w', 'vaegan_h', 'savp_x', 'savp_y', 'savp_w', 'savp_h']
					with open(csv_save_path, 'w') as csvFile:
						writer = csv.writer(csvFile)
						writer.writerow(row)
					csvFile.close()
					#continue

				if not ret:
					done = True
					break

				if (frame_current%frame_sample == 0):
					# cv2.imshow('prev', prev_frame)
					###YOLO
					yolo_indices, boxes = get_yolo_indices(yolo, frame, frame_width, frame_height)
					###END yolo

					###SSD
					ssd_boxes = get_ssd_indices(ssd, frame, frame_width, frame_height)
					###END SSD

					y = rng.randint(y_min, y_max+1, size=size)
					x = rng.randint(x_min, x_max+1, size=size)
					# h = rng.randint(img_height_min, img_height_max + 1, size=size)
					# w = rng.randint(img_width_min, img_width_max + 1, size=size)
					h = rng.choice([320, 340, 360, 380], size=size)
					w = (1. * h / 2.).astype(int)
					# h = rng.choice((140, 160, 180, 200, 220, 240, 260), size=size)
					# w = h
					# h = np.repeat(img_height_max, repeats=size)
					# w = np.repeat(img_width_max, repeats=size)

					coords = np.transpose(np.stack((x, y, w, h)))

					vaegan_indices, vaegan_min = get_vaegan_indices(enc_vaegan, frame, coords, enc_width, enc_height)

					savp_indices, savp_min = get_savp_indices(enc_savp, frame, prev_frame, coords, enc_width, enc_height)

					prev_frame = copy.deepcopy(frame)
					
					# imgs = np.array([cv2.resize(frame[c[1]:c[1]+c[3], c[0]:c[0]+c[2]], (enc_width, enc_height)) for c in coords])/127.5 - 1.
					
					# losses = np.squeeze(enc_pred.predict_on_batch(imgs))
					# # print(losses)
					# indices = np.argsort(np.squeeze(losses))[:8]
					# # print(losses[indices])
					# min_loss = np.argmin(np.squeeze(losses))
					# for index in indices:
					# 	cv2.rectangle(frame, (x[index], y[index]), (x[index]+w[index], y[index]+h[index]), (0,255,255), 1)

					#For YOLO
					for i in yolo_indices:
						i = i[0]
						box = boxes[i]
						yolo_x = box[0]
						yolo_y = box[1]
						yolo_w = box[2]
						yolo_h = box[3]
						yolo_box = copy.deepcopy(box)
						draw_prediction(frame, round(yolo_x), round(yolo_y), round(yolo_x+yolo_w), round(yolo_y+yolo_h))

					#For SSD
					for sbox in ssd_boxes:
						ssd_x = sbox[0]
						ssd_y = sbox[1]
						ssd_w = sbox[2]
						ssd_h = sbox[3]
						ssd_box = copy.deepcopy(sbox)
						cv2.rectangle(frame, (ssd_x,ssd_y), (ssd_x+ssd_w,ssd_y+ssd_h), (255,0,0), 2)
						cv2.putText(frame, 'ssd', (ssd_x-10,ssd_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

					#Calculate IoU
					vaegan_min_iou, vaegan_all_iou, vaegan_min_cov, vaegan_all_cov = get_iou(yolo_box, vaegan_indices, coords, vaegan_min)
					savp_min_iou, savp_all_iou, savp_min_cov, savp_all_cov = get_iou(yolo_box, savp_indices, coords, savp_min)
					ssd_iou, ssd_cov = calculate_iou(yolo_box, ssd_box)

					list_vaegan_min_iou.append(vaegan_min_iou)
					list_vaegan_all_iou.append(vaegan_all_iou)
					list_savp_min_iou.append(savp_min_iou)
					list_savp_all_iou.append(savp_all_iou)
					list_ssd_iou.append(ssd_iou)
					list_vaegan_min_cov.append(vaegan_min_cov)
					list_vaegan_all_cov.append(vaegan_all_cov)
					list_savp_min_cov.append(savp_min_cov)
					list_savp_all_cov.append(savp_all_cov)
					list_ssd_cov.append(ssd_cov)					

					j = 0
					for i in vaegan_indices:
						cv2.rectangle(frame, (x[i], y[i]), (x[i]+w[i], y[i]+h[i]), (0,255,255), 2)
						cv2.putText(frame, 'vaegan', (x[i]-10, y[i]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
						j += 1
						vaegan_x = x[i]
						vaegan_y = y[i]
						vaegan_w = w[i]
						vaegan_h = h[i]
						break

					j = 0
					for i in savp_indices:
						cv2.rectangle(frame, (x[i], y[i]), (x[i]+w[i], y[i]+h[i]), (0,0,255), 2)
						cv2.putText(frame, 'savp', (x[i]-10, y[i]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
						j += 1
						savp_x = x[i]
						savp_y = y[i]
						savp_w = w[i]
						savp_h = h[i]
						break

					cv2.putText(frame, 'IoU', (0, frame_height-40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,0), 2)
					cv2.putText(frame, 'VM: {:.2f}'.format(vaegan_min_iou), (50, frame_height-40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,0), 2)
					cv2.putText(frame, 'VA: {:.2f}'.format(vaegan_all_iou), (170, frame_height-40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,0), 2)
					cv2.putText(frame, 'SM: {:.2f}'.format(savp_min_iou), (290, frame_height-40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,0), 2)
					cv2.putText(frame, 'SA: {:.2f}'.format(savp_all_iou), (410, frame_height-40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,0), 2)
					cv2.putText(frame, 'SD: {:.2f}'.format(ssd_iou), (530, frame_height-40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,0), 2)

					cv2.putText(frame, 'Cov', (0, frame_height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,0), 2)
					cv2.putText(frame, 'VM: {:.2f}'.format(vaegan_min_cov), (50, frame_height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,0), 2)
					cv2.putText(frame, 'VA: {:.2f}'.format(vaegan_all_cov), (170, frame_height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,0), 2)
					cv2.putText(frame, 'SM: {:.2f}'.format(savp_min_cov), (290, frame_height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,0), 2)
					cv2.putText(frame, 'SA: {:.2f}'.format(savp_all_cov), (410, frame_height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,0), 2)
					cv2.putText(frame, 'SD: {:.2f}'.format(ssd_cov), (530, frame_height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,0), 2)

					# cv2.putText(frame, "Loss: {:.04f}".format(losses[min_loss]), (0,frame_height -10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 1, cv2.LINE_AA)
					frames.append(frame)

					cv2.imshow(video_name, frame)
					cv2.waitKey(1)

					row = [frame_saved, vaegan_min_iou, vaegan_all_iou, savp_min_iou, savp_all_iou, ssd_iou, vaegan_min_cov, vaegan_all_cov, savp_min_cov, savp_all_cov, ssd_cov, yolo_x, yolo_y, yolo_w, yolo_h, ssd_x, ssd_y, ssd_w, ssd_h, vaegan_x, vaegan_y, vaegan_w, vaegan_h, savp_x, savp_y, savp_w, savp_h]
					with open(csv_save_path, 'a') as csvFile:
						writer = csv.writer(csvFile)
						writer.writerow(row)
					csvFile.close()

					frame_saved += 1

				frame_current += 1

			cv2.destroyAllWindows()
			video_write_name = os.path.join(subject_save_path, video_name + '.avi')

			mean_vaegan_min_iou = np.mean(np.asarray(list_vaegan_min_iou))
			mean_vaegan_all_iou = np.mean(np.asarray(list_vaegan_all_iou))
			mean_savp_min_iou = np.mean(np.asarray(list_savp_min_iou))
			mean_savp_all_iou = np.mean(np.asarray(list_savp_all_iou))
			mean_ssd_iou = np.mean(np.asarray(list_ssd_iou))
			mean_vaegan_min_cov = np.mean(np.asarray(list_vaegan_min_cov))
			mean_vaegan_all_cov = np.mean(np.asarray(list_vaegan_all_cov))
			mean_savp_min_cov = np.mean(np.asarray(list_savp_min_cov))
			mean_savp_all_cov = np.mean(np.asarray(list_savp_all_cov))
			mean_ssd_cov = np.mean(np.asarray(list_ssd_cov))

			row = ['Mean', mean_vaegan_min_iou, mean_vaegan_all_iou, mean_savp_min_iou, mean_savp_all_iou, mean_ssd_iou, mean_vaegan_min_cov, mean_vaegan_all_cov, mean_savp_min_cov, mean_savp_all_cov, mean_ssd_cov]
			with open(csv_save_path, 'a') as csvFile:
				writer = csv.writer(csvFile)
				writer.writerow(row)
			csvFile.close()

			row = [subject_key, video_name, mean_vaegan_min_iou, mean_vaegan_all_iou, mean_savp_min_iou, mean_savp_all_iou, mean_ssd_iou, mean_vaegan_min_cov, mean_vaegan_all_cov, mean_savp_min_cov, mean_savp_all_cov, mean_ssd_cov]
			with open(summary_save_path, 'a') as csvFile:
				writer = csv.writer(csvFile)
				writer.writerow(row)
			csvFile.close()


			frames = np.asarray(frames)/255.0
			# print(frames.shape)
			write_to_video(frames, video_write_name, frame_height, frame_width, int(sample_fps/2.0))
			# print(frame_current)

def image_generic_generator(video_dir,
	enc_pred,
	rng,
	save_dir,
	batch_size=1024,
	enc_height=64,
	enc_width=64):

	# imaged = []

	videos = glob.glob(os.path.join(video_dir, '*'))
	for video in videos:
		print('Reading ' + video)
		video_name = os.path.basename(video).split(".")[0]

		cap = cv2.VideoCapture(video)

		frames = []

		frame_current = 0

		while(cap.isOpened()):
			ret, frame = cap.read()
			# # print(video)
			# cv2.imshow('frame', frame)
			# cv2.waitKey(10)
			if (frame_current == 0):
				prev_frame = frame
				frame_height = frame.shape[0]
				frame_width = frame.shape[1]
				print('Dimensions: ({},{})'.format(frame_height, frame_width))
				img_height_max = 360
				img_width_max = 180#int(3 * img_height_max / 5)
				# img_width_max = 320
				# img_height_min = 80
				# img_width_min= 200
				# img_width_min = int(3 * img_height_min / 5)
				y_max = frame_height - img_height_max
				x_max = frame_width - img_width_max

				y_min = 0
				x_min = 50
				#frame_current = frame_current + 1
				size = batch_size
				camera_fps = cap.get(cv2.CAP_PROP_FPS)
				#continue

			if not ret:
				done = True
				break

			y = rng.randint(y_min, y_max+1, size=size)
			x = rng.randint(x_min, x_max+1, size=size)
			# h = rng.randint(img_height_min, img_height_max + 1, size=size)
			# w = rng.randint(img_width_min, img_width_max + 1, size=size)
			# h = rng.choice([300, 320, 340, 360, 380], size=size)
			# w = (3 * h / 5).astype(int)
			# h = rng.choice((140, 160, 180, 200, 220, 240, 260), size=size)
			# w = h
			h = np.repeat(img_height_max, repeats=size)
			w = np.repeat(img_width_max, repeats=size)

			coords = np.transpose(np.stack((x, y, w, h)))
			
			imgs = np.array([cv2.resize(frame[c[1]:c[1]+c[3], c[0]:c[0]+c[2]], (64, 64)) for c in coords])/127.5 - 1.
			
			losses = np.squeeze(enc_pred.predict_on_batch(imgs))
			# print(losses)
			indices = np.argsort(np.squeeze(losses))[:8]
			print(losses[indices])
			min_loss = np.argmin(np.squeeze(losses))
			for index in indices:
				cv2.rectangle(frame, (x[index], y[index]), (x[index]+w[index], y[index]+h[index]), (0,255,255), 1)


			cv2.rectangle(frame, (x[min_loss], y[min_loss]), (x[min_loss]+w[min_loss], y[min_loss]+h[min_loss]), (0,0,255), 2)
			cv2.putText(frame, "Loss: {:.04f}".format(losses[min_loss]), (0,frame_height -10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 1, cv2.LINE_AA)
			frames.append(frame)

			cv2.imshow(video_name, frame)
			# cv2.imshow('data', image)
			cv2.waitKey(1)

			frame_current += 1

		cv2.destroyAllWindows()
		video_name = os.path.join(save_dir, video_name + '.avi')


		frames = np.asarray(frames)/255.0
		# print(frames.shape)
		write_to_video(frames, video_name, frame_height, frame_width, camera_fps)

def image_bb_generator(video_dir,
	enc_pred,
	rng,
	save_dir,
	batch_size=1024,
	enc_height=64,
	enc_width=64):
	
	# video_files = os.path.join(video_dir, '*.mp4')

	# videos = glob.glob(video_files)

	images = []
	labels = []

	# iteration = 1

	# for video in videos:
	# 	# print(video)
	# 	video_key = os.path.basename(video).split(".")[0]
	# 	cap = cv2.VideoCapture(video)
	# 	subject_key = os.path.dirname(video).split("/")[-1]
	# 	# print(subject_key)
	# 	# print(cap.get(cv2.CAP_PROP_FPS))
		
	# 	video_fps = cap.get(cv2.CAP_PROP_FPS);
	# 	# print(video_fps)
	# 	frame_sample = int(video_fps/camera_fps)
	# 	if (frame_sample) < 1:
	# 		frame_sample = 1

	# 	done = False
	# 	frame_count = 0

	# 	# frames = []

	# 	video_name = os.path.join(save_dir, subject_key + '_' + video_key + '.avi')
	# 	print('Processing {} {}'.format(subject_key, video_key))
	sequences = glob.glob(args.video_dir)
	for seq in sequences:
		print('Reading ' + seq)
		seq_name = os.path.basename(seq)
		video = os.path.join(seq, 'Video/' + seq_name + '_CAM1.mp4')
		label = os.path.join(seq, 'GroundTruth/face_bb.txt')
		f = open(label, "r")
		f_lines = f.readlines()
		i = 0
		cap = cv2.VideoCapture(video)
		frame_number = int(f_lines[i].split(",")[0])

		frames = []

		frame_current = 0

		list_iou = []
		list_cov = []

		while(cap.isOpened()):
			ret, frame = cap.read()
			# # print(video)
			# cv2.imshow('frame', frame)
			# cv2.waitKey(10)
			if (frame_current == 0):
				prev_frame = frame
				frame_height = frame.shape[0]
				frame_width = frame.shape[1]
				img_height_max = 70
				img_width_max = 70#int(3 * img_height_max / 5)
				# img_width_max = 320
				# img_height_min = 80
				# img_width_min= 200
				# img_width_min = int(3 * img_height_min / 5)
				y_max = frame_height - img_height_max
				x_max = frame_width - img_width_max

				y_min = 0
				x_min = 0
				#frame_current = frame_current + 1
				size = batch_size
				camera_fps = cap.get(cv2.CAP_PROP_FPS)
				#continue

				csv_save_path = os.path.join(save_dir, seq_name + '.csv')
				row = ['Frame', 'IoU', 'Cov', 'Loss']
				with open(csv_save_path, 'w') as csvFile:
					writer = csv.writer(csvFile)
					writer.writerow(row)
				csvFile.close()

			if not ret:
				done = True
				break

			frame_number = int(f_lines[i].split(",")[0])
			
			while frame_current == frame_number:
				x_gt = int(float(f_lines[i].split(",")[2]))
				y_gt = int(float(f_lines[i].split(",")[3]))
				w_gt = int(float(f_lines[i].split(",")[4]))
				h_gt = int(float(f_lines[i].split(",")[5]))

				# print(x_gt)

				# x_gt = max(0, int(x_gt - 0.3*w_gt))

				# # print(x_gt)
				# y_gt = max(0, int(y_gt - 0.3*h_gt))
				# w_gt = int(1.3*w_gt)
				# h_gt = int(1.3*h_gt)


				cv2.rectangle(frame, (x_gt, y_gt), (x_gt+w_gt, y_gt+h_gt), (0,255,0), 2)
				i += 1
				if i < len(f_lines):
					frame_number = int(f_lines[i].split(",")[0])
				else:
					frame_number = 0
					i = 0

			# if frame_count % frame_sample == 0:
				image = cv2.resize(frame, (224,224))
				
				# a = time.clock()

				y = rng.randint(y_min, y_max+1, size=size)
				x = rng.randint(x_min, x_max+1, size=size)
				# h = rng.randint(img_height_min, img_height_max + 1, size=size)
				# w = rng.randint(img_width_min, img_width_max + 1, size=size)
				# h = rng.choice([300, 320, 340, 360, 380], size=size)
				# w = (3 * h / 5).astype(int)
				# h = rng.choice((200, 220, 240, 260, 280, 300, 320), size=size)
				# w = h
				h = np.repeat(70, repeats=size)
				w = np.repeat(70, repeats=size)

				coords = np.transpose(np.stack((x, y, w, h)))
				# print(coords.shape)
				# gt_coords = np.expand_dims(np.transpose(np.array((x_gt, y_gt, w_gt, h_gt))), axis=0)

				# coords = np.vstack((coords, gt_coords))
				# print(coords.shape)
				# print(gt_coords.shape)

				# prev_imgs = np.array([cv2.cvtColor(cv2.resize(prev_frame[c[1]:c[1]+c[3], c[0]:c[0]+c[2]], (64, 64)), cv2.COLOR_BGR2GRAY) for c in coords])
				# curr_imgs = np.array([cv2.cvtColor(cv2.resize(frame[c[1]:c[1]+c[3], c[0]:c[0]+c[2]], (64, 64)), cv2.COLOR_BGR2GRAY) for c in coords])

				# prev_imgs = np.expand_dims(np.expand_dims(prev_imgs, axis=1), axis=-1)/255.
				# curr_imgs = np.expand_dims(np.expand_dims(curr_imgs, axis=1), axis=-1)/255.

				# prev_imgs = np.array([cv2.resize(prev_frame[c[1]:c[1]+c[3], c[0]:c[0]+c[2]], (64, 64)) for c in coords])
				# curr_imgs = np.array([cv2.resize(frame[c[1]:c[1]+c[3], c[0]:c[0]+c[2]], (64, 64)) for c in coords])

				# prev_imgs = np.expand_dims(prev_imgs, axis=1)/255.
				# curr_imgs = np.expand_dims(curr_imgs, axis=1)/255.

				imgs = np.array([cv2.resize(frame[c[1]:c[1]+c[3], c[0]:c[0]+c[2]], (64, 64)) for c in coords])/127.5 - 1.
				
				# losses = np.squeeze(enc_pred.predict_on_batch([prev_imgs, curr_imgs]))
				losses = np.squeeze(enc_pred.predict_on_batch(imgs))
				# print(losses)
				indices = np.argsort(np.squeeze(losses))[:8]
				min_loss = np.argmax(np.squeeze(losses))

				gt_img = np.expand_dims(cv2.resize(frame[y_gt:y_gt+h_gt, x_gt:x_gt+w_gt], (64,64)), axis=0)/127.5 - 1.
				gt_loss = np.squeeze(enc_pred.predict_on_batch(gt_img))
				# loss_sum = np.sum(np.array([1/losses[i] for i in indices]))

				# y_mean = np.sum(np.array([y[i]/losses[i] for i in indices]))/loss_sum
				# x_mean = np.sum(np.array([x[i]/losses[i] for i in indices]))/loss_sum
				# h_mean = np.sum(np.array([h[i]/losses[i] for i in indices]))/loss_sum
				# w_mean = np.sum(np.array([w[i]/losses[i] for i in indices]))/loss_sum

				# y_std = np.std(np.array([y[i] for i in indices]))
				# x_std = np.std(np.array([x[i] for i in indices]))
				# h_std = np.std(np.array([h[i] for i in indices]))
				# w_std = np.std(np.array([w[i] for i in indices]))

				# for index in indices:
				# 	cv2.rectangle(frame, (x[index], y[index]), (x[index]+w[index], y[index]+h[index]), (0,255,255), 1)

				gt_box = [x_gt, y_gt, w_gt, h_gt]
				pred_box = [x[min_loss], y[min_loss], w[min_loss], h[min_loss]]

				iou, cov = calculate_iou(gt_box, pred_box)
				
				list_iou.append(iou)
				list_cov.append(cov)

				cv2.rectangle(frame, (x[min_loss], y[min_loss]), (x[min_loss]+w[min_loss], y[min_loss]+h[min_loss]), (0,255,255), 2)
				# cv2.rectangle(frame, (int(x_mean), int(y_mean)), (int(x_mean+w_mean), int(y_mean+h_mean)), (255,0,0), 2)
				# cv2.putText(frame, "Loss: {:.04f}".format(losses[min_loss]), (0,frame_height -10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 1, cv2.LINE_AA)
				# cv2.putText(frame, "GT Loss: {:.04f}".format(gt_loss), (300,frame_height -10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 1, cv2.LINE_AA)

				# cv2.putText(frame, 'Loss: {:.2f}'.format(losses[min_loss]), (0, frame_height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,0), 2)
				# cv2.putText(frame, 'GT Loss: {:.2f}'.format(gt_loss), (200, frame_height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,0), 2)
				
				label = np.array([x[min_loss], y[min_loss], w[min_loss], h[min_loss], x_max, y_max])

				images.append(image)
				labels.append(label)

				cv2.putText(frame, 'IoU: {:.2f}'.format(iou), (0, frame_height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,0), 2)
				cv2.putText(frame, 'Cov: {:.2f}'.format(cov), (150, frame_height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,0), 2)

				row = [frame_number, iou, cov, losses[min_loss]]
				with open(csv_save_path, 'a') as csvFile:
					writer = csv.writer(csvFile)
					writer.writerow(row)
				csvFile.close()
				

				frames.append(frame)

				# size = 256

				# y = rng.normal(y_mean, y_std, size=size).astype(int)
				# x = rng.normal(x_mean, x_std, size=size).astype(int)
				# h = rng.normal(h_mean, h_std, size=size).astype(int)
				# w = rng.normal(w_mean, w_std, size=size).astype(int)

				# b = time.clock()
				# print(b-a)
				# print(losses.shape)
				
				# for img in curr_imgs:
				# 	cv2.imshow('img', img)
				# 	cv2.waitKey(0)

			cv2.imshow(seq_name, frame)
			# cv2.imshow('data', image)
			cv2.waitKey(1)

			prev_frame = frame
			frame_current += 1
				# yield frame, done
			# frame_count = frame_count + 1

		cv2.destroyAllWindows()

		mean_iou = np.mean(np.asarray(list_iou))
		mean_cov = np.mean(np.asarray(list_cov))

		row = ['Mean', mean_iou, mean_cov]
		with open(csv_save_path, 'a') as csvFile:
			writer = csv.writer(csvFile)
			writer.writerow(row)
		csvFile.close()

		print("Mean IoU of {} is {:04f}".format(seq_name, mean_iou))
		seq_name += '_' + str(mean_iou)

		video_name = os.path.join(save_dir, seq_name + '.avi')


		frames = np.asarray(frames)/255.0
		print(frames.shape)
		write_to_video(frames, video_name, frame_height, frame_width, camera_fps)

	# 	if (len(images) > 10000):
	# 		#Shuffle images and labels
	# 		comb = list(zip(images, labels))
	# 		random.shuffle(comb)
	# 		images, labels = zip(*comb)


	# 		#Convert the list to arrays
	# 		images = np.array(images)
	# 		labels = np.array(labels)
	# 		print(images.shape)
	# 		print(labels.shape)

	# 		file_name = os.path.join(save_dir, 'train_{}'.format(iteration))

	# 		print("Saving npz file to {}".format(file_name))
	# 		np.savez_compressed(file_name, images=images, labels=labels)

	# 		images = []
	# 		labels = []
	# 		iteration = iteration + 1

	# if (len(images) > 0):
	# 	#Shuffle images and labels
	# 	comb = list(zip(images, labels))
	# 	random.shuffle(comb)
	# 	images, labels = zip(*comb)

	# 	#Convert the list to arrays
	# 	images = np.array(images)
	# 	labels = np.array(labels)
	# 	print(images.shape)
	# 	print(labels.shape)

	# 	file_name = os.path.join(save_dir, 'train')

	# 	print("Saving npz file to {}".format(file_name))
	# 	np.savez_compressed(file_name, images=images, labels=labels)

	# 	images = []
	# 	labels = []

def create_model():
	mobilenet  = MobileNetV2(input_shape=(224,224,3), include_top=False, pooling='avg')

	# x = GlobalAveragePooling2D()()
	# x = Flatten()(x)
	x = Dense(1024)(mobilenet.outputs[0])
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	x = Dense(3)(x)
	x = Activation('sigmoid')(x)

	return Model(inputs=mobilenet.inputs, outputs=x)

def train(model, epochs, dataset_file, batch_size=64):
	# j = 0
	# for i in range(epochs):
	# 	#look at the dataset
	# 	dataset_files = glob.glob(dataset)
	# 	for dataset_file in dataset_files:
	data = np.load(dataset_file)
	images = data['images']/255.0
	x = data['labels'][:,0] / data['labels'][:,4]
	y = data['labels'][:,1] / data['labels'][:,5]
	s = (data['labels'][:,3] - 300)/80
	labels = np.transpose(np.vstack((x, y, s)))

	callback_lrscheduler = LearningRateScheduler(lr_scheduler, verbose=1)
	callback_checkpoint = ModelCheckpoint('../weights/models/localization_{epoch:02d}_{val_loss:2f}.h5', monitor='val_loss')
	callback_tensorboard = TensorBoard(batch_size=batch_size)

	callbacks = [callback_lrscheduler, callback_checkpoint, callback_tensorboard]

	model.fit(x=images, y=labels, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=callbacks)
	images = 0
	data = 0

	model.save('models/localization.h5')

def predict(model, video_files, camera_fps=10):
	video_dir = glob.glob(video_files)

	img_height_max = 380
	img_width_max = int(img_height_max * 3.0 / 5.0)
	print(img_width_max)

	for video_file in video_dir:
		video_key = os.path.basename(video_file).split(".")[0]
		subject_key = os.path.dirname(video_file).split("/")[-1]
		cap = cv2.VideoCapture(video_file)
		video_fps = cap.get(cv2.CAP_PROP_FPS);
		# print(video_fps)
		wait_time = int(1000/video_fps)
		frames = []
		frame_count = 0

		frame_sample = int(video_fps/camera_fps)
		if (frame_sample) < 1:
			frame_sample = 1

		while(cap.isOpened()):
			ret, frame = cap.read()

			if not ret:
				break

			frame_height = frame.shape[0]
			frame_width = frame.shape[1]

			if frame_count % frame_sample == 0:

				# a = time.clock()
				image_input = np.expand_dims(cv2.resize(frame, (224,224)),axis=0)
				output = model.predict_on_batch(image_input)
				# b = time.clock()

				# print("Prediction time is {}".format(b-a))

				x = int(np.squeeze(output)[0] * (frame_width - img_width_max))
				y = int(np.squeeze(output)[1] * (frame_height - img_height_max))
				h = int((np.squeeze(output)[2] * 80.0) + 300.0)
				w = int(h * 3.0 / 5.0)

				cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)
				cv2.imshow('res', frame)
				cv2.waitKey(5)

				frames.append(frame)

			frame_count = frame_count + 1

		frames = np.array(frames)/255.0
		result_file = os.path.join('results', subject_key + '_' + video_key + '_annotated.avi')

		write_to_video(frames, result_file, frame_height, frame_width, camera_fps)

def capture(model, camera_fps=10):

	cap = cv2.VideoCapture(0)

	video_fps = cap.get(cv2.CAP_PROP_FPS);
	
	img_height_max = 380
	img_width_max = int(img_height_max * 3.0 / 5.0)

	frame_sample = int(video_fps/camera_fps)
	if (frame_sample) < 1:
		frame_sample = 1

	frame_count = 0

	while(cap.isOpened()):
		ret, frame = cap.read()

		frame_height = frame.shape[0]
		frame_width = frame.shape[1]

		# print(frame_height)
		# print(frame_width)

		# if frame_count % frame_sample == 0:
			# a = time.clock()
		image_input = np.expand_dims(cv2.resize(frame, (224,224)), axis=0)
		output = model.predict_on_batch(image_input)
		# b = time.clock()

		# print("Prediction time is {}".format(b-a))

		 # - img_width_max
		# - img_height_max

		x = int(np.squeeze(output)[0] * (frame_width - img_width_max))
		y = int(np.squeeze(output)[1] * (frame_height - img_height_max))
		h = int((np.squeeze(output)[2] * 80.0) + 300.0)
		w = int(h * 0.6)

		print('Coordinates {} {} {}'.format(x, y, h))

		cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)
		cv2.imshow('res', frame)
		cv2.waitKey(5)
		frame_count = 0

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Train the network')
	parser.add_argument('-vd', '--video_dir', type=str, default='../data/HumanEvaI')
	parser.add_argument('-s', '--save_dir', type=str, default='../results/localization')
	parser.add_argument('-w', '--weights_file', type=str, default='../weights/localization_working.h5')
	parser.add_argument('-g', '--generate_data', action='store_true')
	parser.add_argument('-t', '--train', action='store_true')
	parser.add_argument('-p', '--predict', action='store_true')
	parser.add_argument('-c', '--capture', action='store_true')
	parser.add_argument('-d', '--dataset', choices=['avdiar', 'humaneva'], default='humaneva')
	parser.add_argument('-yc', '--yolo_config', type=str, default='../yolo/yolov3.cfg')
	parser.add_argument('-yw', '--yolo_weights', type=str, default='../yolo/yolov3.weights')
	parser.add_argument('-ycl', '--yolo_classes', type=str, default='../yolo/yolov3.txt')
	parser.add_argument('-sc', '--ssd_config', type=str, default='../ssd/ssd_mobilenet_v2.pbtxt')
	parser.add_argument('-sw', '--ssd_weights', type=str, default='../ssd/ssd_mobilenet_v2.pb')
	parser.add_argument('-v', '--predict_video', type=str, default='data/HumanEvaI/*/*.avi')
	parser.add_argument('-b', '--batch_size', type=int, default=2048)
	parser.add_argument('-e', '--epochs', type=int, default=100)
	args = parser.parse_args()

	if args.generate_data:
		# enc_s = encoder_savp(time=1, latent_dim=8, frame_width=64, frame_height=64, frame_channels=1, batch_size=None)
		# enc_s.load_weights('../weights/enc.whole.bw.h5')
		enc_s = load_model('../weights/enc.025.h5')
		enc_savp = build_encoder_savp(enc_s, time=1, latent_dim=8, frame_width=64, frame_height=64, frame_channels=3, batch_size=None)
		enc_savp.summary()
		# enc = encoder(batch_size=None)
		# enc.load_weights('../weights/vaegan.enc.076.h5')
		# enc_v = load_model('../weights.old/vaegan.enc.076.h5')
		enc_v = load_model('../weights/vaegan.enc.031.h5')
		enc_vaegan = build_encoder_vaegan(enc_v, batch_size=None)
		# enc = load_model('models/enc.050.h5')
		# enc_pred = build_encoder(enc, time=1, latent_dim=8, frame_width=64, frame_height=64, frame_channels=1, batch_size=None)
		# enc_pred.compile(Adam(), 'mean_absolute_error')
		enc_vaegan.summary()

		rng = np.random.RandomState(0)
		if (args.dataset=='avdiar'):
			image_bb_generator(args.video_dir, enc_vaegan, rng, args.save_dir, batch_size=args.batch_size)
		elif (args.dataset=='humaneva'):
			image_humaneva_generator(args.video_dir, enc_vaegan, enc_savp, rng, args.save_dir, args, batch_size=args.batch_size)

	model = create_model()
	model.compile(Adam(), loss='mse')
	model.summary()

	if args.train:
		dataset_file = os.path.join(args.save_dir, '1.npz')
		train(model, args.epochs, dataset_file)
	else:
		model.load_weights(args.weights_file)

	if args.predict:
		predict(model, args.predict_video)

	if args.capture:
		capture(model)

	# while(True):
		# next(data_gen)