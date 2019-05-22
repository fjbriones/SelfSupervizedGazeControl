import cv2
import os
import glob
import csv

def write_to_video(frames, filename, frame_height, frame_width, video_fps):
	# fourcc = 0x00000021
	# fourcc = cv2.VideoWriter_fourcc(*'X264')
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	# print(video_fps)
	# print(frame_height)
	# print(frame_width)
	### I still don't get why opencv uses a size dimensionality of (width, height) while the rest uses (height, width)
	video = cv2.VideoWriter(filename, fourcc, video_fps, (frame_width,frame_height), True)
	for i in range(frames.shape[0]):
		# print(frames[i].shape)
		frame = (frames[i]*255.).astype(int)
		# print(frame)
		# frame.astype(int)
		# cv2.imshow('frame', frame)
		# cv2.waitKey(25)
		cv2.imwrite('tmp.jpg', frame)
		frame_read = cv2.imread('tmp.jpg')
		# cv2.imshow('read frame', frame_read)
		# cv2.waitKey(25)
		# print(frame_sread.shape)
		video.write(frame_read)

	video.release()
	print("Wrote to ", filename)

def make_dir(dir):
	if not os.path.exists(dir):
		os.makedirs(dir)

def count_frames(video_dir, batch_size, camera_fps, time):
	videos = glob.glob(video_dir)
	frame_count = 0
	total_frames = 0
	for video in videos:
		cap = cv2.VideoCapture(video)
		video_fps = cap.get(cv2.CAP_PROP_FPS)
		frame_sample = int(video_fps/camera_fps)
		if frame_sample < 1:
			frame_sample = 1	
		i = 0
		j = 0
		while(cap.isOpened()):
			ret, frame = cap.read()
			# print(video)
			if not ret:
				break
			total_frames = total_frames + 1
			if j % frame_sample == 0:
				i = i + 1
				if i % (2*time) == 0:
					i = 0
					frame_count = frame_count + 1
			j = j + 1
	steps = int(frame_count/batch_size)
	# print(total_frames)
	# print(steps)
	# print(frame_sample)
	return steps

def count_images(images_dir, batch_size, time):
	images = glob.glob(images_dir)
	steps = int(len(images)/(2*time*batch_size))
	return steps

def count_celeba_data(
	mode,
	images_partition_file='../data/celeba-dataset/list_eval_partition.csv',
	batch_size=64):
	with open(images_partition_file) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		line_count = 0
		count = 0
		for row in csv_reader:
			if ((line_count > 0) and (mode == int(row[1]))):
				count += 1
			line_count += 1
		return int(count/batch_size)

def count_humanm_data(
	video_dir, 
	batch_size=64):
	video_folders_dir = os.path.join(video_dir, '*_act_14_*/*.jpg')
	video_frames = glob.glob(video_folders_dir)
	count = len(video_frames)
	return int(count/batch_size)


def set_trainable(model, trainable):
	model.trainable = trainable
	for layer in model.layers:
		layer.trainable = trainable

