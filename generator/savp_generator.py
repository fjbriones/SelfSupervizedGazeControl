from utils import write_to_video, make_dir
import cv2
import youtube_dl
import json
import argparse
import os
import glob
import numpy as np

def video_generator(video_dir, 
	frame_height=64, 
	frame_width=64, 
	frame_channels=3, 
	batch_size=1, 
	time=10, 
	camera_fps=2,
	crop_height=224,
	crop_width=112,
	json_files="dataset_json", 
	json_filename="kinetics_train.json"):
	while(True):
		video_folders_dir = os.path.join(video_dir, '*_act_14_*')
		video_folders = glob.glob(video_folders_dir)
		# print("Number of data ". len(videos))
		# # print(videoss_files)
		# previous_frames = np.empty((batch_size, time, frame_height, frame_width, frame_channels))
		current_frames = np.empty((batch_size, time, frame_height, frame_width, frame_channels))
		next_frames = np.empty((batch_size, time, frame_height, frame_width, frame_channels))
		batch_count = 0
		for video_folder in video_folders:
			key  = os.path.basename(video_folder)
			# print(key)
			i = 0
			key_frame_id = 1;
			img_filename = video_folder + "/" + key + "_{:06d}.jpg".format(key_frame_id)
			# print(img_filename)
			img = cv2.imread(img_filename, cv2.IMREAD_COLOR)
			
			while img is not None:
				height = img.shape[0]
				width = img.shape[1]
				#Height start at 0, cut image in half crosswise
				height_start = 0
				#Width start is half the crop width
				width_start = int((width - crop_width)/2)
				# print(width_start)
				# print(img.shape)
				cropped_frame = img[height_start:height_start + crop_height, width_start:width_start + crop_width]
				# cv2.imshow('cropped', cropped_frame)
				# cv2.waitKey(0)
				resized_frame = cv2.resize(cropped_frame, (frame_width, frame_height))/255.
				if frame_channels == 1:
					resized_frame = np.expand_dims(resized_frame, axis=2)
				if i == 0:
					current_frames[batch_count,i,:,:,:] = resized_frame
				elif i < time:
					current_frames[batch_count,i,:,:,:] = resized_frame
					next_frames[batch_count,i-1,:,:,:] = resized_frame
				else:
					next_frames[batch_count,i-1,:,:,:] = resized_frame
					if i == time:
						i = -1
						batch_count = batch_count + 1
						if batch_count % batch_size == 0:
							yield current_frames, next_frames
							# cv2.destroyAllWindows()
							batch_count = 0
							current_frames = np.empty_like(current_frames)
							next_frames = np.empty_like(next_frames)
				i += 1
				key_frame_id += 5
				img_filename = video_folder + "/" + key + "_{:06d}.jpg".format(key_frame_id)
				# print(img_filename)
				img = cv2.imread(img_filename, cv2.IMREAD_COLOR)

		# for video in videos:
		# 	# print(video)
		# 	video_key = os.path.basename(video).split(".")[0]
		# 	# print(video_key)
		# 	# json_file = os.path.join(json_files, json_filename)
		# 	# with open(json_file) as f:
		# 	# 	data = json.load(f)
		# 	# if data[key]['annotations']['label'] == "walking the dog":
		# 	# segment = data[video_key]["annotations"]["segment"]
		# 	# print(segment)
		# 	# print(video)
		# 	cap = cv2.VideoCapture(video)
		# 	# print(cap.get(cv2.CAP_PROP_FPS))
			
		# 	video_fps = cap.get(cv2.CAP_PROP_FPS);
		# # 	# print(video_fps)
		# 	frame_sample = int(video_fps/camera_fps)
		# 	if (frame_sample) < 1:
		# 		frame_sample = 1

		# 	# print(frame_sample)
		# video_folders_dir = os.path.join(video_dir, '*.avi')
		# videos = glob.glob(video_folders_dir)
		# for video in videos:
		# 	i = 0
		# 	frame_count = 0
		# 	cap = cv2.VideoCapture(video)
		# 	video_fps = cap.get(cv2.CAP_PROP_FPS);
		# 	frame_sample = int(video_fps/camera_fps)
		# 	if (frame_sample) < 1:
		# 		frame_sample = 1
		# 	current_frames = np.empty((batch_size, 2*time - 1, frame_height, frame_width, frame_channels))
		# 	next_frames = np.empty((batch_size, 2*time - 1, frame_height, frame_width, frame_channels))
		# 	batch_count = 0
		# 	while(cap.isOpened()):
		# 		ret, frame = cap.read()
		# 		# print(video)
		# 		if not ret:
		# 			break
		# 		###loader for kth
		# 		if frame_count % frame_sample == 0:
		# 			resized_frame = (cv2.cvtColor(cv2.resize(frame, (frame_width, frame_height)), cv2.COLOR_BGR2GRAY)-127.5)/127.5
		# 			resized_frame = np.expand_dims(resized_frame, axis=4)
		# 			if i == 0:
		# 				current_frames[batch_count,i,:,:,:] = resized_frame
		# 			elif i < 2*time - 1:
		# 				current_frames[batch_count,i,:,:,:] = resized_frame
		# 				next_frames[batch_count,i-1,:,:,:] = resized_frame
		# 			else:
		# 				next_frames[batch_count,i-1,:,:,:] = resized_frame
		# 				if i == 2*time - 1:
		# 					i = -1
		# 					batch_count = batch_count + 1
		# 					if batch_count % batch_size == 0:
		# 						yield current_frames, next_frames
		# 						batch_count = 0
		# 						current_frames = np.empty_like(current_frames)
		# 						next_frames = np.empty_like(next_frames)
		# 			i = i + 1			
		# 		frame_count = frame_count + 1

def discriminator_loader(video_generator, latent_dim=8, seed=0, time_init=10):
	rng = np.random.RandomState(seed)
	while True:
		x, y = next(video_generator)
		# print("Discriminator")
		batch_size = y.shape[0]
		time = y.shape[1]
		z_p = rng.normal(size=(batch_size, time, latent_dim))

		y_real = np.ones((batch_size), dtype='float32')
		y_fake = np.zeros((batch_size), dtype='float32')

		yield [x[:,0], x[:,0:time_init], y, x, z_p], [y_real, y_real, y_fake, y_fake]

def discriminator_data(x, y, latent_dim=8, seed=0, time_init=10):
	rng = np.random.RandomState(seed)

	batch_size = y.shape[0]
	time = y.shape[1]
	z_p = rng.normal(size=(batch_size, time, latent_dim))

	y_real = np.ones((batch_size), dtype='float32')
	y_fake = np.zeros((batch_size), dtype='float32')

	return [x[:,0], x[:,0:time_init], y, x, z_p], [y_real, y_real, y_fake, y_fake]

def generator_loader(video_generator, latent_dim=8, seed=0, eval=False, time_init=10):
	rng = np.random.RandomState(seed)
	while True:
		x, y = next(video_generator)
		# print("Generator")
		batch_size = y.shape[0]
		time = y.shape[1]
		z_p = rng.normal(size=(batch_size, time, latent_dim))

		y_real = np.ones((batch_size), dtype='float32')

		# y_real = np.zeros((batch_size), dtype='float32')
		# if eval:
		yield [x[:,0], x[:,0:time_init], y, x, z_p], [y_real, y_real, y]
		# else:
			# yield [x0[:,0], np.expand_dims(x0[:,0], axis=1), z_p], [x[:,0], x, y, z_p], [y_real, y_real, y]

def generator_data(x, y, latent_dim=8, seed=0, time_init=10):
	rng = np.random.RandomState(seed)

	batch_size = y.shape[0]
	time = y.shape[1]
	z_p = rng.normal(size=(batch_size, time, latent_dim))

	y_real = np.ones((batch_size), dtype='float32')

	# y_real = np.zeros((batch_size), dtype='float32')

	return [x[:,0], x[:,0:time_init], y, x, z_p], [y_real, y_real, y]


def vaegan_loader(video_generator, latent_dim=8, seed=0, time_init=10):
	rng = np.random.RandomState(seed)
	while True:
		x, y = next(video_generator)
		batch_size = y.shape[0]
		time = y.shape[1]
		z_p = rng.normal(size=(batch_size, time, latent_dim))

		yield [x[:,0], x, z_p], y

def encoder_loader(video_generator, latent_dim=8, seed=0, time_init=10):
	rng = np.random.RandomState(seed)
	while True:
		x, y = next(video_generator)
		batch_size = y.shape[0]
		time = y.shape[1]
		z_p = rng.normal(size=(batch_size, time, latent_dim * 2))
		yield [x[:,0], x[:,0:time_init], x, y], [z_p, y]

def encoder_data(x, y, latent_dim=8, seed=0, time_init=10):
	rng = np.random.RandomState(seed)

	batch_size = y.shape[0]
	time = y.shape[1]
	z_p = rng.normal(size=(batch_size, time, latent_dim * 2))
	return [x[:,0], x[:,0:time_init], x, y], [z_p, y]

def download_videos(directory_json, output_dir, label):
	#Read all .json files
	new_directory = os.path.join(directory_json, '*.json')
	json_files = glob.glob(new_directory)
	# print(json_files)
	for json_file in json_files:
		folder_name = os.path.join(output_dir, os.path.split(json_file)[1].split('.')[0])
		# print(folder_name)
		with open(json_file) as f:
			data = json.load(f)
		for key in data.keys():
			if data[key]['annotations']['label'] == label:
				# print(data[key]['url'])
				output_file = os.path.join(folder_name, key+'.mp4')
				video_url = data[key]['url']
				ydl_opts = {
					'outtmpl': output_file,
					'format': '160'
				}
				ydl = youtube_dl.YoutubeDL(ydl_opts)
				try:
					info_dict = ydl.extract_info(video_url, download=True)
				except:
					print('Unable to download: ', video_url)
					continue
			# print(info_dict)

def crop_videos(dataset_json, input_dir, output_dir, dataset="kinetics_train"):
	# video_files = os.path.join(directory_json, directory_json_files)
	json_file = os.path.join(dataset_json, dataset+".json")
	with open(json_file) as f:
		data = json.load(f)
	video_file_dir = os.path.join(input_dir, dataset, "*.mp4")
	video_files = glob.glob(video_file_dir)
	print(video_file_dir)
	for video in video_files:
		print("Reading ", video)
		video_key = os.path.basename(video).split(".")[0]
		segment = data[video_key]["annotations"]["segment"]
		# print(segment)
		cap = cv2.VideoCapture(video)
		# print(cap.get(cv2.CAP_PROP_FPS))
		
		video_fps = cap.get(cv2.CAP_PROP_FPS)
		video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		time = int((segment[1] - segment[0]) * video_fps)

		frames = np.empty((time, video_height, video_width, 3))

		i = 0
		frame_count = -1;
		j = 0
		while(cap.isOpened()):
			ret, frame = cap.read()
			# print(video)
			if not ret:
				break

			frame_count = frame_count + 1

			current_time = frame_count / video_fps

			if current_time < segment[0]:
				# j = 0
				# print("Current time ", current_time)
				continue
			else:
				if j == time:
					print("Breaking")
					break
				# cv2.imshow('frame1', frame)
				# cv2.waitKey(25)
				frames[j] = frame
				j = j + 1
			# else:
			# 	break

		save_dir = os.path.join(output_dir, dataset)
		make_dir(save_dir)
		output_file = os.path.join(output_dir, dataset, video_key + ".avi")
		write_to_video(frames, output_file, video_height, video_width, video_fps)

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Download videos from youtube')
	parser.add_argument('-d', '--dataset_json', type=str, default='dataset_json')
	parser.add_argument('-o', '--output_dir', type=str, default='/media/fjbriones/Vorcha/Datasets/Kinetics-600')
	parser.add_argument('-ol', '--output_dir_local', type=str, default='data')
	parser.add_argument('-l', '--label', type=str, default='news anchoring')
	args = parser.parse_args()

	# download_videos(args.dataset_json, args.output_dir, args.label)

	crop_videos(args.dataset_json, args.output_dir, args.output_dir_local)
	crop_videos(args.dataset_json, args.output_dir, args.output_dir_local, dataset="kinetics_val")
	crop_videos(args.dataset_json, args.output_dir, args.output_dir_local, dataset="kinetics_test")


