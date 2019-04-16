from data_generator import video_generator, generator_loader, vaegan_loader, encoder_loader
from models import trial_predictor, encoder, generator, build_graph
from models import discriminator
from keras.models import Model, load_model
from keras.layers import Input
from keras.optimizers import RMSprop
from train import set_trainable
from losses import kl_loss
from utils import write_to_video, count_frames, count_images
import numpy as np
import argparse
import cv2
import glob
import os

def main_trial(args):
	test_gen = video_generator(args.test_directory, args.batch_size)

	inputs = Input(shape=(args.time, args.frame_height, args.frame_width, args.frame_channels))
	outputs = trial_predictor(inputs)

	model = Model(inputs=inputs, outputs=outputs)
	model.compile(optimizer='rmsprop',
		loss='mean_squared_error',
		metrics=['accuracy'])
	model.summary()

	print('Loading weights at ' + args.weights_directory)
	model.load_weights(args.weights_directory)

	loss, accuracy = model.evaluate_generator(test_gen,
		steps=98)
	print(loss)

	predict_gen = video_generator(args.test_directory)
	for j in range(len(glob.glob(os.path.join(args.test_directory,'*.mp4')))):
		frames, next_actual_frames = next(predict_gen)
		next_pred_frames = model.predict(frames, batch_size=args.batch_size)

		# next_pred_frames = np.empty_like(frames)

		# for i in range(len(next_actual_frames.squeeze())):
		# 	if i == 0:
		# 		next_pred_frames[0,i] = model.predict(np.expand_dims(frames[:,i],axis=0), batch_size=args.batch_size)
		# 	else:
		# 		next_pred_frames[0,i] = model.predict(np.expand_dims(next_pred_frames[:,i-1],axis=0), batch_size=args.batch_size)

		frames = frames.squeeze()
		next_pred_frames = next_pred_frames.squeeze()
		next_actual_frames = next_actual_frames.squeeze()

		write_to_video(next_actual_frames, 'results/{:03d}_actual.avi'.format(j))
		write_to_video(next_pred_frames, 'results/{:03d}_pred_trial.avi'.format(j))

		for i in range(len(frames)):
			cv2.imshow('actual', next_actual_frames[i])
			cv2.imshow('prediction', next_pred_frames[i])
			# cv2.imshow('reference', frames[i])
			cv2.waitKey(25)
			# if cv2.waitKey(25) & 0xFF == ord('q'):
			# 	break

def main(args):
	video_gen = video_generator(
		video_dir=args.test_directory,
		frame_height=args.frame_height,
		frame_width=args.frame_width,
		frame_channels=args.frame_channels,
		batch_size=args.batch_size, 
		time=args.time,
		camera_fps=args.camera_fps,
		json_filename="kinetics_test.json")

	rmsprop = RMSprop(lr=0.0003)

	predict_gen = vaegan_loader(video_gen, time_init=2*args.time-1, latent_dim=args.latent_dim)

	enc = encoder(time=2*args.time-1, latent_dim=args.latent_dim, frame_height=args.frame_height, frame_width=args.frame_width, frame_channels=args.frame_channels)
	gen = generator(time=args.time, latent_dim=args.latent_dim, frame_height=args.frame_height, frame_width=args.frame_width, frame_channels=args.frame_channels)
	dis_gan = discriminator(time=2*args.time-1, name='gan', frame_height=args.frame_height, frame_width=args.frame_width, frame_channels=args.frame_channels)
	dis_vae = discriminator(time=2*args.time-1, name='vae', frame_height=args.frame_height, frame_width=args.frame_width, frame_channels=args.frame_channels)

	encoder_train, generator_train, discriminator_train, vaegan= build_graph(enc, gen, dis_gan, dis_vae, time=args.time, latent_dim=args.latent_dim, frame_channels=args.frame_channels)

	set_trainable(dis_vae, True)
	set_trainable(enc, False)
	set_trainable(gen, False)
	discriminator_train.compile(rmsprop, 
		['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], 
		loss_weights=[1., 1., 1., 1.])
	discriminator_train.summary()

	set_trainable(dis_vae, False)
	set_trainable(dis_gan, False)
	set_trainable(gen, True)
	generator_train.compile(rmsprop, 
		['binary_crossentropy','binary_crossentropy','mean_absolute_error'], 
		loss_weights=[1., 1., 5.])
	generator_train.summary()

	set_trainable(gen, False)
	set_trainable(enc, True)
	encoder_train.compile(rmsprop, [kl_loss, 'mean_absolute_error'])
	encoder_train.summary()

	if args.load_epoch>0:
		discriminator_train.load_weights('models/discriminator.{:03d}.h5'.format(args.load_epoch))
		generator_train.load_weights('models/generator.{:03d}.h5'.format(args.load_epoch))
		encoder_train.load_weights('models/encoder.{:03d}.h5'.format(args.load_epoch))
		print('Loaded weights {:03d}'.format(args.load_epoch))
	# gen_single = generator(time=1, latent_dim=args.latent_dim, frame_height=args.frame_height, frame_width=args.frame_width, frame_channels=args.frame_channels)
	# gen_single.compile(rmsprop, ['mean_absolute_error'])
	# gen_single.load_weights(args.weights_directory, by_name=True)
	# gen_single.summary()
	# for j in range(len(glob.glob(os.path.join(args.test_directory,'*.mp4')))):

	seed = 0

	val_video_dir = os.path.join(args.test_directory, '*_act_14_*/*.jpg')
	val_steps = count_images(val_video_dir, batch_size=args.batch_size, time=args.time)

	# val_video_dir = os.path.join(args.test_directory, '*.avi')
	# val_steps = count_frames(val_video_dir, batch_size=args.batch_size, time=args.time, camera_fps=args.camera_fps)

	# video_loader_val_enc = video_generator(
	# 						video_dir = args.test_directory,  
	# 						frame_height=args.frame_height,
	# 						frame_width=args.frame_width,
	# 						frame_channels=args.frame_channels,
	# 						batch_size = args.batch_size, 
	# 						time = args.time,
	# 						camera_fps = args.camera_fps,
	# 						json_filename = "kinetics_val.json")
	# enc_loader_val = encoder_loader(video_loader_val_enc, latent_dim=args.latent_dim, seed=seed, time_init=args.time)
	# enc_losses_val = encoder_train.evaluate_generator(enc_loader_val, steps=val_steps)
	# print('Encoder: ')
	# print(encoder_train.metrics_names)
	# print(enc_losses_val)

	j = 0
	while True:
		j = j + 1
		inputs, next_actual_frames = next(predict_gen)
		# print(inputs.shape)
		# next_pred_frames = np.empty_like(next_actual_frames)
		# next_pred_frames[:,0] = gen.predict([np.repeat(np.expand_dims(inputs[0][:,0], axis=0), args.time, axis=1), np.repeat(inputs[1][0], args.time, axis=1)], batch_size=args.batch_size)[:,0]
		# for k in range(1, args.time):
		# 	next_pred_frames[:,k] = gen.predict([np.expand_dims(next_pred_frames[:,k-1], axis=0), inputs[1][k]]*args.time, batch_size=args.batch_size)[:,k]
		# out = dis_gan.predict(next_pred_frames, batch_size=args.batch_size)
		# print(out)
		# current_pred_frames = vaegan.predict(inputs, batch_size=args.batch_size)
		# current_pred_frames[:,0] = x
		# for i in range(30):
		# next_pred_frames = vaegan.predict(inputs, batch_size=args.batch_size)
		next_pred_frames = np.empty_like(next_actual_frames)
		# print(next_pred_frames.shape)
		# next_pred_frames = gen.predict(inputs, batch_size= args.batch_size)
		# next_pred_frames[:,0:args.time - 1] = inputs[1][:, 1:args.time]
		for i in range(args.time):
			next_gen_frames = gen.predict(inputs, batch_size=args.batch_size)
			if i == 0:
				next_pred_frames[:, :args.time] = next_gen_frames
			else:
				next_pred_frames[:, args.time + i - 1] = next_gen_frames[:, args.time - 1]
			inputs[1][:, 0:args.time-1] = inputs[1][:, 1:args.time]
			inputs[1][:, args.time-1] = next_gen_frames[:, args.time-1] 
		# losses = vaegan.evaluate(inputs, next_actual_frames, batch_size=args.batch_size)
		# print("losses", losses) 	
		for i in range(next_pred_frames.shape[1]):
			# cv2.imshow('previous', inputs[1][0,i])
			# cv2.imshow('next pred {:04d}'.format(j), next_pred_frames[0,i])
			# cv2.imshow('next actual {:04d}'.format(j), next_actual_frames[0,i])
			cv2.imshow('next pred', next_pred_frames[0,i])
			cv2.imshow('next actual', next_actual_frames[0,i])
			# filename_pred = "results_images/predicted_{:04d}_{:04d}.jpg".format(j,i)
			# filename_actual = "results_images/actual_{:04d}_{:04d}.jpg".format(j,i)
			# cv2.imwrite(filename_pred, (255.*next_pred_frames[0,i]).astype(int))
			# cv2.imwrite(filename_actual, (255.*next_actual_frames[0,i]).astype(int))
			cv2.waitKey(0)
			# inputs[0] = np.expand_dims(next_actual_frames[0, next_pred_frames.shape[1] - 1], axis=0)
			# inputs[3] = next_pred_frames

		
		# out = dis_gan.predict(next_pred_frames, batch_size=args.batch_size)
		# # print('Discriminator: ', out)
		# next_pred_frames = np.empty_like(next_actual_frames)
		# # next_pred_frames = vaegan.predict(inputs)
		# # next_pred_frames = gen_single.predict(inputs)
		# # prev_frame = np.expand_dims(x, axis=0)
		# for i in range(args.time):
		# # 	z = z_p[:,i]
		# # 	z = np.expand_dims(z, axis=1)
		# 	pred_frame = gen_single.predict([inputs[0], np.expand_dims(inputs[1][:,i], axis=1), np.expand_dims(inputs[2][:,i], axis=1)], batch_size=args.batch_size)
		# 	next_pred_frames[0,i] = pred_frame.squeeze()
		# # 	loss = gen_single.evaluate([x, prev_frame, z], np.expand_dims(next_actual_frames[:,i], axis=1), batch_size=args.batch_size)
		# # 	print('Generator loss: ', loss.squeeze())
		# # 	# pred_frames = vaegan.predict(inputs, batch_size=args.batch_size)
		# 	cv2.imshow('previous', inputs[1][0,i])
		# 	cv2.imshow('next pred', next_pred_frames[0,i])
		# 	cv2.imshow('next actual', next_actual_frames[0,i])
		# 	cv2.waitKey(0)
		# # 	next_pred_frames[:, i] = pred_frame
		# # 	prev_frame = pred_frame
		# 	# print(i)
		# 	# print(np.sum(pred_frames[:,0]))
		# 	# inputs[1] = pred_frames
		# prev_frame = pred_frame
		# for k in range(args.time, 2*args.time-1):
		# 	pred_frame = gen_single.predict([inputs[0], prev_frame, np.expand_dims(inputs[2][:,k], axis=1)], batch_size=args.batch_size)
		# 	next_pred_frames[0,k] = pred_frame.squeeze()
		# # 	loss = gen_single.evaluate([x, prev_frame, z], np.expand_dims(next_actual_frames[:,i], axis=1), batch_size=args.batch_size)
		# # 	print('Generator loss: ', loss.squeeze())
		# # 	# pred_frames = vaegan.predict(inputs, batch_size=args.batch_size)
		# 	# cv2.imshow('previous', inputs[1][0,j])
		# 	cv2.imshow('next pred', next_pred_frames[0,k])
		# 	cv2.imshow('next actual', next_actual_frames[0,k])
		# 	cv2.waitKey(0)
		# 	prev_frame = pred_frame

		# out = dis_gan.predict(next_pred_frames, batch_size=args.batch_size)
		# print('Discriminator for predicted: ', out.squeeze())
		# out = dis_gan.predict(next_actual_frames, batch_size=args.batch_size)
		# print('Discriminator for actual: ', out.squeeze())

		# encoder_out = encoder_train.evaluate(
		# 	[inputs[0], inputs[1], inputs[3], inputs[2]], 
		# 	[inputs[4], next_actual_frames], 
		# 	batch_size=args.batch_size)
		# print("Encoder {:04d} KL: {:.4f} Rec: {:.4f}".format(j, encoder_out[1], encoder_out[2]))

		# cv2.waitKey(0)

		# cv2.destroyAllWindows()
		# next_pred_frames = np.empty_like(frames)

		# for i in range(len(next_actual_frames.squeeze())):
		# 	if i == 0:
		# 		next_pred_frames[0,i] = model.predict(np.expand_dims(frames[:,i],axis=0), batch_size=args.batch_size)
		# 	else:
		# 		next_pred_frames[0,i] = model.predict(np.expand_dims(next_pred_frames[:,i-1],axis=0), batch_size=args.batch_size)

		# frames = frames.squeeze()
		next_pred_frames = next_pred_frames.squeeze()
		next_actual_frames = next_actual_frames.squeeze()
		# print(next_actual_frames.shape)
		# current_pred_frames = current_pred_frames.squeeze()

		write_to_video(next_actual_frames, 
			'results/{:03d}_actual.avi'.format(j),
			frame_height=args.frame_height,
			frame_width=args.frame_width,
			video_fps=2)
		write_to_video(next_pred_frames, 
			'results/{:03d}_pred.avi'.format(j),
			frame_height=args.frame_height,
			frame_width=args.frame_width,
			video_fps=2)

		# for i in range(len(next_actual_frames)):
		# 	cv2.imshow('actual', next_actual_frames[i])
		# 	cv2.imshow('prediction', next_pred_frames[i])
		# 	# cv2.imshow('before', current_pred_frames[i])
		# 	# cv2.imshow('reference', frames[i])
		# 	cv2.waitKey(25)
		# 	# if cv2.waitKey(25) & 0xFF == ord('q'):
			# 	break
		# break

if __name__=="__main__":
	parser = argparse.ArgumentParser(description='Test the network')
	parser.add_argument('-t', '--test_directory', type=str, default='data/kth_test')
	parser.add_argument('-p', '--predict_images', action='store_true')
	parser.add_argument('-le', '--load_epoch', type=int, default=50)
	parser.add_argument('-b', '--batch_size', type=int, default=1)
	parser.add_argument('-i', '--time', type=int, default=5)
	parser.add_argument('-l', '--latent_dim', type=int, default=8)
	parser.add_argument('-r', '--trial', action='store_true')
	parser.add_argument('-c', '--camera_fps', type=int, default=10)
	parser.add_argument('-fh', '--frame_height', type=int, default=64)
	parser.add_argument('-fw', '--frame_width', type=int, default=64)
	parser.add_argument('-fc', '--frame_channels', type=int, default=1)
	args = parser.parse_args()

	if args.trial:
		main_trial(args)
	else:
		main(args)


