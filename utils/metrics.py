import numpy as np

def calculate_iou(yolo_box, input_box):
	y1_int = max(yolo_box[1], input_box[1])
	x1_int = max(yolo_box[0], input_box[0])
	y2_int = min(yolo_box[1] + yolo_box[3], input_box[1] + input_box[3])
	x2_int = min(yolo_box[0] + yolo_box[2], input_box[0] + input_box[2])

	area_int = max(0, y2_int - y1_int) * max(0, x2_int - x1_int)
	area_gt = yolo_box[2] * yolo_box[3]
	area_input = input_box[2] * input_box[3]

	iou = float(area_int)/float(area_input + area_gt - area_int)
	cov = float(area_int)/float(area_gt)

	# print(float(area_int)/float(area_input + area_gt - area_int))

	return iou, cov

def get_iou(yolo_box, indices, coords, min_loss):
	min_loss_box = coords[min_loss]
	min_loss_iou, min_loss_cov = calculate_iou(yolo_box, min_loss_box)
	# print(coords[indices])
	coords_avg = np.mean(coords[indices], axis=0)
	# print(coords_avg)
	all_iou, all_cov = calculate_iou(yolo_box, coords_avg)

	return min_loss_iou, all_iou, min_loss_cov, all_cov