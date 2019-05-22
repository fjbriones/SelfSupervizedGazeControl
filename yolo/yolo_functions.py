import cv2
import numpy as np

#For yolo from https://github.com/arunponnusamy/object-detection-opencv
##start for yolo
def get_output_layers(net):

	layer_names = net.getLayerNames()

	output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

	return output_layers


def draw_prediction(img, x, y, x_plus_w, y_plus_h):
	# if(class_id==0):

	cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), (0,255,0), 2)

	cv2.putText(img, 'yolo', (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
##end for yolo

def get_yolo_indices(yolo, frame, frame_width, frame_height):

	scale = 0.00392

	blob = cv2.dnn.blobFromImage(frame, scale, (416,416), (0,0,0), True, crop=False)

	yolo.setInput(blob)

	outs = yolo.forward(get_output_layers(yolo))

	class_ids = []
	confidences = []
	boxes = []
	conf_threshold = 0.5
	nms_threshold = 0.4

	for out in outs:
		for detection in out:
			scores = detection[5:]
			class_id = np.argmax(scores)
			confidence = scores[class_id]
			if confidence > 0.5 and class_id == 0:
				center_x = int(detection[0] * frame_width)
				center_y = int(detection[1] * frame_height)
				w = int(detection[2] * frame_width)
				h = int(detection[3] * frame_height)
				x = center_x - w / 2
				y = center_y - h / 2
				confidences.append(float(confidence))
				boxes.append([x, y, w, h])

	return cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold), boxes