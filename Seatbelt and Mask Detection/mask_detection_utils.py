'''
Reference:
Face-Mask-Detection https://github.com/balajisrinivas/Face-Mask-Detection
'''

# Importing the necessary libraries
import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array


def detect_and_predict_mask(frame, face_net, mask_net):
	'''
	The function takes the image frame, face detector model, and mask detector model as input
	and returns the array of box surrounding the detected faces and the prediction array
	Input:
		frame: frame object read from the video stream
		face_net: model object for the pre-trained face detector network
		mask_net: model object for the pre-treained mask detector network
	Output:
		locs: numpy array of corners of the box surrounding the faces detected in the frame
		preds: numpy array of probability of prediction for 'mask' and 'no mask' label
	'''

	# construct a blob from the input frame
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))

	# get the box corners of the detected faces using the face_net model
	face_net.setInput(blob)
	detections = face_net.forward()

	# initialze empty list for faces, surrounding box coordinates, and predictions
	faces, locs, preds = list(), list(), list()

	# iterate over detections
	for i in range(0, detections.shape[2]):

		# execute only if probability > 0.5
		if detections[0, 0, i, 2] > 0.5:
			
			# calculate the coordinates of the bounding box corners
			(h, w) = frame.shape[:2]
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# handle the coordinates that fall beyond the frame dimensions
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the ROI for the face, change the color channel and resize
			# it to 224x224 which is the suitable input size for our mask_net model
			face = frame[startY:endY, startX:endX]
			face = cv2.resize(cv2.cvtColor(face, cv2.COLOR_BGR2RGB), (224, 224))

			# preprocess the input face to an array of pixel intensity (r, g, b) values
			face = preprocess_input(img_to_array(face))

			# append face data and bounding box corners to the lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# predict the mask over the faces if at least one face is detected
	if len(faces) > 0:
		faces = np.array(faces, dtype="float32")
		# predict if the faces contain mask or not by using the mask_net model object
		preds = mask_net.predict(faces, batch_size=32)

	# return the array of box surrounding the detected faces and the prediction array
	return (locs, preds)