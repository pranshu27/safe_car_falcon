# Importing the necessary libraries
import os
import cv2
import imutils
from imutils.video import VideoStream
from tensorflow.keras.models import load_model
from mask_detection_utils import detect_and_predict_mask

# load the pre-trained model files for serialized face detector
# by using the network configuration file and pre-trained weights
path = os.getcwd() + r'\Seatbelt and Mask Detection\\'
face_net = cv2.dnn.readNet(path+'deploy.prototxt', path+'res10_300x300_ssd_iter_140000.caffemodel')

# load the pre-trained model files for face mask detector
mask_net = load_model(path + 'mask_detector.model')

# capture the video stream from the webcam (source id = 0)
video_stream = VideoStream(src=0).start()

# loop over the frames captured in video stream unless any key is pressed
while (1):

	# capture the current frame from the video_stream object and resize it
    frame = imutils.resize(video_stream.read(), width=800)
	
	# detect masks in each of the faces within the frame
    locs, preds = detect_and_predict_mask(frame, face_net, mask_net)

    # annotate the detected faces with the predicted output
    for i in range(len(preds)):

        # get the coordinates of box bounding the detected face
        startX, startY, endX, endY = locs[i]

        # label green if the face has a mask on it, else red
        label = 'Mask' if preds[i][0] > preds[i][1] else 'No Mask'
        color = (0, 255, 0) if label == 'Mask' else (0, 0, 255)
        
        # add probability to the label
        label = '{}: {:.2f}%'.format(label, max(preds[i][0], preds[i][1])*100)

        # put the bounding rectangle and prediction label
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cv2.putText(frame, label, (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

	# display the frame and exit if any key is pressed
    cv2.imshow("Window", frame)
    if cv2.waitKey(1) != -1:
	    break

# clear the windows and stop video stream
cv2.destroyAllWindows()
video_stream.stop()