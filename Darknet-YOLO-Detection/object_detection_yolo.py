# This code is written at BigVision LLC. It is based on the OpenCV project. It is subject to the license terms in the LICENSE file found in this distribution and at http://opencv.org/license.html

# Usage example:  python3 object_detection_yolo.py --video=run.mp4
#                 python3 object_detection_yolo.py --image=bird.jpg

import cv2 as cv
import argparse
import sys
import numpy as np
import os.path
import math
from keras.models import load_model

# Initialize the parameters
confThreshold = 0.5  #Confidence threshold
nmsThreshold = 0.4   #Non-maximum suppression threshold
inpWidth = 416       #Width of network's input image
inpHeight = 416      #Height of network's input image

parser = argparse.ArgumentParser(description='Object Detection using YOLO in OPENCV')
parser.add_argument('--image', help='Path to image file.')
parser.add_argument('--video', help='Path to video file.')
args = parser.parse_args()
        
# Load names of classes
classesFile = "coco.names";
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = "yolov3-spp.cfg";
modelWeights = "yolov3-spp.weights";

net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_OPENCL)

# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def distance_to_camera(left, top, right, bottom):
    (h,w) = (frame.shape[0], frame.shape[1])
    initial_width = 2
    known_distance = 1
    initial_pixel_distance = (frame.shape[1]*0.90 - frame.shape[1]*0.1)
    focal_length = (initial_pixel_distance * known_distance) / initial_width
    width = right-left
    dist =  (initial_width * focal_length) / width
    #cv.line(frame,(w//2, h),(left+(right-left)//2, bottom),(0,0,0), 2)
    cv.putText(frame, '%.2fm'%(dist), (left, bottom), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)

# def BirdEyes(frame):
#     (h,w) = (frame.shape[0], frame.shape[1])
#     # Points of the corners of the bounding box
#     pts1 = np.float32([[0,530],[w,530],[0,h],[w,h]])
#     # Points of the corners of the perspective transformation 
#     pts2 = np.float32([[0,0],[w,0],[0,h],[w,h]])

#     # Compute the perspective transform matrix and then apply it
#     transform_matrix = cv.getPerspectiveTransform(pts1,pts2)
#     dst = cv.warpPerspective(frame,transform_matrix,(w,h))

#     # Return the warped image
#     return dst

# def DistanceDetec(left, top, right, bottom):
#     dst = BirdEyes(frame)
#     (h,w) = (dst.shape[0], dst.shape[1])

#     delta_x = w//2-(left+(right-left)//2)
#     delta_y = h - bottom

#     # Distance in metres,we should measure the object in 10 metres and then tranfer px to meters
#     # 0.05 - for test 
#     metres = ' %.1f m' % (np.sqrt(delta_x**2 + delta_y**2) * 0.05 )  

#     #cv.putText(frame, metres, (left, bottom), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)
#     cv.line(frame,(w//2, h),(left+(right-left)//2, bottom),(0,0,0), 2)
#     cv.putText(frame, metres, (left+(right-left)//2, bottom), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)
    
# Draw the predicted bounding box
def drawPred(classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 2)

    label = '%.2f' % conf
        
    # Get the label for the class name and its confidence
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)
        if classes[classId] == "traffic light":
            states = ['red', 'yellow', 'green', 'off']
            crop_frame = frame[top:bottom,left:right]
            predicted_state = test_an_image(crop_frame, model=load_model('model.h5'))
            for idx in predicted_state:
                cv.putText(frame, states[idx], (860,540), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)

    #Display the label at the top of the bounding box
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)

def test_an_image(file_path, model):
    """
    resize the input image to [32, 32, 3], then feed it into the NN for prediction
    :param file_path:
    :return:
    """

    desired_dim=(32,32)
    
    img_resized = cv.resize(file_path, desired_dim, interpolation=cv.INTER_LINEAR)
    img_ = np.expand_dims(np.array(img_resized), axis=0)

    predicted_state = model.predict_classes(img_)

    return predicted_state


# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIds = []
    confidences = []
    boxes = []
    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        # Remove big bounding box
        if width*height < (frameHeight*frameWidth)/3:
            drawPred(classIds[i], confidences[i], left, top, left + width, top + height)
            distance_to_camera(left, top, left + width, top + height)

# Process inputs
winName = 'Deep learning object detection in OpenCV'
cv.namedWindow(winName, cv.WINDOW_NORMAL)

outputFile = "yolo_out_py.avi"
if (args.image):
    # Open the image file
    if not os.path.isfile(args.image):
        print("Input image file ", args.image, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.image)
    outputFile = args.image[:-4]+'_yolo_out_py.jpg'
elif (args.video):
    # Open the video file
    if not os.path.isfile(args.video):
        print("Input video file ", args.video, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.video)
    outputFile = args.video[:-4]+'_yolo_out_py.avi'
else:
    # Webcam input
    cap = cv.VideoCapture(0)

# Get the video writer initialized to save the output video
if (not args.image):
    vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M','J','P','G'), 30, (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

while cv.waitKey(1) < 0:
    
    # get frame from the video
    hasFrame, frame = cap.read()
    
    # Stop the program if reached end of video
    if not hasFrame:
        print("Done processing !!!")
        print("Output file is stored as ", outputFile)
        cv.waitKey(3000)
        # Release device
        cap.release()
        break

    # Create a 4D blob from a frame.
    blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

    # Sets the input to the network
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = net.forward(getOutputsNames(net))

    # Remove the bounding boxes with low confidence
    postprocess(frame, outs)

    # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
    cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    # Write the frame with the detection boxes
    if (args.image):
        cv.imwrite(outputFile, frame.astype(np.uint8));
    else:
        vid_writer.write(frame.astype(np.uint8))

    # demo_flag = True
    # states = ['red', 'yellow', 'green', 'off']
    # if demo_flag:
    #     predicted_state = test_an_image(frame, model=load_model('model.h5'))
    #     for idx in predicted_state:
    #         cv.putText(frame, states[idx], (860,540), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)

    
    cv.imshow(winName, frame)

