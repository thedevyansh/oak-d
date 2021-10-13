#!/usr/bin/env python3

from pathlib import Path
import cv2
import depthai as dai
import numpy as np

nnPath = str((Path(__file__).parent / Path('./models/mobilenet-ssd_openvino_2021.2_6shave.blob')).resolve())

# MobilenetSSD class labels
labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# Start defining a pipeline
pipeline = dai.Pipeline()

# Define a source - color camera
cam = pipeline.createColorCamera()
cam.setPreviewSize(300, 300)
cam.setInterleaved(False)

# Define a neural network that will make predictions based on the source frames
nn = pipeline.createMobileNetDetectionNetwork()
nn.setBlobPath(nnPath)

nn.setConfidenceThreshold(0.7)
nn.setNumInferenceThreads(2)
nn.input.setBlocking(False)

cam.preview.link(nn.input)

# Create XlinkOut nodes
xoutFrame = pipeline.createXLinkOut()
xoutFrame.setStreamName("rgb")
cam.preview.link(xoutFrame.input)

xoutNN = pipeline.createXLinkOut()
xoutNN.setStreamName("nn")
nn.out.link(xoutNN.input)

# Pipeline defined, now the device is connected to
with dai.Device(pipeline) as device:

    # Start pipeline
    device.startPipeline()

    # Output queues will be used to get the rgb frames and nn data from the
    # output streams defined above.
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    detections = []
    frame = None

    # nn data (bounding box locations) are in <0..1> range - they need to be normalized with frame width/height
    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    def displayFrame(name, frame):
        for detection in detections:
            bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            cv2.putText(frame, labelMap[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
        cv2.imshow(name, frame)

    while True:

        inRgb = qRgb.tryGet()
        inDet = qDet.tryGet()

        if inRgb is not None:
            frame = inRgb.getCvFrame()

        if inDet is not None:
            detections = inDet.detections

        # if the frame is available, render detection data on frame and display.
        if frame is not None:
            displayFrame("Object Detection", frame)

        if cv2.waitKey(1) == ord('q'):
            break
