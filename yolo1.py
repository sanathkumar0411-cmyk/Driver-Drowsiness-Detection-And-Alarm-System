import numpy as np
import time
import cv2
import os
import numpy as np
import threading


args = {"confidence":0.5, "threshold":0.3}
flag = False

labelsPath = "yolo-coco/coco.names"
LABELS = open(labelsPath).read().strip().split("\n")
final_classes = ['person']

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

weightsPath = os.path.abspath("yolo-coco/yolov3-tiny.weights")
configPath = os.path.abspath("yolo-coco/yolov3-tiny.cfg")

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

vs = cv2.VideoCapture(0)
writer = None
(W, H) = (None, None)

flag=True

while True:

    (grabbed, frame) = vs.read()


    if not grabbed:
        break


    if W is None or H is None:
        (H, W) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
        swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()


    boxes = []
    confidences = []
    classIDs = []


    for output in layerOutputs:

        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]


            if confidence > args["confidence"]:

                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
        args["threshold"])


    
    if len(idxs) > 0:
        
        for i in idxs.flatten():
            
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            
            if(LABELS[classIDs[i]] in final_classes):
                
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(LABELS[classIDs[i]],
                    confidences[i])
                cv2.putText(frame, text, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    else:
        flag=True

    cv2.imshow("Output", frame) 

    if cv2.waitKey(1) == ord('q'):
        break


vs.release()
cv2.destroyAllWindows()


