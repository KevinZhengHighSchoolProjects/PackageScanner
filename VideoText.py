from imutils.video import VideoStream
from imutils.video import FPS
from imutils.object_detection import non_max_suppression
import numpy as np
from pytesseract import Output
import pytesseract
import argparse
import imutils
import time
import cv2
import gspread
import pandas as pd
import os
import pickle
import re
from oauth2client.service_account import ServiceAccountCredentials

import smtplib, ssl

port = 465  # For SSL
smtp_server = "smtp.gmail.com"
sender_email = "***"  # Enter your address
password = "***"


pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']

# add credentials to the account
creds = ServiceAccountCredentials.from_json_keyfile_name('package-333117-8e4176ad8d62.json', scope)

# authorize the clientsheet 
client = gspread.authorize(creds)
# get the instance of the Spreadsheet
sheet = client.open('Package')

# get the first sheet of the Spreadsheet
sheet_instance = sheet.get_worksheet(1)

lastName = sheet_instance.col_values(1)
firstName = [[],[],[]]
firstName[0] = sheet_instance.col_values(2)
firstName[1] = sheet_instance.col_values(3)
firstName[2] = sheet_instance.col_values(4)
dorm = sheet_instance.col_values(5)
grade = sheet_instance.col_values(6)
email = sheet_instance.col_values(7)


def send_message(receiver, name):
    message = """\
Subject: Your Package has arrived!

Hi {name}, your package has arrived at the admissions office. Please pick it up. 

This message is sent from Python."""
    
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver, message.format(name=name))

def decode_predictions(scores, geometry):
        # grab the number of rows and columns from the scores volume, then
        # initialize our set of bounding box rectangles and corresponding
        # confidence scores
        (numRows, numCols) = scores.shape[2:4]
        rects = []
        confidences = []
        # loop over the number of rows
        for y in range(0, numRows):
                # extract the scores (probabilities), followed by the
                # geometrical data used to derive potential bounding box
                # coordinates that surround text
                scoresData = scores[0, 0, y]
                xData0 = geometry[0, 0, y]
                xData1 = geometry[0, 1, y]
                xData2 = geometry[0, 2, y]
                xData3 = geometry[0, 3, y]
                anglesData = geometry[0, 4, y]
                # loop over the number of columns
                for x in range(0, numCols):
                        # if our score does not have sufficient probability,
                        # ignore it
                        if scoresData[x] < args["min_confidence"]:
                                continue
                        # compute the offset factor as our resulting feature
                        # maps will be 4x smaller than the input image
                        (offsetX, offsetY) = (x * 4.0, y * 4.0)
                        # extract the rotation angle for the prediction and
                        # then compute the sin and cosine
                        angle = anglesData[x]
                        cos = np.cos(angle)
                        sin = np.sin(angle)
                        # use the geometry volume to derive the width and height
                        # of the bounding box
                        h = xData0[x] + xData2[x]
                        w = xData1[x] + xData3[x]
                        # compute both the starting and ending (x, y)-coordinates
                        # for the text prediction bounding box
                        endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                        endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                        startX = int(endX - w)
                        startY = int(endY - h)
                        # add the bounding box coordinates and probability score
                        # to our respective lists
                        rects.append((startX, startY, endX, endY))
                        confidences.append(scoresData[x])
        # return a tuple of the bounding boxes and associated confidences
        return (rects, confidences)

def stillImageText(image1, min_conf):
        image = cv2.imread(image1)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cong = r'--oem 2 --psm 6'
        results = pytesseract.image_to_data(rgb)
        words = ""
        for x, b in enumerate(results.splitlines()):
                if x != 0:
                        b = b.split()
                        if len(b)==12:
                                if float(b[10]) > min_conf and len(b[11]) > 1:
                                        x, y, w, h = int(b[6]), int(b[7]), int(b[8]), int(b[9])
                                        cv2.rectangle(image, (x, y), (w+x, h+y), (0, 0, 255), 3)
                                        cv2.putText(image, b[11], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 255),2)
                                        words += " "
                                        words += b[11].upper()
        # show the output image
        
        words = re.sub(r"[,.;@#?!&$]+\ *", " ", words)
        print(words)
        words = words.split(" ")
        print(words)
        
        ln = []
        for i in range(len(lastName)):
                print(lastName[i])
                if lastName[i] in words:
                        ln.append(i)
                        break;
        fn = -1
        fni = -1
        if ln != []:
                for i in range(len(ln)):
                        if firstName[0][ln[i]] in words:
                                fn = ln[i]
                                fni = 0
                        elif firstName[1][ln[i]] in words:
                                fn = ln[i]
                                fni = 1
                        elif firstName[2][ln[i]] in words:
                                fn = ln[i]
                                fni = 2
        if ln != []:
                if fn != -1:
                        print("Match Found!", lastName[fn], firstName[fni][fn], dorm[fn], grade[fn], email[fn])
                        send_message(email[fn], firstName[0][fn] + " " + lastName[fn])   
                        
        return(image)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-east", "--east", type=str, default="frozen_east_text_detection.pb",
        help="path to input EAST text detector")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
        help="minimum probability required to inspect a region")
ap.add_argument("-w", "--width", type=int, default=320,
        help="resized image width (should be multiple of 32)")
ap.add_argument("-e", "--height", type=int, default=320,
        help="resized image height (should be multiple of 32)")

args = vars(ap.parse_args())

min_conf = 80;

# initialize the original frame dimensions, new frame dimensions,
# and ratio between the dimensions
(W, H) = (None, None)
(newW, newH) = (args["width"], args["height"])
(rW, rH) = (None, None)
# define the two output layer names for the EAST detector model that
# we are interested -- the first is the output probabilities and the
# second can be used to derive the bounding box coordinates of text
layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]
# load the pre-trained EAST text detector
print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet(args["east"])


print("[INFO] starting video stream...")
vs = cv2.VideoCapture(0)
time.sleep(1.0)
vs.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
vs.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
print(vs)
# start the FPS throughput estimator
fps = FPS().start()
mode = 0

# loop over frames from the video stream
while True:
        # grab the current frame, then handle if we are using a
        # VideoStream or VideoCapture object
        r1, frame1 = vs.read()
        #print(frame1.shape)
        r, frame = vs.read()
        # resize the frame, maintaining the aspect ratio
        ##frame = imutils.resize(frame, width=1000)
        orig = frame.copy()
        # if our frame dimensions are None, we still need to compute the
        # ratio of old frame dimensions to new frame dimensions
        if W is None or H is None:
                (H, W) = frame.shape[:2]
                rW = W / float(newW)
                rH = H / float(newH)
        # resize the frame, this time ignoring aspect ratio
        ##frame = cv2.resize(frame, (newW, newH))

        # construct a blob from the frame and then perform a forward pass
        # of the model to obtain the two output layer sets
        blob = cv2.dnn.blobFromImage(frame, 1.0, (newW, newH),
                (123.68, 116.78, 103.94), swapRB=True, crop=False)
        net.setInput(blob)
        (scores, geometry) = net.forward(layerNames)
        # decode the predictions, then  apply non-maxima suppression to
        # suppress weak, overlapping bounding boxes
        (rects, confidences) = decode_predictions(scores, geometry)
        boxes = non_max_suppression(np.array(rects), probs=confidences)
        # loop over the bounding boxes
        for (startX, startY, endX, endY) in boxes:
                # scale the bounding box coordinates based on the respective
                # ratios
                startX = int(startX * rW)
                startY = int(startY * rH)
                endX = int(endX * rW)
                endY = int(endY * rH)
                # draw the bounding box on the frame
                cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # update the FPS counter
        fps.update()
        # show the output frame
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
                break
        elif key == ord(" "):
                if mode == 0:
                        img_name = "img1.jpg"
                        cv2.imwrite(img_name, frame1)
                        image1 = stillImageText(img_name, min_conf)
                        mode = 1;
                else:
                        mode = 0;
        if mode == 0:
                cv2.imshow("Text Detection", orig)
        elif mode == 1:
                cv2.imshow("Text Detection", image1)
                
# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# if we are using a webcam, release the pointer
if not args.get("video", False):
        vs.stop()
# otherwise, release the file pointer
else:
        vs.release()
# close all windows
cv2.destroyAllWindows()
