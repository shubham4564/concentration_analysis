from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np

from keras.preprocessing.image import img_to_array
import sys

import matplotlib.pyplot as plt

# models
# face and eyes are templates from opencv
face_cascade = cv2.CascadeClassifier('haarcascade_files/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_files/haarcascade_eye.xml')
distract_model = load_model('distraction_model.hdf5', compile=False)
emotion_model_path = '_mini_XCEPTION.72-0.64.hdf5'
img_path = sys.argv[1]
detection_model_path = 'hc_ff_d.xml'

# frame params
frame_w = 720
border_w = 2
min_size_w = 240
min_size_h = 240
min_size_w_eye = 60
min_size_h_eye = 60
scale_factor = 1.1
min_neighbours = 5
# image iterators
# i = image filename number
# j = controls how often images should be saved
i = 0
j = 0

# hyper-parameters for bounding boxes shape
# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry","disgust","scared", "happy", "sad", "surprised","neutral"]

# Video writer
# IMPORTANT:
# - frame width and height must match output frame shape
# - avi works on ubuntu. mp4 doesn't :/
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
video_out = cv2.VideoWriter('video_out.avi', fourcc, 10.0,(1200, 900))

#reading the frame
orig_frame = cv2.imread(img_path)
eframe = cv2.imread(img_path,0)
efaces = face_detection.detectMultiScale(eframe,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
 
# init camera window
cv2.namedWindow('Concentration Analysis')
camera = cv2.VideoCapture(0)
#cap.set(cv2.cv.CV_CAP_PROP_FPS, 60)

# Check if camera opened successfully
if (camera.isOpened() == False): 
    print("Unable to read camera feed")
    
counter = 0
frame_array = []
conc_array = []
cumulative_perconc = 0

while True:
    # get frame
    ret, frame = camera.read()
    
    
    counter += 1
   
   # efaces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)

    # if we have a frame, do stuff
    if ret:
        
        
        canvas = np.zeros((250, 300, 3), dtype="uint8")
        frameClone = frame.copy()
        
        # make frame bigger
        frame = imutils.resize(frame, width=frame_w)

        # use grayscale for faster processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect face(s)
        faces = face_cascade.detectMultiScale(gray,scaleFactor=scale_factor,minNeighbors=min_neighbours,minSize=(min_size_w,min_size_h),flags=cv2.CASCADE_SCALE_IMAGE)

        # for each face, detect eyes and distraction
        if len(faces) > 0:
            
            # loop through faces
            for (x,y,w,h) in faces:
                # draw face rectangle
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                # get gray face for eye detection
                roi_gray = gray[y:y+h, x:x+w]
                # get colour face for distraction detection (model has 3 input channels - probably redundant)
                roi_color = frame[y:y+h, x:x+w]
                # detect gray eyes
                eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=scale_factor,minNeighbors=min_neighbours,minSize=(min_size_w_eye,min_size_w_eye))

                # init probability list for each eye prediction
                probs = list()

                # loop through detected eyes
                for (ex,ey,ew,eh) in eyes:
                    # draw eye rectangles
                    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),border_w)
                    # get colour eye for distraction detection
                    roi = roi_color[ey+border_w:ey+eh-border_w, ex+border_w:ex+ew-border_w]
                    # match CNN input shape
                    roi = cv2.resize(roi, (64, 64))
                    # normalize (as done in model training)
                    roi = roi.astype("float") / 255.0
                    # change to array
                    roi = img_to_array(roi)
                    # correct shape
                    roi = np.expand_dims(roi, axis=0)

                    # distraction classification/detection
                    prediction = distract_model.predict(roi)
                    # save eye result
                    probs.append(prediction[0])

                # get average score for all eyes
                probs_mean = np.mean(probs)

                # get label
                if probs_mean <= 0.5:
                    label = 'distracted'
                else:
                    label = 'focused'
                
                # insert label on frame
                cv2.putText(frame,label,(x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0,0,255), 3, cv2.LINE_AA)
        #else: continue
              
        efaces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
        
        if len(efaces) > 0:
            efaces = sorted(efaces, reverse=True,
            key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            (fX, fY, fW, fH) = efaces
                        # Extract the ROI of the face from the grayscale image, resize it to a fixed 48x48 pixels, and then prepare
                # the ROI for classification via the CNN
            eroi = gray[fY:fY + fH, fX:fX + fW]
            eroi = cv2.resize(eroi, (48, 48))
            eroi = eroi.astype("float") / 255.0
            eroi = img_to_array(eroi)
            eroi = np.expand_dims(eroi, axis=0)

            preds = emotion_classifier.predict(eroi)[0]
            emotion_probability = np.max(preds)
            elabel = EMOTIONS[preds.argmax()]   
        else: continue
        
        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                # construct the label text
                text = "{}: {:.2f}%".format(emotion, prob * 100)
                w = int(prob * 300)
                cv2.rectangle(canvas, (7, (i * 35) + 5),
                (w, (i * 35) + 35), (0, 0, 255), -1)
                cv2.putText(canvas, text, (10, (i * 35) + 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 2)
                cv2.putText(frameClone, elabel, (fX, fY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                              (0, 0, 255), 2)
        
        if probs_mean > 0.5:
            if elabel == 'neutral':
                perconc = (emotion_probability*0.9) * 100
            elif elabel == 'happy':
                perconc = (emotion_probability*0.6) * 100
            elif elabel == 'surprised':
                perconc = (emotion_probability*0.5) * 100
            elif elabel == 'sad':
                perconc = (emotion_probability*0.3) * 100
            elif elabel == 'scared':
                perconc = (emotion_probability*0.3) * 100
            elif elabel == 'angry':
                perconc = (emotion_probability*0.25) * 100
            elif elabel == 'disgust':
                perconc = (emotion_probability*0.2) * 100
        else:
            perconc = 0
        
        
        frame_array.append(counter)
        #print (frame_array)
        conc_array.append(perconc)
        #print (conc_array)
        
        plt.plot(frame_array, conc_array)
        # naming the x axis 
        plt.xlabel('Frames') 
        # naming the y axis 
        plt.ylabel('Concentration (%)') 

        # giving a title to my graph 
        plt.title('Concentration Graph') 
        plt.show() 
        
        # Write the frame to video
        video_out.write(frame)
               
        # display frame in window
        cv2.imshow('Concentration Analysis', frame)   
        cv2.imshow("Probabilities", canvas)

        # quit with q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # no frame, don't do stuff
    #else:
     #   break

     # close
camera.release()
video_out.release()
cv2.destroyAllWindows()

