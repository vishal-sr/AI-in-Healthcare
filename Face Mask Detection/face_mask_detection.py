import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

cap = cv2.VideoCapture(0) # Video source capturing
cap.set(3, 640) # Width of the video window
cap.set(4, 480) # Height of the video window

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') # Face detector
maskClassifier = load_model('maskclassifier.model') # Mask classifier

while True:
   
    _, frame = cap.read() # Reading frame from video source

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) # Converting RGB to Grayscale
    
    faces = faceCascade.detectMultiScale( # Detecting faces
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        )
    
    for (x, y, h, w) in faces: 

        faceROI = frame[y : y + h, x : x + w, :] # Cropping face region of interest

        faceROI = cv2.resize(faceROI, (160, 160)) # Resizing faceROI to 160x160
                                                  # Because, Our VGG16 model accepts 160x160 as input 
        faceROI = img_to_array(faceROI)
        faceROI = faceROI.reshape(1, 160, 160, 3) # Changing dimensions to 1x160x160x3, Because our VGG16 
                                                  # take input as 4D matrix(BATCH_SIZE, 160, 160, #Channels)

        prediction = maskClassifier(faceROI) # Making predictions
        (withoutmask, withmask) = prediction[0].numpy()
        
        # Drawing bounding boxes using OpenCV
        (label, color, prob) = ('Mask', (0, 255, 0), withmask*100.0) if withmask > withoutmask else ('No mask', (0, 0, 255), withoutmask*100.0)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        cv2.rectangle(frame, (x + 15, y + 2), (x + w - 15, y + 20), (0, 0, 0), -1) #lower
        cv2.rectangle(frame, (x + 15, y - 2), (x + w - 15, y - 20), (0, 0, 0), -1) #upper

        cv2.putText(frame, str(prob)+' %', (x + 20, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.putText(frame, label, (x + 20, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

        
    cv2.imshow('Video', frame) # Displaying the video
    

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release() # Releasing the capture
cv2.destroyAllWindows()
