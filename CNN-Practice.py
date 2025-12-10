import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import numpy as np 
import matplotlib.pyplot as plt
import cv2
import os
from sklearn.utils.class_weight import compute_class_weight


TRAIN_FILE_PATH = 'archive/train' #Path to the train set
TEST_FILE_PATH = 'archive/test' #Path to the test set
WEIGHTS_FILE = 'emotion_model.weights.h5' #path to the file where you want to save your model weights. (Ensure it ends with weights.h5)
JSON_FILE = 'emotion_model.json' #Path to the model of the CNN.

IMG_SIZE = (48, 48)
BATCH_SIZE = 64
SEED = 123 
EPOCHS = 30 

IMAGE_MAP = {

    #Path to the images you want to display
    
    0: "Path/to/Angry_Hamster.jpg",      
    1: "C:/path/to/my/images/Disgust.png",    
    2: "Path/to/Fearful_Hamster.jpg",    
    3: "Path/to/Happy_Hamster.webp",      
    4: "Path/to/sad-hamster.gif",        
    5: "C:/path/to/my/images/Surprised.png",  
    6: "C:/path/to/my/images/Neutral.png"     
}
if not os.path.exists(TRAIN_FILE_PATH) or not os.path.exists(TEST_FILE_PATH):
    print(f"ERROR: Dataset directory not found. Please verify paths: {TRAIN_FILE_PATH} and {TEST_FILE_PATH}")
    exit()
try:
    with open(JSON_FILE, 'r') as json_file:
        loaded_model_json = json_file.read()
    emotion_model = model_from_json(loaded_model_json)
    emotion_model.load_weights(WEIGHTS_FILE) 
except Exception as e:
    print(f"\nERROR: Could not load trained model for inference. Check if '{JSON_FILE}' and '{WEIGHTS_FILE}' exist.")
    exit()

print("\n--- 5.1 Starting Real-Time Emotion Detection (Press 'q' to quit) ---")

emotion_dict = {
    0: "Angry", 1: "Disgust", 2: "Fearful", 
    3: "Happy", 4: "Sad", 5: "Surprised", 6: "Neutral" 
}

cap = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1280, 720))
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in num_faces:
        
        y_start = max(0, y)
        y_end = min(frame.shape[0], y + h)
        x_start = max(0, x)
        x_end = min(frame.shape[1], x + w)
        
        roi_gray_frame = gray_frame[y_start:y_end, x_start:x_end]
        
        try:
            cropped_img = cv2.resize(roi_gray_frame, (48, 48))
        except cv2.error:
            continue
        
        processed_img = cropped_img / 255.0
        processed_img = np.expand_dims(np.expand_dims(processed_img, -1), 0) 

        emotion_prediction = emotion_model.predict(processed_img, verbose=0)
        maxindex = int(np.argmax(emotion_prediction))

        cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 2)
        
        try:
            image_path = IMAGE_MAP[maxindex]
            emotion_img = cv2.imread(image_path)
            
            if emotion_img is not None:
                display_img = cv2.resize(emotion_img, (400, 400), interpolation=cv2.INTER_AREA)
                cv2.imshow('Predicted Emotion Image', display_img)
            else:
                print(f"Warning: Image file not found at {image_path}")

        except KeyError:
            print(f"Error: Missing image path for index {maxindex}. Check IMAGE_MAP.")

    cv2.imshow('Emotion Detection (Webcam)', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

