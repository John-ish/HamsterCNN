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


TRAIN_FILE_PATH = 'archive/train' #path to the train dataset
TEST_FILE_PATH = 'archive/test' #path to the test dataset
WEIGHTS_FILE = 'emotion_model1.weights.h5' #path to the file where you want to save your model weights. (Ensure it ends with weights.h5)
JSON_FILE = 'emotion_model1.json' #path to the model .json file

IMG_SIZE = (48, 48)
BATCH_SIZE = 64
SEED = 123 
EPOCHS = 30 # Number of training epochs per run

if not os.path.exists(TRAIN_FILE_PATH) or not os.path.exists(TEST_FILE_PATH):
    print(f"ERROR: Dataset directory not found. Please verify paths: {TRAIN_FILE_PATH} and {TEST_FILE_PATH}")
    exit()

print("--- 2. MODERN DATA LOADING (tf.data.Dataset) ---")

train_ds = keras.utils.image_dataset_from_directory(
    TRAIN_FILE_PATH,
    labels="inferred",
    label_mode="categorical",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    shuffle=True,
    seed=SEED
)

validation_ds = keras.utils.image_dataset_from_directory(
    TEST_FILE_PATH,
    labels="inferred",
    label_mode="categorical",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    shuffle=False, 
    seed=SEED
)

num_classes = len(train_ds.class_names) 
INPUT_SHAPE = IMG_SIZE + (1,) 

rescale_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (rescale_layer(x), y))
validation_ds = validation_ds.map(lambda x, y: (rescale_layer(x), y))

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
validation_ds = validation_ds.cache().prefetch(buffer_size=AUTOTUNE)

print("\n--- 2.3 Calculating Class Weights for Imbalance ---")

y_train_ints = np.concatenate([np.argmax(y, axis=-1) for x, y in train_ds])

class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_ints),
    y=y_train_ints
)

class_weights_dict = dict(enumerate(class_weights_array))
print("Class Weights (index: weight) - Higher weights help with Sad/Angry/Disgust:", class_weights_dict)

print("\n--- 3.1 Defining Increased-Capacity CNN Architecture ---")


emotion_model = Sequential()

# Layer 1: Increased from 32/64 to 64/128
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=INPUT_SHAPE, name='conv2d_1'))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', name='conv2d_2'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

# Layer 2: Increased from 128/128 to 256/256
emotion_model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', name='conv2d_3'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', name='conv2d_4'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

# Classifier
emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(num_classes, activation='softmax')) 

initial_learning_rate = 0.0001
lr_schedule = ExponentialDecay(initial_learning_rate, decay_steps=100000, decay_rate=0.96)
optimizer = Adam(learning_rate=lr_schedule)

emotion_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

SHOULD_TRAIN = False # Default: skip training if weights exist

if os.path.exists(WEIGHTS_FILE):
    print(f"\n--- 4.1 Weights Found! Attempting to load existing model state from {WEIGHTS_FILE} ---")
    try:
        emotion_model.load_weights(WEIGHTS_FILE) 
        print("Weights loaded successfully.")
        # Change this line to True if you want to train for more epochs,
        # otherwise, leave as False to skip training and go straight to webcam.
        SHOULD_TRAIN = False 
        
    except ValueError as e:
        print(f"\nFATAL ERROR: Incompatible weights found. Architecture mismatch!")
        print(f"Details: {e}")
        print("Starting training from scratch and overwriting the incompatible weights file.")
        os.remove(WEIGHTS_FILE) # Delete old, bad weights
        SHOULD_TRAIN = True 
else:
    print("\n--- 4.1 No Weights Found. Starting training from scratch. ---")
    SHOULD_TRAIN = True

if SHOULD_TRAIN:
    print("\n--- 4.2 Running Model Training with Class Weighting ---")
    print(f"Training for {EPOCHS} epochs.")
    emotion_model_info = emotion_model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=validation_ds,
        class_weight=class_weights_dict 
    )
    print("Training finished. Saving model files.")
    if not os.path.exists(JSON_FILE):
        model_json = emotion_model.to_json()
        with open(JSON_FILE, "w") as json_file:
            json_file.write(model_json)
        print("Model architecture saved to emotion_model.json.")
        
    
    emotion_model.save_weights(WEIGHTS_FILE)
    
else:
    print("\n--- 4.2 Skipping Training. Proceeding to Inference. ---")
try:
    with open(JSON_FILE, 'r') as json_file:
        loaded_model_json = json_file.read()
    emotion_model = model_from_json(loaded_model_json)
    
    emotion_model.load_weights(WEIGHTS_FILE) 
except Exception as e:
    print(f"\nERROR: Could not load trained model for inference. Check if '{JSON_FILE}' and '{WEIGHTS_FILE}' exist.")
    print(f"Details: {e}")
    exit()

print("\n--- 5.1 Starting Real-Time Emotion Detection (Press 'q' to quit) ---")

emotion_dict = {
    0: "Angry", 1: "Disgust", 2: "Fearful", 
    3: "Happy", 4: "Sad", 5: "Surprised", 6: "Neutral" 
}

# Using Haar Cascade for face detection
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
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 2)
        
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

    cv2.imshow('Emotion Detection (Accuracy Optimized)', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
