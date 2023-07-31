import pandas as pd
import os
import librosa
import numpy as np
from tensorflow.keras.models import load_model

audio_dataset_path = 'data/wavfiles'
metadata = pd.read_csv('data/bird_songs_metadata.csv')
metadata.head()

def features_extractor(file):
    audio, sample_rate = librosa.load(file, res_type='kaiser_fast') 
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    return mfccs_scaled_features

# Load the trained model
model = load_model('saved_models\\audio_classification.hdf5')  # Replace 'path_to_your_model.h5' with the actual path

# Function to predict the class of an audio file
def predict_audio_class(file_path):
    data = features_extractor(file_path)
    data = data.reshape(1, -1)  # Reshape data to match the model input shape
    predicted_class = np.argmax(model.predict(data))
    return predicted_class

# Example: Predict the class of an audio file
audio_file_path = r"data\testfile\american\564545-5.wav"  # Replace with the path to your audio file
predicted_class_index = predict_audio_class(audio_file_path)
print("Predicted Class Index:", predicted_class_index)

# Get the corresponding class label from the metadata
# predicted_class_label = metadata.loc[predicted_class_index, 'name']
# print("Predicted Class Label:", predicted_class_label)

if predicted_class_index == 0:
    result = "American Robin"
elif predicted_class_index == 1:
    result = "Bewick's Wren"
elif predicted_class_index == 2:
    result = "Northern Cardinal"
elif predicted_class_index == 3:
    result = "Northern Mockingbird"
elif predicted_class_index == 4:
    result = "Sparrow"
print(result)