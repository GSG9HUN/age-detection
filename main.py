import os,random
import shutil


for i in range(0,100):
    if len(str(i)) == 1:
        directory = "0"+str(i)
        fileName = random.choice(os.listdir(".\\DataSet\\training\\"+directory))
        shutil.move(".\\DataSet\\training\\"+directory+"\\"+fileName,".\\DataSet\\validate\\"+directory+"\\"+fileName)
        continue
    fileName = random.choice(os.listdir(".\\DataSet\\training\\"+str(i)))
    shutil.move(".\\DataSet\\training\\"+str(i)+"\\"+fileName,".\\DataSet\\validate\\"+str(i)+"\\"+fileName)


"""
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('path/to/trained/model')

# Load and preprocess the new image
img = image.load_img('path/to/new/image', target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# Perform age classification
age_classes = ['0-2', '4-6', '8-12', '15-20', '25-32'] # the age classes used during training
predictions = model.predict(img_array)

# Get the predicted age class
predicted_age_class = age_classes[np.argmax(predictions)]

# Print the predicted age class
print('The predicted age class is:', predicted_age_class)
"""