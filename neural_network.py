import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mean_squared_error, mean_absolute_error
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set up the data generators
train_datagen = ImageDataGenerator(rescale=1./255,validation_split = 0.2)
test_datagen = ImageDataGenerator(rescale=1./255)

trainingDirectory = ".\\DataSet\\training"
testDirectory = ".\\DataSet\\validate"

train_generator = train_datagen.flow_from_directory(
    trainingDirectory,
    target_size=(224, 224),
    batch_size=32,
    color_mode="grayscale",
    subset='training',
    class_mode='sparse')

test_generator = test_datagen.flow_from_directory(
    trainingDirectory,
    target_size=(224, 224),
    batch_size=32,
    color_mode="grayscale",
    subset="validation",
    class_mode='sparse')

# Define the neural network architecture
model = tf.keras.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='linear')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss=mean_squared_error,
              metrics=[mean_absolute_error])

# Train the model
history = model.fit(
    train_generator,
    epochs=30,
    validation_data=test_generator
)

# Evaluate the model on the testing set.
test_generator = test_datagen.flow_from_directory(
        testDirectory,
        target_size=(224, 224),
        batch_size=32,
        color_mode='grayscale',
        class_mode='sparse',
        shuffle=False)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_generator, verbose=2)
print('Test accuracy:', test_acc)

model.save('model\\age_model');