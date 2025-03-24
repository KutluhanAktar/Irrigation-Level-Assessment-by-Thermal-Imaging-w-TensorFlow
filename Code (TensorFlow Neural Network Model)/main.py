# Irrigation Level Assessment by Thermal Imaging w/ TensorFlow
#
# Windows, Linux, or Ubuntu
#
# By Kutluhan Aktar
#
# Collect irrigation level data by thermal imaging, build and train a neural network model, and run the model directly on Wio Terminal.
#
#
# For more information:
# https://www.theamplituhedron.com/projects/Irrigation_Level_Assessment_by_Thermal_Imaging_w_TensorFlow/

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from test_data import test_inputs, test_labels
from tflite_to_c_array import hex_to_c_array

# Create a class to build a neural network model after combining, visualizing, and scaling (normalizing) the thermal imaging data collected for each irrigation level. 
class Irrigation_Level:
    def __init__(self, data_files):
        self.scale_val = 22
        self.labels = []
        self.data_files = data_files
        self.model_name = "irrigation_model"
    # Combine data from all CSV files to define inputs and scale (normalize) inputs depending on the neural network model.     
    def combine_and_scale_data_to_define_inputs(self):
        # Define the output array.
        output = []
        for file in self.data_files:
            # Define the given CSV file path.
            csv_path = "data\{}.csv".format(file)
            # Read data from the given CSV file.
            with open(csv_path, 'r') as f:
                data = np.genfromtxt(f, dtype=float, delimiter=',')
                # Append the recently collated data to the output array.
                output.append(data)
            f.close()
        # Combine all data from each irrigation level (class) to create the inputs array.
        self.inputs = np.concatenate([output[i] for i in range(len(output))])     
        # Scale the inputs array.
        self.inputs = self.inputs / self.scale_val
    # Assign labels for each thermal imaging array (input) according to the given irrigation level.
    def define_and_assign_labels(self):
        _class = 0
        for file in self.data_files:
            # Define the irrigation classes:
            if (file == "dry"):
                _class = 0
            elif (file == "moderate"):
                _class = 1
            elif (file == "sufficient"):
                _class = 2
            elif (file == "excessive"):
                _class = 3
            # Define the given CSV file path.
            csv_path = "data\{}.csv".format(file)
            # Read data from the given CSV file.
            with open(csv_path, 'r') as f:
                data = np.genfromtxt(f, dtype=float, delimiter=',')
                # Assign labels for each input in the given irrigation level (CSV file).
                for i in range(len(data)):
                    self.labels.append(_class)
            f.close()
        self.labels = np.asarray(self.labels)
    # Split inputs and labels into training and test sets.
    def split_data(self):
        # (training)
        self.train_inputs = self.inputs
        self.train_labels = self.labels
        # (test)
        self.test_inputs = test_inputs / self.scale_val
        self.test_labels = test_labels
        # Print the total input and label numbers.
        print("\r\nTotal Input: " + str(len(self.train_inputs)) + "\r\nTotal Label: " + str(len(self.train_labels)) + "\r\n")
    # Build and train an artificial neural network (ANN) to make predictions on the irrigation levels (classes) based on thermal imaging. 
    def build_and_train_model(self):
        # Build the neural network:
        self.model = keras.Sequential([
            keras.Input(shape=(192,)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(4, activation='softmax')
        ])
        # Compile:
        self.model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])
        # Train:
        self.model.fit(self.train_inputs, self.train_labels, epochs=20)
        # Test the accuracy:
        print("\n\nModel Evaluation:")
        test_loss, test_acc = self.model.evaluate(self.test_inputs, self.test_labels) 
        print("Evaluated Accuracy: ", test_acc)
    # Save the model for further usage:
    def save_model(self):
        self.model.save("model/{}.h5".format(self.model_name))        
    # Convert the TensorFlow Keras H5 model (.h5) to a TensorFlow Lite model (.tflite).
    def convert_TF_model(self, path):
        #model = tf.keras.models.load_model(path + ".h5")
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        #converter.optimizations = [tf.lite.Optimize.DEFAULT]
        #converter.target_spec.supported_types = [tf.float16]
        tflite_model = converter.convert()
        # Save the TensorFlow Lite model.
        with open(path + '.tflite', 'wb') as f:
            f.write(tflite_model)
        print("\r\nTensorFlow Keras H5 model converted to a TensorFlow Lite model!\r\n")
        # Convert the recently created TensorFlow Lite model to hex bytes (C array) to generate a .h file string.
        with open("model/{}.h".format(self.model_name), 'w') as file:
            file.write(hex_to_c_array(tflite_model, self.model_name))
        print("\r\nTensorFlow Lite model converted to a C header (.h) file!\r\n")
    # Run Artificial Neural Network (ANN):
    def Neural_Network(self, save):
        self.combine_and_scale_data_to_define_inputs()
        self.define_and_assign_labels()
        self.split_data()
        self.build_and_train_model()
        if save:
            self.save_model()

# Define a new class object named 'irrigation_rate':
irrigation_rate = Irrigation_Level(["dry", "moderate", "sufficient", "excessive"])

# Artificial Neural Network (ANN):
irrigation_rate.Neural_Network(True)

# Convert the TensorFlow Keras H5 model to a TensorFlow Lite model:
irrigation_rate.convert_TF_model("model/{}".format(irrigation_rate.model_name))
            