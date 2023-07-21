import tensorflow as tf
import numpy as np
import pandas as pd
import streamlit as st 
from tensorflow.keras import regularizers
from keras.layers import Dropout
from keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

@st.cache_data()
def ann() : 

  # # Load and prepare the MNIST dataset
  # mnist = tf.keras.datasets.mnist

  # # Split dataset in data of Train and Data od Test
  # (x_train, y_train), (x_test, y_test) = mnist.load_data()

  # # The pixel values of the images range from __0__ through __255__.
  # x_train, x_test = x_train / 255.0, x_test / 255.0


  # Load and prepare the MNIST dataset
  mnist = tf.keras.datasets.mnist
  # Split dataset in data of Train and Data od Test
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  x_train, x_val, y_train, y_val  = train_test_split( x_train, y_train, test_size=0.166, random_state=4)
  x_train,x_val, x_test = x_train / 255.0, x_val/ 255.0, x_test / 255.0

  x_train = x_train.reshape((-1, 28, 28, 1))
  y_train = to_categorical(y_train, 10)
  x_val = x_val.reshape((-1, 28, 28, 1))
  y_val= to_categorical(y_val, 10)

  model = tf.keras.Sequential([
      # Capa de entrada
      tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28,28,1),kernel_regularizer=regularizers.l2(1e-5)),
      tf.keras.layers.Dropout(0.3),#regularización y reducir el sobreajuste del modelo
      
      # Capa de convolución 1
      tf.keras.layers.Conv2D(64, (3, 3), activation='relu',kernel_regularizer=regularizers.l2(1e-5)),
      tf.keras.layers.Dropout(0.3),#regularización y reducir el sobreajuste del modelo
      
      # Capa de reducción de muestreo 1
      tf.keras.layers.MaxPooling2D((2, 2)),
      
      # Capa de convolución 2
      tf.keras.layers.Conv2D(128, (3, 3), activation='relu',kernel_regularizer=regularizers.l2(1e-5)),
      tf.keras.layers.Dropout(0.3),#regularización y reducir el sobreajuste del modelo
      
      # Capa de reducción de muestreo 2
      tf.keras.layers.MaxPooling2D((2, 2)),

      
      # Capa completamente conectada 1
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(256, activation='relu',kernel_regularizer=regularizers.l2(1e-5)),
      tf.keras.layers.Dropout(0.3),#regularización y reducir el sobreajuste del modelo
      
      # Capa completamente conectada 2
      tf.keras.layers.Dense(128, activation='relu',kernel_regularizer=regularizers.l2(1e-5)),
      tf.keras.layers.Dropout(0.3),#regularización y reducir el sobreajuste del modelo
      
      # Capa de salida
      tf.keras.layers.Dense(10, activation='softmax')  # num_classes representa el número de clases para la tarea de clasificación
  ])



  # model.compile(optimizer = 'adam',
  #             loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  #             metrics = ['accuracy'])
  # Compilar el modelo
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

  model.fit(x_train,
            y_train,
            epochs=20,
            validation_data=(x_val, y_val) )


  return model




