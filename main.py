import streamlit as st
from grafic import *
from model_trained import *
import tensorflow as tf
import numpy as np


# if __name__=='__main__':
st.title( 'Convolutional Neural Network for MNIST Handwritten Digit Classification')


st.write('''The **MNIST** dataset is a collection of 70,000 handwritten digits (0 to 9) in 
	28x28 grayscale images. It serves as a standard benchmark for image classification 
	tasks and is widely used in machine learning and computer vision research.''')

st.write('''This project develops a **Convolutional Neural Network (CNN)** to recognize handwritten digits
	using the MNIST dataset. The CNN architecture includes convolutional, pooling, and fully 
	connected layers. Training employs backpropagation and stochastic gradient descent to 
	optimize model parameters. The goal is to achieve high accuracy in digit recognition for 
	practical applications like optical character recognition and document processing.''')

st.image('./0_u5-PcKYVfUE5s2by.gif')


model_ann = ann()

st.title(' Dibuje un numero del 0-9. ')
sample_img = grafic() 

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


if np.all( np.reshape(sample_img, (28, 28)) == 0):

	st.title("Dibuje algo en el lienzo. ")
else :
	# Show result
	predictions = model_ann(sample_img).numpy()
	arr = tf.nn.softmax(predictions).numpy()[0]
	dict_arr = dict(zip(range(len(arr)), arr.tolist()))
	d_ordenado = dict(sorted(dict_arr.items(), key=lambda x: x[1], reverse=True))

	st.title( f" { round( list(d_ordenado.values())[0]*100,2) } % de ser [ {list(d_ordenado.keys())[0]} ] "  ) 

	# for key, value in d_ordenado.items():

	# 	st.write( 'El numero que ingresaste tiene un ' , value*100 , 'de ser' ,key )

	# st.write(np.reshape(sample_img, (28, 28)) )

	# # Save in doc
	# imagen = Image.fromarray(np.uint8( np.reshape(sample_img *255,(28,28)) ))
	# imagen.save("lienzoIMG.png")

	# im = Image.fromarray(np.uint8( np.reshape( x_train[17:18] *255,(28,28)) ))
	# im.save("mnist_data.png")


	# xaux=x_test
	# yauc=y_test
	# for n in range(0,50)  :
	#     predictions = model_ann(xaux[n:n+1]).numpy()
	#     arr = tf.nn.softmax(predictions).numpy()[0]
	#     dict_arr = dict(zip(range(len(arr)), arr.tolist()))
	#     d_ordenado = dict(sorted(dict_arr.items(), key=lambda x: x[1], reverse=True))

	#     st.write('<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
	#     for key, value in d_ordenado.items():
	#         st.write( "El valor real  [", yauc[n], '] predicho :[',key, "]on", value*100,'probabilidad','\n')
	#     st.write(np.reshape(xaux[n:n+1], (28, 28)) )


github_code = {'Convolutional Neural Network Code' : 'https://github.com/BlasFerreira/TensorFlowMNIST/blob/main/model_trained.py'}


for project, link in github_code.items():
    st.write(f"[{project}]({link})")