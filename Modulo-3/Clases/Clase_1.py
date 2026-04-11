#Importamos la libreria para las funciones matematicas

import numpy as np

#Esta es la definicion de las calses
# En el escalon unitario la salida es 1 cuando la entrada es mayor o igual a 0
# y 0 en caso contrario
# 

def escalon_unitario(x):
    """Esta función devuelve 1 si x es mayor o igual a 0, y 0 en caso contrario."""
    if x >= 0:
        return 1
    else:
        return 0

#Define aqui la funcion sigmoide

def sigmoide(x):
    """Esta función devuelve el valor de la función sigmoide evaluada en x."""
    return 1 / (1 + np.exp(-x)) 

#define aqui la funcion tangente hiperbólica
# numpy tiene un metodo que calcula esta funcion, se llama tanh

def tangente_hiperbolica(x):
    """Esta función devuelve el valor de la función tangente hiperbólica evaluada en x."""
    return np.tanh(x)   

# Define aqui la funcion ReLU

def relu(x):
    """Esta función devuelve el valor de la función ReLU evaluada en x."""
    return max(0, x)

### Procesamiento dentro del perceptron

#En este bloque programaremos el proceso de un perceptron.Las 2 tareas importantes son:
#1. La multiplicacion de los valores de la entrada (inputs con los pesos y los pesos
# 
#
 
#Cada fila de esta matriz representa una combinacion de entradas, y cada columna representa una entrada diferente. Por ejemplo, la primera fila [0, 0] representa la combinación de entradas donde ambas entradas son 0, la segunda fila [0, 1] representa la combinación de entradas donde la primera entrada es 0 y la segunda entrada es 1, y así sucesivamente.

inputs = np.array([[0,0],[0,1],[1,0],[1,1]])

# los targets son los valores que queremos que el perceptron aprenda a precedir.
targets = np.array([[0],[1],[1],[1]])

#En esta varibale de tipo lista almacenaremos las salidas del perceptron
outputs = []

#Modifica el valor de los pesos (weights) y del bias del perceptron 
#para que al hacer la suma ponderada (o combinacion lineal) y llama
#a la funcion de activacion la salida del perceptron sea igual a los targets

weights = np.array([[1],[-1]])
bias = -0.5

# Para cada entrada ejecuta le procedimiento del perceptron y almacena la salida en la variable outputs
for i in range(targets.shape[0]):
    #Calcula la suma ponderada (o combinacion lineal) de las entradas y los pesos, y luego agrega el bias
    h = np.dot(inputs[i], weights) + bias
    y = escalon_unitario(h)  # Puedes cambiar esta función de activación por sigmoide, tangente_hiperbolica o relu
    outputs.append([h,y])
    
# Imprime las salidas del perceptron
print('Input 1','Input 2','Combinacion lineal','target','Output')
for i in range(len(outputs)):
    print(inputs[i][0],'      ', inputs[i][1],'      ', outputs[i][0][0],'       ', targets[i][0],'       ', outputs[i][1])
    