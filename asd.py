import numpy as np
import os
import cv2 as ocv
import pickle as pk


img_c = []
img_b = []


class neuronal_network:
    def __init__ (self):
        self.weights = None
        self.inputs = None
    
    def init_pes (self, input_size):
        self.weights = np.random.rand(input_size)

    def forward_propagation (self, inputs):
        output = np.dot(inputs.reshape(-1), self.weights)
        return output
    
    def entreanamiento(self, expected_output, learning_rate = 0.01, epochs = None):
        for epoch in range(epochs):
            output = self.forward_propagation(self.inputs)
            error = expected_output - output 
            self.weights += learning_rate * error * self.inputs
    
    def guardado(self, nombre):
        with open(nombre, "wb") as file:
            pk.dump(self.weights, nombre)


path_c = "C:\\Users\\GAME\\Desktop\\aumento de datos\\perros\\"
path_b = "C:\\Users\\GAME\\Desktop\\aumento de datos\\grises\\"

for a in os.listdir(path_c):
    img1 = ocv.imread(path_c + a)
    img1 = ocv.resize(img1, (200, 200))
    img1 = np.asarray(img1)
    img_c.append(img1)

for b in os.listdir(path_b):
    img2 = ocv.imread(path_b + b)
    img2 = ocv.resize(img2, (200, 200))
    img2 = np.asarray(img2)
    img_b.append(img2)

red_n = neuronal_network()

pesos = red_n.init_pes(200 * 200 * 3)

for i in range(len(img_c)):
    init = red_n.forward_propagation(img_c[i])
    entrenamiento = red_n.entreanamiento(img_b[i], epochs = 1000)

red_n.guardado("charmander.pkl")

