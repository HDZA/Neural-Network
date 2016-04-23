'''
Created on Apr 14, 2016

@author: nox
'''
import NeuralNetwork

if __name__ == '__main__':
    pass
training_data = [[0 for x in range(3)] for x in range(4)]
training_data[0][0] = 0
training_data[0][1] = 0
training_data[0][2] = 0
        
training_data[1][0] = 1
training_data[1][1] = 0
training_data[1][2] = 1
        
training_data[2][0] = 0
training_data[2][1] = 1
training_data[2][2] = 1
        
training_data[3][0] = 1
training_data[3][1] = 1
training_data[3][2] = 0

myTestNN = NeuralNetwork.NeuralNetwork()
for i in range(0,1000):
    myTestNN.beginTraining(training_data)
myTestNN.calculateForOneInput(1, 0)



