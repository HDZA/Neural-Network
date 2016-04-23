import Node
import math
from twisted.python.util import println
class NeuralNetwork():

    #The manager of the node objects. Will create a basic instance of the NN I have in mind.
    def __init__(self):
        self = self
        self.setWeights()
    neuralNetwork = [Node.Node() for i in range(7)]
    weights = [0 for x in range(9)]
    gradients = [0.00 for x in range(len(weights))] #Tie this value to the number of weights. There will always be a node gradient for every weight
    collectiveGradients = [0.00 for x in range(len(weights))]
    weightChange = [0 for x in range(len(weights))]
    previousWeight = [0 for x in range(len(weights))]
    learningRate = 0.7
    momentium = 0.3
    
    
    def setWeights(self): #For the sake of being able to debug stuff easily i'm going to copy the example weights shown in the video
        self.weights[0] =  0.09750161997450091
        self.weights[1] = -0.4635107399577998
        self.weights[2] =  0.461587116462548
        self.weights[3] =  0.22341077197888182
        self.weights[4] =  0.9487814395569221
        self.weights[5] = -0.06782947598673161
        self.weights[6] =  0.7792991203673414
        self.weights[7] =  0.581714099641357
        self.weights[8] = -0.22791948943117624
        
    def beginTraining(self, training_data):
        #imagine diagram with one clockwise turn and read from left to right.
        #The output layer
        self.resetCollectiveGradients() #Reset the gradients otherwise my training section will be messed up after the first iteration
        MSEInputs = [] #Rather than
        for training_set in training_data:
            self.neuralNetwork[0].output = 1 #B1
            self.neuralNetwork[1].output = training_set[1]#I2
            self.neuralNetwork[2].output = training_set[0] #I1
            
            #The hidden layer
            #Calculations for H2
            self.neuralNetwork[4].sum = ((self.neuralNetwork[0].output * self.weights[0])+(self.neuralNetwork[1].output * self.weights[2])+(self.neuralNetwork[2].output*self.weights[4]))
            self.neuralNetwork[4].output = self.sigmoid(self.neuralNetwork[4].sum)
            
            
            #Calculations for H1
            self.neuralNetwork[5].sum = ((self.neuralNetwork[0].output * self.weights[1])+(self.neuralNetwork[1].output * self.weights[3])+(self.neuralNetwork[2].output*self.weights[5]))
            self.neuralNetwork[5].output = self.sigmoid(self.neuralNetwork[5].sum)
           
            
            #Calculations for O1
            self.neuralNetwork[6].sum = ((self.neuralNetwork[3].output * self.weights[6])+(self.neuralNetwork[4].output * self.weights[7])+(self.neuralNetwork[5].output*self.weights[8]))
            self.neuralNetwork[6].output = self.sigmoid(self.neuralNetwork[6].sum)
            
            #Section that calculates node deltas
            error = self.neuralNetwork[6].output - training_set[2]
            
            self.neuralNetwork[6].nodeDelta = -1*error*(self.dSigMoid(self.neuralNetwork[6].sum)) #01
            self.neuralNetwork[5].nodeDelta = self.dSigMoid(self.neuralNetwork[5].sum) * (self.weights[8]) * (self.neuralNetwork[6].nodeDelta) #H2
            self.neuralNetwork[4].nodeDelta = self.dSigMoid(self.neuralNetwork[4].sum) * (self.weights[7]) * (self.neuralNetwork[6].nodeDelta) #This should be equal to -0.005 h2

            
            
            #Calculate gradients
            self.gradients[8] = self.neuralNetwork[5].output * self.neuralNetwork[6].nodeDelta #This should equal 0.016 H1->O1
            self.gradients[7] = self.neuralNetwork[4].output * self.neuralNetwork[6].nodeDelta #This should equal 0.016 H2->O1
            self.gradients[6] = self.neuralNetwork[3].output * self.neuralNetwork[6].nodeDelta #B1->O1
            self.gradients[5] = self.neuralNetwork[2].output * self.neuralNetwork[5].nodeDelta #I1->H1
            self.gradients[4] = self.neuralNetwork[2].output * self.neuralNetwork[4].nodeDelta #I1->H2
            self.gradients[3] = self.neuralNetwork[1].output * self.neuralNetwork[5].nodeDelta #I2->H1
            self.gradients[2] = self.neuralNetwork[1].output * self.neuralNetwork[4].nodeDelta #I2->H2
            self.gradients[1] = self.neuralNetwork[0].output * self.neuralNetwork[5].nodeDelta #B1->H1
            self.gradients[0] = self.neuralNetwork[0].output * self.neuralNetwork[4].nodeDelta #B1->H2
           
            
            #We're going with a batch training approach. So save all the gradients
            
            for gradient in range(0,len(self.gradients)): 
                self.collectiveGradients[gradient] += self.gradients[gradient] #TODO: Double check that the weights and the gradients you generated line up correctly. I think there might be an error here.
                
            MSEInputs.append(math.pow(training_set[2]-self.neuralNetwork[6].output, 2))
            
        #Training section
        self.printMSE(MSEInputs)
        for index in range(0,len(self.collectiveGradients)):
            self.weights[index]+=self.changeInWeight(self.collectiveGradients[index], self.previousWeight[index])
            self.previousWeight[index] = self.changeInWeight(self.collectiveGradients[index], self.previousWeight[index])
        
        
    def calculateForOneInput(self, input1, input2):
            self.neuralNetwork[0].output = 1 #B1
            self.neuralNetwork[1].output = input2#I2
            self.neuralNetwork[2].output = input1 #I1
            
            #The hidden layer
            self.neuralNetwork[3].output = 1
            
            #Calculations for H2
            self.neuralNetwork[4].sum = ((self.neuralNetwork[0].output * self.weights[0])+(self.neuralNetwork[1].output * self.weights[2])+(self.neuralNetwork[2].output*self.weights[4]))
            self.neuralNetwork[4].output = self.sigmoid(self.neuralNetwork[4].sum)
            
            #Calculations for H1
            self.neuralNetwork[5].sum = ((self.neuralNetwork[0].output * self.weights[1])+(self.neuralNetwork[1].output * self.weights[3])+(self.neuralNetwork[2].output*self.weights[5]))
            self.neuralNetwork[5].output = self.sigmoid(self.neuralNetwork[5].sum)
            
            #Calculations for O1
            self.neuralNetwork[6].sum = ((self.neuralNetwork[3].output * self.weights[6])+(self.neuralNetwork[4].output * self.weights[7])+(self.neuralNetwork[5].output*self.weights[8]))
            self.neuralNetwork[6].output = self.sigmoid(self.neuralNetwork[6].sum)
            print("The output is: " + str(int(round(self.neuralNetwork[6].output)))) #Even with a massive training loop my outputs will never be whole numbers on their own. Help that along by rounding.
            println()
    def setInputs(self,input1, input2):
        self.input1 = input1;
        self.input2 = input2;

    
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))
    
    def dSigMoid(self,x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    
    def changeInWeight(self, gradient, previousWeightAdjustment):
        return self.learningRate * gradient + self.momentium * previousWeightAdjustment
    
    def printWeights(self):
        print(".............")
        for weight in self.weights:
            print(weight)   
    
    def resetCollectiveGradients(self):
        for index in range(0, len(self.collectiveGradients)):
            self.collectiveGradients[index] = 0.0
    def printMSE(self, MSEInputs):
            print("The MSE is: " + str((MSEInputs[0] + MSEInputs[1] + MSEInputs[2] + MSEInputs[3])/len(MSEInputs)))