import numpy as np

""""
@author: Gustavo Machado
"""

class RedeNeural():
    def __init__(self, inputs, outputs):
        self.__inputs = np.array(inputs)
        self.__outputs = np.array(outputs)
        self.__weights0 = 2*np.random.random((2,3)) - 1
        self.__weights1 = 2*np.random.random((3,1)) - 1

    def getInput(self):
        return self.__inputs
    
    def getOutputs(self):
        return self.__outputs
    
    def getWeights0(self):
        return self.__weights0
    
    def getWeights1(self):
        return self.__weights1
    
    def setWeights1(self, value):
        self.__weights1 = value
    
    def setWeights0(self, value):
        self.__weights0 = value

    def sigmoid(self, sum):
        return 1/(1+np.exp(-sum))

    def sigmoidDerivated(self, sig):
        return sig*(1-sig)

    def SumWeights0(self, weights0, InputLayer):
        weights0 = self.getWeights0()
        return np.dot(InputLayer, weights0)

    def HidenLayer(self, SumWeights0):
        return self.sigmoid(SumWeights0)

    def SumWeights1(self, HidenLayer, weights1):
        return np.dot(HidenLayer, weights1)
        
    def OutputLayer(self, SumWeights1):
        return self.sigmoid(SumWeights1)

    def OutputCalc(self, InputLayer):
        sumWeights0 = self.SumWeights0(self.getWeights0(), InputLayer)
        hidenLayer = self.HidenLayer(sumWeights0)
        sumWeights1 = self.SumWeights1(hidenLayer, self.getWeights1())
        outputLayer =  self.OutputLayer(sumWeights1)
        return outputLayer[0]

    def train(self, epochs = 100000, LearningRate=0.5, Momentum=1):
        for j in range(epochs):
            sumWeights0 = self.SumWeights0(self.getWeights0(), self.getInput())
            hidenLayer = self.HidenLayer(sumWeights0)
            sumWeights1 = self.SumWeights1(hidenLayer, self.getWeights1())
            outputLayer =  self.OutputLayer(sumWeights1)

            MistakeRateOutput = self.getOutputs() - outputLayer
            Average = np.mean(np.abs(MistakeRateOutput))

            DerivatedOutput = self.sigmoidDerivated(outputLayer)
            deltaOutput = MistakeRateOutput*DerivatedOutput

            weights1T = self.getWeights1().T
            deltaOutputWithWeights = deltaOutput.dot(weights1T)
            deltaHidenLayer = deltaOutputWithWeights * self.sigmoidDerivated(hidenLayer)

            HidenLayerT = hidenLayer.T
            NewWeights = HidenLayerT.dot(deltaOutput)
            self.setWeights1((self.getWeights1()*Momentum)+(NewWeights*LearningRate))

            InputLayerT = self.getInput().T
            NewWeights0 = InputLayerT.dot(deltaHidenLayer)
            self.setWeights0((self.getWeights0()*Momentum) + (NewWeights0*LearningRate))
        
        print("Taxa de erro = " + str(Average))


XOR = RedeNeural([[0.0,0.0],[0.0,1.0], [1.0,0.0], [1.0,1.0]], [[0],[1],[1],[0]])

XOR.train()
print("Neural Network trained")
print(XOR.OutputCalc(XOR.getInput()[0]))
print(XOR.OutputCalc(XOR.getInput()[1]))
print(XOR.OutputCalc(XOR.getInput()[2]))
print(XOR.OutputCalc(XOR.getInput()[3]))