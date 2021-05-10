'''
    ME674 Soft Computing in Engineering
    Programming Assignment 1
    Predicting Strength of High-Performance Concrete Using Artificial Neural Network

    Name = Mayank Pathania
    Roll No. = 204103314
    Specialization = Machine Design

    Artificial Neural Network with
        Single hidden layer
        Batch mode of training
        Log sigmoid transfer functions
'''


import random
import math

def main():
    print("This file only contains class")
    print("Please run main.py file to test the ANN Model")

class ANN_BATCH:
    def __init__(self, neurons_input, neurons_output, training_patterns, testing_patterns, neurons_hidden, learning_rate, momentum_coeff, tolerance):
        self.__neurons_input = neurons_input
        self.__neurons_output = neurons_output
        self.__training_patterns = training_patterns
        self.__testing_patterns = testing_patterns
        self.__neurons_hidden = neurons_hidden
        self.__learning_rate = learning_rate
        self.__momentum_coeff = momentum_coeff
        self.__tolerance = tolerance

        self.__V = [[random.uniform(-1,1) for column in range(0,self.__neurons_hidden )] for row in range(0,self.__neurons_input + 1)]
        self.__W = [[random.uniform(-1,1) for column in range(0,self.__neurons_output)] for row in range(0,self.__neurons_hidden + 1)]
        self.__max_input = [100000 for x in range(0, self.__neurons_input)]
        self.__min_input = [-100000 for x in range(0, self.__neurons_input)]
        self.__max_output = [100000 for x in range(0, self.__neurons_output)]
        self.__min_output = [-100000 for x in range(0, self.__neurons_output)]

    def train_model(self, input_data, target_values):
        if len(input_data[0]) != self.__neurons_input:
            print("Input Data not compatible for training")
        elif len(target_values[0]) != self.__neurons_output:
            print("Target Values data is not compatible for training")
        elif len(input_data) < self.__training_patterns:
            print("Not enough training patterns")
        else :
            for column in range(0, len(input_data[0])):
                temp = [x[column] for x in input_data]
                self.__max_input[column] = max(temp) + 5
                self.__min_input[column] = min(temp) - 5
            for column in range(0, len(target_values[0])):
                temp = [x[column] for x in target_values]
                self.__max_output[column] = max(temp) + 5
                self.__min_output[column] = min(temp) - 5
            return self.__train(input_data[:self.__training_patterns], target_values[:self.__training_patterns])
        print("returning -1")
        return -1,-1

    def test_model(self, input_data, target_values):
        if len(input_data[0]) != self.__neurons_input:
            print("Input Data not compatible for testing")
        elif len(target_values[0]) != self.__neurons_output:
            print("Target Values data is not compatible for testing")
        elif len(input_data) < self.__testing_patterns:
            print("Not enough testing patterns")
        else :
            return self.__test(input_data[:self.__testing_patterns], target_values[:self.__testing_patterns])
        print("returning -1,-1")
        return -1,-1

    def predict(self, input_data):
        if len(input_data[0]) != self.__neurons_input:
            print("Input Data not compatible with model")
            return -1

        input_data = self.__normalize_input(input_data)
        patterns = len(input_data)
        input_hidden = [[0 for x in range(0,patterns)] for x in range(0, self.__neurons_hidden)]
        output_hidden = [[0 for x in range(0,patterns)] for x in range(0, self.__neurons_hidden)]
        input_output = [[0 for x in range(0,patterns)] for x in range(0, self.__neurons_output)]
        output_output = [[0 for x in range(0,patterns)] for x in range(0, self.__neurons_output)]
        for p in range(0, patterns):
            for j in range(0, self.__neurons_hidden):
                for i in range(1, self.__neurons_input + 1):
                    input_hidden[j][p] += self.__V[i][j]*input_data[p][i - 1]
                input_hidden[j][p] += self.__V[0][j]
                output_hidden[j][p] = self.__transfer_function_hidden(input_hidden[j][p])
            
            for k in range(0, self.__neurons_output):
                for j in range(1, self.__neurons_hidden + 1):
                    input_output[k][p] += self.__W[j][k]*output_hidden[j - 1][p]
                input_output[k][p] += self.__W[0][k]
                output_output[k][p] = self.__denormalize(self.__transfer_function_output(input_output[k][p]),k)
        return output_output

    def save_model_data(self,W_file = "Weights_W.dat", V_file = "Weights_V.dat", I_file = "normalization_data_input.dat", O_file = "normalization_data_output.dat"):
        with open(W_file,"w") as file:
            for row in self.__W:
                file.writelines([str(x)+'\t' for x in row])
                file.writelines('\n')
        with open(V_file,"w") as file:
            for row in self.__V:
                file.writelines([str(x)+'\t' for x in row])
                file.writelines('\n')
        with open(I_file, "w") as file:
            file.writelines([str(x)+'\t' for x in self.__max_input])
            file.writelines('\n')
            file.writelines([str(x)+'\t' for x in self.__min_input])
            file.writelines('\n')
        with open(O_file, "w") as file:
            file.writelines([str(x)+'\t' for x in self.__max_output])
            file.writelines('\n')
            file.writelines([str(x)+'\t' for x in self.__min_output])
            file.writelines('\n')

    def load_model_data(self, W_file = "Weights_W.dat", V_file = "Weights_V.dat", I_file = "normalization_data_input.dat", o_file = "normalization_data_output.dat"):
        with open(W_file, "r") as file:
            self.__W = []
            line = file.readline()
            while len(line) != 0:
                self.__W.append([float(x) for x in line.split('\t')])
                line = file.readline()
        
        with open(V_file, "r") as file:
            self.__V = []
            line = file.readline()
            while len(line) != 0:
                self.__V.append([float(x)  for x in line.split('\t')])
                line = file.readline()
        
        with open(I_file, "r") as file:
            line = file.readline()
            self.__max_input = [float(x) for x in line.split('\t')]
            line = file.readline()
            self.__min_input = [float(x) for x in line.split('\t')]

        with open(o_file, "r") as file:
            line = file.readline()
            self.__max_output = [float(x) for x in line.split('\t')]
            line = file.readline()
            self.__min_output = [float(x) for x in line.split('\t')]

        self.__neurons_hidden = len(self.__V[0])
        self.__neurons_input = len(self.__V - 1)
        self.__neurons_output = len(self.__W[0])

        

    def __transfer_function_hidden(self,x):
        return 1/(1 + math.exp(-x))

    def __transfer_function_output(self,x):
        return 1/(1 + math.exp(-x))

    def __mean_square_error(self, vector):
        return sum([x**2 for x in vector])/len(vector)

    def __normalize_input(self,input_data):
        for column in range(0,len(input_data[0])):
            max_ = self.__max_input[column]
            min_ = self.__min_input[column]
            for row in range(0,len(input_data)):
                input_data[row][column] = 8*(input_data[row][column]-min_)/(max_ - min_) - 4
        return input_data

    def __normalize_target(self,target_values):
        for column in range(0, len(target_values[0])):
            max_ = self.__max_output[column]
            min_ = self.__min_output[column]
            for row in range(0,len(target_values)):
                target_values[row][column] = 0.1 + 0.8*(target_values[row][column]-min_)/(max_ - min_)
        return target_values
    
    def __denormalize(self, x, column):
        return self.__min_output[column] + ((x - 0.1)*(self.__max_output[column] - self.__min_output[column])/0.8)

    def __forward_pass(self,input_data, target_values):
        patterns = len(input_data)
        IH = [[0 for x in range(0,patterns)] for x in range(0, self.__neurons_hidden)]
        OH = [[0 for x in range(0,patterns)] for x in range(0, self.__neurons_hidden)]
        IO = [[0 for x in range(0,patterns)] for x in range(0, self.__neurons_output)]
        OO = [[0 for x in range(0,patterns)] for x in range(0, self.__neurons_output)]
        E = [0 for x in range(0,self.__neurons_output)]
        for p in range(0, patterns):
            for j in range(0, self.__neurons_hidden):
                for i in range(1, self.__neurons_input + 1):
                    IH[j][p] += self.__V[i][j]*input_data[p][i - 1]
                IH[j][p] += self.__V[0][j]
                OH[j][p] = self.__transfer_function_hidden(IH[j][p])
            
            for k in range(0, self.__neurons_output):
                for j in range(1, self.__neurons_hidden + 1):
                    IO[k][p] += self.__W[j][k]*OH[j - 1][p]
                IO[k][p] += self.__W[0][k]
                OO[k][p] = self.__transfer_function_output(IO[k][p])
                E[k] += 0.5*((target_values[p][k] - OO[k][p])**2)
        return ([(x/patterns) for x in E], OO, OH)

    def __train(self,input_data,target_values):
        input_data = self.__normalize_input(input_data)
        target_values = self.__normalize_target(target_values)
        iteration = 0
        delV_momentum = [[0 for column in range(0, self.__neurons_hidden)] for row in range(0, self.__neurons_input + 1)]
        delW_momentum = [[0 for column in range(0, self.__neurons_output)] for row in range(0, self.__neurons_hidden + 1)]
        E = [100000]
        OO = []
        convergence = []
        while self.__mean_square_error(E) > self.__tolerance:
            (E, OO, OH) = self.__forward_pass(input_data, target_values)
            delV = [[0 for column in range(0, self.__neurons_hidden)] for row in range(0, self.__neurons_input + 1)]
            delW = [[0 for column in range(0, self.__neurons_output)] for row in range(0, self.__neurons_hidden + 1)]

            for p in range(0, self.__training_patterns):
                for k in range(0, self.__neurons_output):
                    val = (target_values[p][k] - OO[k][p])*OO[k][p]*(1 - OO[k][p])
                    for j in range(0, self.__neurons_hidden):
                        for i in range(0, self.__neurons_input):
                            delV[i + 1][j] += (val*self.__W[j][k]*OH[j][p]*(1 - OH[j][p])*input_data[p][i])
                        delV[0][j] += (val*self.__W[j][k]*OH[j][p]*(1 - OH[j][p]))
                        delW[j + 1][k] += (val*OH[j][p])
                    delW[0][k] += val
                
            for k in range(0, self.__neurons_output):
                for j in range(0, self.__neurons_hidden + 1):
                    val = (delW[j][k]*self.__learning_rate)/self.__training_patterns
                    self.__W[j][k] += val + self.__momentum_coeff*delW_momentum[j][k]
                    delW_momentum[j][k] = val

            for i in range(0, self.__neurons_input + 1):
                for j in range(0, self.__neurons_hidden):
                    val = (delV[i][j]*self.__learning_rate)/(self.__training_patterns*self.__neurons_output)
                    self.__V[i][j] += val + self.__momentum_coeff*delV_momentum[i][j]
                    delV_momentum[i][j] = val
            iteration += 1
            convergence.append([iteration, self.__mean_square_error(E)])
            if iteration%100 == 0: print("iteration:  ",iteration,";\tMSE:  ",self.__mean_square_error(E),";")
        for i in range(0,len(OO)):
            for j in range(0,len(OO[0])):
                OO[i][j] = self.__denormalize(OO[i][j],i)
                target_values[j][i] = self.__denormalize(target_values[j][i],i)
        return (convergence, OO)

    def __test(self,input_data, target_values):
        input_data = self.__normalize_input(input_data)
        target_values = self.__normalize_target(target_values)
        (error, OO, OH) = self.__forward_pass(input_data, target_values)
        for i in range(0,len(OO)):
            for j in range(0,len(OO[0])):
                OO[i][j] = self.__denormalize(OO[i][j],i)
                target_values[j][i] = self.__denormalize(target_values[j][i],i)
        return (self.__mean_square_error(error),OO)


if __name__ == '__main__':
    main()
