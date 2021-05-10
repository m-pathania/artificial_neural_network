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

import math
import random

from model import ANN_BATCH

def main():
    # Reading Parameters from file
    try:
        filename = 'input_parameters.dat'
        (neurons_input, neurons_output, training_patterns, testing_patterns) = (int(x) for x in read_parameters(filename))
        print("Reading Input parameters from '",filename,"' file")
    except:
        print("For present case:\t===>\nInputs = 8\nOutputs = 1\nTraining Patterns = 600\nTesting Patterns = 200")
        neurons_input = int(input("\nEnter Number of Inputs:\t"))
        neurons_output = int(input("Enter Number of Outputs:\t"))
        training_patterns = int(input("Enter Number of Training Patterns:\t"))
        testing_patterns = int(input("Enter Number of Testing Patterns:\t"))
    
    try:
        filename = 'model_parameters.dat'
        (neurons_hidden, learning_rate, momentum_coeff, tolerance) =  read_parameters(filename)
        print("Reading Model parameters from '",filename,"' file")
    except:
        neurons_hidden = 20
        learning_rate = 0.9
        momentum_coeff = 0.6
        tolerance = float(5e-6)
    
    neurons_hidden = int(neurons_hidden)

    # Importing testing and training data
    (training_input, training_target) = import_data('data_training.csv', neurons_input, neurons_output)
    (testing_input, testing_target) = import_data('data_testing.csv', neurons_input, neurons_output)

    # Printing parameters to consol
    print("Input Parameters:")
    print("Neurons Input = \t", neurons_input)
    print("Neurons Output = \t", neurons_output)
    print("Training Patterns = \t", training_patterns)
    print("Testing Patterns = \t", testing_patterns)

    print("\nModel Parameters:")
    print("Neurons Hidden = \t", neurons_hidden)
    print("Learning Rate = \t", learning_rate)
    print("Momentum Coefficient = \t", momentum_coeff)
    print("Tolerance/Error = \t", tolerance)

    # Creating ANN Model
    my_model = ANN_BATCH(neurons_input, neurons_output, training_patterns, testing_patterns, neurons_hidden, learning_rate, momentum_coeff, tolerance)

    # Training Model with training Data
    print("MSE is printed every 100 iteration")
    (convergence_data, O_train) = my_model.train_model(training_input, training_target)
    print("\nTraining Completed:")
    print("Iterations :\t",convergence_data[-1][0],"\nMean Square Error :\t",convergence_data[-1][1])

    # Saving model data from using reuse
    my_model.save_model_data()
    
    # Testing Model
    (error, O_testing) = my_model.test_model(testing_input, testing_target)
    print("\nTesting Complete")
    print("Mean Square Error :",error)

    # writing data to file
    with open("output_training.dat","w") as file:
        file.writelines("Target Value,Predicted Value\n")
        for i in range(0,len(O_train[0])):
            file.writelines(str(training_target[i][0])+","+str(O_train[0][i])+"\n")

    with open("convergence_data.dat","w") as file:
        file.writelines("Iterations,MSE\n")
        for x in convergence_data:
            file.writelines(str(x[0])+","+str(x[1])+"\n")

    with open("output_testing.dat","w") as file:
        file.writelines("Target Value,Predicted Value\n")
        for i in range(0,len(O_testing[0])):
            file.writelines(str(testing_target[i][0])+","+str(O_testing[0][i])+"\n")


def read_parameters(filename):
    with open(filename,'r') as file:
        val1 = float(file.readline().split('=')[-1])
        val2 = float(file.readline().split('=')[-1])
        val3 = float(file.readline().split('=')[-1])
        val4 = float(file.readline().split('=')[-1])
        
    return (val1, val2, val3, val4)


def import_data(filename, inputs, outputs):
    with open(filename,'r') as file:
        input_data = []
        target_value = []
        line = file.readline()
        line = file.readline()
        while len(line) > 0:
            input_data.append([float(x) for x in line.split(',')[:inputs]])
            target_value.append([float(x) for x in line.split(',')[inputs:inputs+outputs]])
            line = file.readline()
    return (input_data, target_value)

if __name__ == '__main__':
    main()
