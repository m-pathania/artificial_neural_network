#%%
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import math
from scipy import stats as st
from sklearn.metrics import r2_score

# %%
def read_parameters(filename):
    with open(filename,'r') as file:
        val1 = float(file.readline().split(' = ')[1])
        val2 = float(file.readline().split(' = ')[1])
        val3 = float(file.readline().split(' = ')[1])
        val4 = float(file.readline().split(' = ')[1])
        
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

def post_processing(filename):
    a = []
    with open(filename,"r") as file:
        line = file.readline()
        line = file.readline()
        while len(line) != 0:
            c = line.split(',')
            a.append([float(c[0]), float(c[1])])
            line = file.readline()

    print((sum([(x[0] - x[1])**2 for x in a]))/len(a))
    b = [abs(x[0] - x[1]) for x in a]
    print("Maximum absolute difference:\t",max(b))
    print("Minimum absolute difference:\t",min(b))
    return a

(neurons_input, neurons_output, training_patterns, testing_patterns) = (int(x) for x in read_parameters('input_parameters.dat'))
(neurons_hidden, learning_rate, momentum_coeff, tolerance) =  read_parameters('model_parameters.dat')
neurons_hidden = int(neurons_hidden)

(training_input, training_target) = import_data('data_training.csv', neurons_input, neurons_output)
(testing_input, testing_target) = import_data('data_testing.csv', neurons_input, neurons_output)

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

a = post_processing("results_testing.dat")
x = [x[0] for x in a]
y = [x[1] for x in a]
diff = [abs(x[0] - x[1]) for x in a]
plt.scatter(x,y,c=diff,cmap = 'Reds',
    s=15,edgecolors='black')
cbar = plt.colorbar()
cbar.set_label('Absolute Difference')
plt.xlabel('Target Values\n(Strength in MPa)')
plt.ylabel('Output Values\n(Strength in MPa)')
plt.xlim([0,90])
plt.ylim([0,90])
plt.savefig('Comparison.png',bbox_inches='tight',pad_inches=0.5,transparent='true')


# %%
itr = []
err = []
with open('results_training.dat','r') as file:
    line = file.readline()
    line = file.readline()
    while len(line) > 0:
        itr.append(float(line.split(',')[0]))
        err.append(float(line.split(',')[1]))
        line = file.readline()

plt.plot(itr,err)
plt.title("Convergence Plot")
plt.xlabel('Iterations')
plt.ylabel('Mean Square Error \n(Normalized Values)')
plt.savefig('Convergence_Plot.png',bbox_inches='tight',pad_inches=0.5,transparent='true')
# %%
plt.scatter([x[0] for x in training_input],training_target,
c='green',s=15,edgecolors='black')
plt.xlabel('Cement')
plt.ylabel('Strength')
plt.savefig('Plot_01.png',bbox_inches='tight',pad_inches=0.5,transparent='true')
# %%
plt.scatter([x[1] for x in training_input],training_target,
c='green',s=15,edgecolors='black')
plt.xlabel('Blast Furnace Slag')
plt.ylabel('Strength')
plt.savefig('Plot_02.png',bbox_inches='tight',pad_inches=0.5,transparent='true')
# %%
plt.scatter([x[2] for x in training_input],training_target,
c='green',s=15,edgecolors='black')
plt.xlabel('Fly Ash')
plt.ylabel('Strength')
plt.savefig('Plot_03.png',bbox_inches='tight',pad_inches=0.5,transparent='true')
# %%
plt.scatter([x[3] for x in training_input],training_target,
c='green',s=15,edgecolors='black')
plt.xlabel('Water')
plt.ylabel('Strength')
plt.savefig('Plot_04.png',bbox_inches='tight',pad_inches=0.5,transparent='true')
# %%
plt.scatter([x[4] for x in training_input],training_target,
c='green',s=15,edgecolors='black')
plt.xlabel('Superplasticizer')
plt.ylabel('Strength')
plt.savefig('Plot_05.png',bbox_inches='tight',pad_inches=0.5,transparent='true')
# %%
plt.scatter([x[5] for x in training_input],training_target,
c='green',s=15,edgecolors='black')
plt.xlabel('Course Aggregate')
plt.ylabel('Strength')
plt.savefig('Plot_06.png',bbox_inches='tight',pad_inches=0.5,transparent='true')
# %%
plt.scatter([x[6] for x in training_input],training_target,
c='green',s=15,edgecolors='black')
plt.xlabel('Fine Aggregate')
plt.ylabel('Strength')
plt.savefig('Plot_07.png',bbox_inches='tight',pad_inches=0.5,transparent='true')
# %%
plt.scatter([x[7] for x in training_input],training_target,
c='green',s=15,edgecolors='black')
plt.xlabel('Age')
plt.ylabel('Strength')
plt.savefig('Plot_08.png',bbox_inches='tight',pad_inches=0.5,transparent='true')
# %%
data = pd.read_csv("data_training.csv")
data.head()
# %%
d = np.array(data)
# %%
for i in range(0,9):
    print(np.array([np.min(d[:,i]),np.max(d[:,i]),np.mean(d[:,i]),np.median(np.sort(d[:,i]))]))
# %%
data = pd.read_csv("results_testing.dat",header=1)
data.head()
# %%
x = np.array(data.iloc[:,1])
y = np.array(data.iloc[:,0])
# %%
score = r2_score(y,x)
print(score)
# %%
