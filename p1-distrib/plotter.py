
import matplotlib.pyplot as plt
import numpy as np

training_folder = 'mydata/training/'
file_names = ['constant_0-1', 'constant_0-01',  'per_epoch', 'per_step']

for file in file_names:
    path = training_folder + file + '-epoch.csv'
    y = np.loadtxt(path)
    x = np.arange(0, len(y), 1, dtype=int)
    plt.plot(x, y, label=file)

plt.title("Training Dataset Log Likelihood per Epoch")    
plt.legend()

plt.savefig(training_folder + "training_ll.png", dpi=216)
plt.show()



for file in file_names:
    path = training_folder + file + '-dev.csv'
    y = np.loadtxt(path)
    x = np.arange(0, len(y), 1, dtype=int)
    plt.plot(x, y, label=file)

plt.title("Dev Accuracy per Epoch")    
plt.legend()

plt.savefig(training_folder + "dev_acc.png", dpi=216)
plt.show()