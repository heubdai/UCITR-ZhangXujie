import numpy as np

data = np.zeros((1,370))
with open("../datasets/LD2011_2014.txt", 'r') as f:
    firstline = f.readline().split(';')
    line_dim = len(firstline)-1
    for line in f:
        line = np.array(line.split(';')[1:])
        line_data = np.zeros((1,line_dim))
        for i in range(line_dim):
            line_data[0,i]=np.float64(line[i])
        data = np.append(data,line_data,axis=0)

        print(data.shape)