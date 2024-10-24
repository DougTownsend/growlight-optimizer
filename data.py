import numpy as np
import matplotlib.pyplot as plt
import os

def sort_by_column(data, col):
    retval = np.copy(data)
    idx = np.argsort(retval[:,col])
    retval = retval[idx]
    return retval

def filter_oversample(data):
    filtered_data = []
    current_xval = float("inf")
    y_sum = 0
    count = 0
    data = sort_by_column(data, 0)
    for i in range(len(data)):
        print(i)
        if data[i][0] != current_xval:
            if current_xval != float("inf"):
                filtered_data.append([current_xval, y_sum / float(count)])
            current_xval = data[i][0]
            count = 1
            y_sum = data[i][1]
        else:
            y_sum += data[i][1]
            count += 1 
    filtered_data.append([current_xval, y_sum / float(count)])
    filtered_data = np.asarray(filtered_data)
    median_filtered = []
    for i in range(3, len(filtered_data)-3):
        median_filtered.append([filtered_data[i][0], np.median(filtered_data[i-3:i+3, 1])])
    filtered_data = np.asarray(median_filtered)
    return filtered_data

def linear_fill(data, start, stop, step):
    idx = np.argsort(data[:,0])
    data = data[idx]
    out = []
    x = np.linspace(start, stop, int(abs(start-stop)/step) + 1, endpoint=True)
    print(x)
    for xval in x:
        if xval <= data[0,0]:
            out.append([xval,data[0,1]])
        elif xval >= data[-1,0]:
            out.append([xval,data[-1,1]])
        else:
            idx_above = 0
            for i in range(len(data)):
                if data[i,0] > xval:
                    idx_above = i
                    break
            i = idx_above - 1
            dy = data[i+1,1] - data[i,1]
            dx = data[i+1,0] - data[i,0]
            y_int = data[i,1] - ((dy/dx) * data[i,0])
            out.append([xval, ((dy/dx) * xval) + y_int])
    return np.asarray(out)


lm301_mult = 1.51
deepred_mult = 0.83
lm301_count = 400
deepred_count = 80

lm301_data = np.genfromtxt('./data/lm301h_evo_4000K_spectrum_norm.csv', delimiter=',')
deepred_data = np.genfromtxt('./data/660nm_spectrum_norm.csv', delimiter=',')
lm301_data[:,1] *= lm301_mult * lm301_count
deepred_data[:,1] *= deepred_mult * deepred_count

out_data = np.copy(lm301_data)
deepred_offset = 0
for i in range(len(out_data)):
    if out_data[i,0] == deepred_data[0,0]:
        deepred_offset = i
        break
print(out_data[deepred_offset,0])
print(deepred_data[0,0])
for i in range(len(deepred_data)):
    out_data[i+deepred_offset,1] += deepred_data[i,1]

plt.scatter(out_data[:,0], out_data[:,1])
plt.show()  
