import numpy as np
import scipy.io
from scipy.stats import mode

print "Imports successful, fetching data..."

Iowa_MIS_dataset = scipy.io.loadmat('Iowa_MIS_dataset.mat')
data = Iowa_MIS_dataset['dat_all']

print "Received data, processing..."

# general data parameters
N = data.shape[0]
D = data.shape[1]-1
C = 12

np.random.permutation(data)

p_tr, p_vl, p_ts = 0.8, 0.1, 0.1
i_tr = int(np.floor(N*p_tr))
i_vl = i_tr + int(np.floor(N*p_vl))
i_ts = N  #int(np.floor(N*p_ts))
# for i in [p_tr, p_vl, p_ts, i_tr, i_vl, i_ts]:
#     print i
x_tr, x_vl, x_ts = data[:i_tr,:-1], data[i_tr:i_vl,:-1], data[i_vl:i_ts,:-1]
y_tr, y_vl, y_ts = data[:i_tr,-1:], data[i_tr:i_vl,-1:], data[i_vl:i_ts,-1:]

# for i in [data, x_tr, x_vl, x_ts, y_tr, y_vl, y_ts]:
#     print i.shape

print "Processed, initializing knn..."

max_k = 20
norm = "L2"
for k in xrange(1,max_k,2):
    # print "Trying with k = ", k
    num_correct = 0
    for i in xrange(x_vl.shape[0]):

        # calculate the L1 norm between each point that you used to train the model
        # and each point that you use to validate the model.    
        # your code here:
        diffs = np.subtract(x_tr, x_vl[i])
        if norm == "L2":
            diffs = np.square(diffs)
        elif norm == "L1":
            diffs = np.abs(diffs)
        else:
            print "invalid norm"
#         print "diffs ", diffs.shape
        dists = np.sum(diffs, axis=1)
#         print "dists ", dists.shape

        # obtain the indices that would sort the array in ascending order
        # your code here:
        inds = np.argsort(dists)
        k_inds = inds[:k]

        # obtain the labels for the KNNs using their indices
        # your code here:
        labels = y_tr[k_inds]

        # have the neighbors vote [hint: use the scipy function 'mode']
        # your code here:
        predicted_label = mode(labels)
        
        # print "Finished classifying validation number: {}, matched with {}, and labeled as {}. Real label was {}".format(i, k_inds, predicted_label, predicted_label[1][0])

        if  int(predicted_label[1][0]) == int(y_vl[i]):
            num_correct += 1

    print 'accuracy with ', k, 'nearest neighbors: ', float(num_correct)/x_vl.shape[0]        
