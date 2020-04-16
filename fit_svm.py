import sys, numpy as np
sys.path.insert(1, './libsvm/python/')
from svmutil import *
sys.path.insert(1, './libsvm/tools/')
from grid import find_parameters


def confusion_matrix(true, pred):
    #Gets unique classes
    classes = np.unique(true)
    K = len(classes)
    #Maps Classes with appropriate index for Confusion Matrix
    classInd = {}
    for i in range(K):
        classInd[classes[i]] = i
    result = np.zeros((K, K))
    #Computers Confusion Matrix
    for i in range(len(true)):
        result[ classInd[true[i]] ][ classInd[pred[i]] ] += 1

    return result



#Read in Data
trainRAD_y, trainRAD_x = svm_read_problem('./rad_d2')
trainCust_y, trainCust_x = svm_read_problem('./cust_d2')
testRAD_y, testRAD_x = svm_read_problem('./rad_d2.t')
testCust_y, testCust_x = svm_read_problem('./cust_d2.t')


#Perform Cross Validation to find Best Parameters
print('INITIALIZING CROSS VALIDATION')
bestRAD_acc, bestRAD_params = find_parameters('./rad_d2')
print('Best Accuracy with RAD is:', bestRAD_acc,'\nWith C and y:', bestRAD_params)
bestCust_acc, bestCust_params = find_parameters('./cust_d2')
print('Best Accuracy with Custom is:', bestCust_acc,'\nWith C and y:', bestCust_params)

#Perform SVM training
print('TRAINING SVMs')
parameter_string = '-s 0 -t 2 -g ' + str(bestRAD_params['g']) + ' -c ' + str(bestRAD_params['c'])
mRAD = svm_train(trainRAD_y, trainRAD_x, parameter_string)
parameter_string = '-s 0 -t 2 -g ' + str(bestCust_params['g']) + ' -c ' + str(bestCust_params['c'])
mCust = svm_train(trainCust_y, trainCust_x, parameter_string)

#Check Accuracy
print('TRAINING COMPLETED.')
print('RAD Data:')
predRAD_labels, predRAD_acc, pRAD_vals = svm_predict(testRAD_y, testRAD_x, mRAD)
print('Custom Data:')
predCust_labels, predCust_acc, pCust_vals = svm_predict(testCust_y, testCust_x, mCust)

#Print Confusion Matrix
print('CONFUSION MATRIX:')
print('RAD:')
print(confusion_matrix(testRAD_y, predRAD_labels)) 
print('Cust:')
print(confusion_matrix(testCust_y, predCust_labels)) 

#print('Predicted Labels:', pred_labels)
#print('Predicted Accuracy:', pred_acc)
#print('Probability Values:', p_vals)
