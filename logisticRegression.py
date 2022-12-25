import numpy as np
import pandas as pd
# from scipy.optimize import minimize

def main():
    
    # TODO put the data file path in path variable 
    path = "" 
    columnNames  = ["age", "salary", "purchased"]

    data = readDataFromCsvFile(path, columnNames)

    datadataRowsCountCount = data.shape[0]
    dataColumnsCount = data.shape[1]
    xValues = extract_X_ValuesFromRecords(data, datadataRowsCountCount, dataColumnsCount)
    yValues = extract_Y_ValuesFromRecords(data, datadataRowsCountCount, dataColumnsCount)
    changeColumnsType(xValues, columnNames[0:2])
    changeColumnsType(yValues, columnNames[2:])
    xValues = addColumn(xValues,0,"Ones", 1)
    dataRowsCount = xValues.shape[0]
    parametersCount = xValues.shape[1]

    all_theta = np.zeros((1, parametersCount))
    theta = np.zeros(parametersCount)

    y_0 = np.array([1 if label == 0 else 0 for label in yValues.iloc[:,0]])
    y_0 = np.reshape(y_0, (dataRowsCount, 1))

    all_theta = one_vs_all(xValues, yValues, 1, 0.3)

    y_pred = predict_all(np.matrix(xValues), np.matrix(all_theta))
    correct = [1 if a == b else 0 for (a, b) in zip(y_pred, yValues.iloc[:, 0])]
    accuracy = (sum(map(int, correct)) / float(len(correct)))

    print("Are customer interested in purchasing the new car ? ")
    print(areCustomerInterestedInPurshasing(y_pred))
    print("=========================================")
    print ('accuracy = {0}%'.format(calculateAccurcy(np.matrix(yValues), y_pred) * 100))

def calculateSegmoidValueOf(z):
    return 1 / (1 + np.exp(-z))

def calculateCostValueOf(thetaValues, xValues, yValues, learningRate):
    sigmoidOfXmultiplyByTheta = calculateSegmoidValueOf(xValues * thetaValues.T)
    first = np.multiply(-yValues, np.log(sigmoidOfXmultiplyByTheta))
    second = np.multiply((1 - yValues), np.log(1 - sigmoidOfXmultiplyByTheta))
    sumOfThetaPowerTwoValues = np.sum(np.power(thetaValues[:, 1: thetaValues.shape[1]], 2))
    reg = (learningRate / 2 * len(xValues)) * sumOfThetaPowerTwoValues
    return np.sum(first - second) / (len(xValues)) + reg

def gradient_with_loop(theta, xValues, yValues, learningRate):
    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)
    
    error = calculateSegmoidValueOf(xValues * theta.T) - yValues
    
    for i in range(parameters):
        term = np.multiply(error, xValues[:,i])
        
        if (i == 0):
            grad[i] = np.sum(term) / len(xValues)
        else:
            grad[i] = (np.sum(term) / len(xValues)) + ((learningRate / len(xValues)) * theta[:,i])
    
    return grad

def gradient(theta, xValues, yValues, learningRate):
    error = calculateSegmoidValueOf(xValues * theta.T) - yValues
    grad = ((xValues.T * error) / len(xValues)).T + ((learningRate / len(xValues)) * theta)
    grad[0, 0] = np.sum(np.multiply(error, xValues[:,0])) / len(xValues)
    
    return np.array(grad).ravel()

def one_vs_all(xValues, yValues, num_labels, learning_rate):
    
    dataRowsCount = xValues.shape[0] 
    parametersCount = xValues.shape[1]
    all_theta = np.zeros((num_labels, parametersCount))
    
    for i in range(1, num_labels + 1):
        theta = np.zeros(parametersCount)
        y_i = np.array([1 if label == i else 0 for label in yValues.iloc[:, 0]])
        y_i = np.reshape(y_i, (dataRowsCount, 1))

    #     # minimize the objective function
    #     fmin = minimize(fun=calculateCostValueOf, x0=theta, args=(xValues, y_i, learning_rate), method='TNC', jac=gradient)
    #     all_theta[i-1,:] = fmin.x
    
    return all_theta

def predict_all(xValues, all_theta):
    dataRowsCount = xValues.shape[0]
    parametersCount = xValues.shape[1]
    num_labels = all_theta.shape[0]
    
    h = calculateSegmoidValueOf(xValues * all_theta.T)
    
    h_argmax = np.argmax(h, axis=1)
    h_argmax = h_argmax + 1
    
    return h_argmax

def readDataFromCsvFile(path, listOfColumnsNames):
    return pd.read_csv(path, header = None, names = listOfColumnsNames)

def changeColumnsType(modelRecords, columnsName):
    for name in columnsName:
        modelRecords[name] = modelRecords[name].astype(str).astype(int)

def extract_X_ValuesFromRecords(modelRecords, recordsLength, columnsLength):
    return modelRecords.iloc[1: recordsLength, 0: columnsLength - 1] 

def extract_Y_ValuesFromRecords(modelRecords, recordsLength, columnsLength):
    return modelRecords.iloc[1: recordsLength, columnsLength - 1: columnsLength]

def addColumn(modelRecords, position, columnName, value):
    modelRecords.insert(position, columnName, value)
    return modelRecords

def calculateAccurcy(yValues, yPredicted):
    sumOfCurrectPrediction = 0
    for (realValue, predictedValue) in zip(yValues[:, 0], yPredicted):
        if(realValue == predictedValue):
            sumOfCurrectPrediction += 1
    return sumOfCurrectPrediction / float(yValues.shape[0])

def areCustomerInterestedInPurshasing(predictedValues):
    yes = 0
    no = 0
    for i in predictedValues[:, 0]:
        if(i == 1):
            yes += 1
        else: 
            no += 1
    return "yes" if (yes >= no) else "no"

if __name__ == "__main__":
    main()