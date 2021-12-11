
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


nSamples = 400
trainSamples = int( 0.7*nSamples)
testSamples = int(0.1*nSamples)
valSamples = int(0.2*nSamples)
refinement = 1
lenGrid = 300
inputShape = 300

def getLoadData():

    sideLoadInp = np.zeros((nSamples, lenGrid, lenGrid))
    upLoadInp = np.zeros((nSamples, lenGrid, lenGrid))
    lowLoadInp = np.zeros((nSamples, lenGrid, lenGrid))

    load = pd.read_csv('Data_CSV/load.csv', delimiter="\t")

    print(load.columns)

    sideLoad = load[' Side Load '].to_numpy()
    upperLoad = load[' Upper Load '].to_numpy()
    lowerLoad = load[' Lower Load '].to_numpy()

    for i in range(0, nSamples):
        sideLoadInp[i][0][:] = sideLoad[i]
        upLoadInp[i][0][:] = upperLoad[i]
        lowLoadInp[i][0][:] = lowerLoad[i]


    return sideLoadInp, upLoadInp, lowLoadInp




def SplitData ():

    maskedIndexTotal_shfld =[]


    maskedIndexFile = open('maskedIndexTotal.pkl', 'rb')

    inputData = np.load('inputData.npy')
    outputData = np.load('outputData.npy')
    meanStdStress = np.load('meanStdStress.npy', allow_pickle = True)
    maskedIndexTotal = pickle.load(maskedIndexFile)


    maskedIndexFile.close()

    sideLoadInp, upLoadInp, lowLoadInp = getLoadData()


    print(inputData.shape, outputData.shape, sideLoadInp.shape)

    # for i in range(0, nSamples):
    #     print(str(len(maskedIndexTotal[i][0]))+"\t"+str(len(maskedIndexTotal[i][0]))+"\n")

    randIndices = np.arange(0, nSamples)
    np.random.shuffle(randIndices)

    inputData = inputData[randIndices][:][:]
    outputData = outputData[randIndices][:][:]
    sideLoadInp = sideLoadInp[randIndices][:][:]
    upLoadInp = upLoadInp[randIndices][:][:]
    lowLoadInp = lowLoadInp[randIndices][:][:]
    meanStdStress_shuffled = meanStdStress[randIndices][:][:]

    for i in randIndices:
        maskedIndexTotal_shfld.append(maskedIndexTotal[i])

    trainInput1 = inputData[0:trainSamples][:][:]
    trainOutput = outputData[0:trainSamples][:][:]

    valInput1 = inputData[trainSamples:valSamples+trainSamples][:][:]
    valOutput = outputData[trainSamples:valSamples+trainSamples][:][:]

    testInput1 = inputData[valSamples+trainSamples: valSamples+trainSamples+testSamples][:][:]
    testOutput = outputData[valSamples+trainSamples: valSamples+trainSamples+testSamples][:][:]


    trainInput2_1 = sideLoadInp[0:trainSamples][:][:]
    trainInput2_2 = upLoadInp[0:trainSamples][:][:]
    trainInput2_3 = lowLoadInp[0:trainSamples][:][:]

    valInput2_1 = sideLoadInp[trainSamples:valSamples + trainSamples][:][:]
    valInput2_2 = upLoadInp[trainSamples:valSamples + trainSamples][:][:]
    valInput2_3 = lowLoadInp[trainSamples:valSamples + trainSamples][:][:]

    testInput2_1 = sideLoadInp[valSamples + trainSamples: valSamples + trainSamples + testSamples][:][:]
    testInput2_2 = upLoadInp[valSamples + trainSamples: valSamples + trainSamples + testSamples][:][:]
    testInput2_3 = lowLoadInp[valSamples + trainSamples: valSamples + trainSamples + testSamples][:][:]


    return trainInput1,trainInput2_1, trainInput2_2, trainInput2_3, trainOutput, valInput1, valInput2_1, valInput2_2, valInput2_3, valOutput, testInput1,testInput2_1, testInput2_2, testInput2_3, testOutput, maskedIndexTotal_shfld, meanStdStress_shuffled





trainInput1,trainInput2_1, trainInput2_2, trainInput2_3, trainOutput, valInput1, valInput2_1, valInput2_2, valInput2_3, valOutput, testInput1,testInput2_1, testInput2_2, testInput2_3, testOutput, maskedIndexTotal_shfld, meanStdStress_shuffled = SplitData()


# print(testInput2.shape)

maskedIndexFile = open('maskedIndexTotal_suffled.pkl', 'wb')

pickle.dump(maskedIndexTotal_shfld, maskedIndexFile)

maskedIndexFile.close()

np.save('trainInput1.npy', trainInput1)
np.save('trainInput2_1.npy', trainInput2_1)
np.save('trainInput2_2.npy', trainInput2_2)
np.save('trainInput2_3.npy', trainInput2_3)
np.save('trainOutput.npy', trainOutput)

np.save('testInput1.npy', testInput1)
np.save('testInput2_1.npy', testInput2_1)
np.save('testInput2_2.npy', testInput2_2)
np.save('testInput2_3.npy', testInput2_3)
np.save('testOutput.npy', testOutput)

np.save('valInput1.npy', valInput1)
np.save('valInput2_1.npy', valInput2_1)
np.save('valInput2_2.npy', valInput2_2)
np.save('valInput2_3.npy', valInput2_3)
np.save('valOutput.npy', valOutput)

np.save('meanStdStress_shuffled.npy', meanStdStress_shuffled)




