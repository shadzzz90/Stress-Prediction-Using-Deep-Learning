import numpy as np
from numpy import save
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import random
import pickle
import matplotlib
import copy
from sklearn.preprocessing import MinMaxScaler, StandardScaler

nPoints = 2000
nSamples = 400
refinement = 1
lenGrid = 300

def genGridData(X,Y,Z):

    X = X.reshape((X.shape[1],))
    Y = Y.reshape((Y.shape[1],))
    Z = Z.reshape((Z.shape[1],))

    print(np.amax(X), np.amax(Y))
    x = np.arange(0, lenGrid, refinement)
    y = np.arange(0, lenGrid, refinement)

    x, y = np.meshgrid(x, y)

    gridZ = griddata((X, Y), Z, (x, y), method='linear')

    loc_NaN = np.isnan(gridZ)

    gridZ[loc_NaN] = 0

    maskIndex = np.asarray(np.where(gridZ == 0))


    return x,y, gridZ, maskIndex

def EllipsePtGenerator():


    totalHolePts = []

    for n in range(0, nSamples):



        dataframe = pd.read_csv("Data_CSV/Job-{}_loc.csv".format(n))

        # print(dataframe.columns)

        centreX = dataframe['CentreX'].to_numpy()
        centreY = dataframe[' CentreY'].to_numpy()
        ellipseRad1 = dataframe[' EllipseRad1'].to_numpy()
        ellipseRad2 = dataframe['EllipseRad2 '].to_numpy()

        allHolePointsX = []
        allHolePointsY = []

        for j in range(len(centreX)):

            randPoints = []

            for i in range(0,nPoints):


                randX = random.uniform(centreX[j] - ellipseRad1[j], centreX[j] + ellipseRad1[j])
                randY = random.uniform(centreY[j] - ellipseRad2[j], centreY[j] + ellipseRad2[j])
                test = ((randX - centreX[j])**2/ellipseRad1[j]**2) + ((randY-centreY[j])**2/ellipseRad2[j]**2)

                if test <= 1:


                    randPoints.append([randX, randY])

                else:
                    continue


            randPointsX = [item[0] for item in randPoints]
            randPointsY = [item[1] for item in randPoints]

            randPointsX = np.asarray(randPointsX)
            randPointsY = np.asarray(randPointsY)

            allHolePointsX.append(randPointsX)
            allHolePointsY.append(randPointsY)

            # allHolePointsX = np.array(allHolePointsX)
            # allHolePointsY = np.array(allHolePointsY)

            # print(allHolePointsX.shape, allHolePointsY.shape)


        totalHolePts.append([allHolePointsX, allHolePointsY])

        del dataframe

    return totalHolePts



def InputDataGenerator(totalHolePts):

    inputData = []
    maskedIndexTotal = []
    gridMeshtotal = []

    for i in range(0, nSamples):

        dataframe = pd.read_csv("Data_CSV/Job-{}.csv".format(i))
        # print(dataframe.columns)

        X = dataframe['X'].to_numpy()
        X = X.reshape((1,X.shape[0]))
        # print(X.shape)
        Y = dataframe['Y'].to_numpy()
        Y = Y.reshape((1,Y.shape[0]))
        # print(Y.shape)
        Ones = np.ones((1,X.shape[1]))

        nTotalHolePts = 0


        allHolePointsX, allHolePointsY = totalHolePts[i]


        for holePtX, holePtY in zip(allHolePointsX, allHolePointsY):

            holePtX = holePtX.reshape((1, holePtX.shape[0]))
            # print(holePtX.shape)

            nTotalHolePts = nTotalHolePts + holePtX.shape[1]


            holePtY = holePtY.reshape((1, holePtY.shape[0]))
            # print(holePtY.shape)

            newX = np.hstack((X,holePtX))
            newY = np.hstack((Y,holePtY))

            X = newX
            Y = newY

        zeros = np.zeros((1, nTotalHolePts))
        newZ = np.hstack((Ones, zeros))
        # plt.scatter(X,Y,newZ)
        # plt.show()
        x,y,gridZ,maskedIndex = genGridData(X,Y,newZ)


        inputData.append(gridZ)
        maskedIndexTotal.append(maskedIndex)


        del dataframe


    print('Input Data Generated')

    return inputData, maskedIndexTotal



def OutputDataGenerator(totalHolePts):

    outputData = []

    meanStdStress = []


    for i in range(0, nSamples):

        dataframe = pd.read_csv("Data_CSV/Job-{}.csv".format(i))
        # print(dataframe.columns)

        X = dataframe['X'].to_numpy()
        X = X.reshape((1,X.shape[0]))
        # print(X.shape)
        Y = dataframe['Y'].to_numpy()
        Y = Y.reshape((1,Y.shape[0]))
        # print(Y.shape)

        S = dataframe['       S-Mises'].to_numpy()

        S = S.reshape((1,S.shape[0]))

        mean = np.mean(S, axis=1)

        std = np.std(S)

        S = (S - mean)/std

        meanStdStress.append([mean, std])



        # scale = StandardScaler()
        #
        # S = scale.fit_transform(S)


        nTotalHolePts = 0

        allHolePointsX, allHolePointsY = totalHolePts[i]

        for holePtX, holePtY in zip(allHolePointsX, allHolePointsY):

            holePtX = holePtX.reshape((1, holePtX.shape[0]))
            # print(holePtX.shape)

            nTotalHolePts = nTotalHolePts + holePtX.shape[1]


            holePtY = holePtY.reshape((1, holePtY.shape[0]))
            # print(holePtY.shape)

            newX = np.hstack((X,holePtX))
            newY = np.hstack((Y,holePtY))

            X = newX[:]
            Y = newY[:]

        zeros = np.zeros((1, nTotalHolePts))
        S_new = np.hstack((S, zeros))
        # plt.scatter(X,Y,newZ)
        # plt.show()
        _, _, gridS, _ = genGridData(X,Y,S_new)


        outputData.append(gridS)

        del dataframe


    print('Output Data Generated')

    return outputData, meanStdStress



def ContourPlotter(inputData, maskedIndexTotal, outputData, meanStdStress):

    for i in range(0,nSamples):

        gridGeo = inputData[i]
        gridS = outputData[i]
        maskIndex = maskedIndexTotal[i]


        # gridGeo[maskIndex[0][:], maskIndex[1][:]] = np.nan
        #
        # gridS[maskIndex[0][:], maskIndex[1][:]] = np.nan

        x = np.arange(0, lenGrid, refinement)
        y = np.arange(0, lenGrid, refinement)

        x, y = np.meshgrid(x, y)


        # print(np.isnan(gridZ).any())

        plt.contourf(x, y, gridGeo, cmap='rainbow')
        plt.colorbar()
        plt.title('Job-{} Binary Input'.format(i))
        plt.savefig('DataImages4/Job-{}_Input.png'.format(i))

        plt.clf()

        plt.contourf(x, y, gridS, cmap='rainbow')
        plt.colorbar()
        plt.title('Job-{} Mises Stress Output'.format(i))
        plt.savefig('DataImages4/Job-{}_Output.png'.format(i))
        plt.clf()
        print('Figure-{} saved '.format(i))



totalHolePts = EllipsePtGenerator()

inputData, maskedIndexTotal = InputDataGenerator(totalHolePts)

outputData, meanStdStress = OutputDataGenerator(totalHolePts)

ContourPlotter(inputData, maskedIndexTotal, outputData, meanStdStress)
#
inputData = np.array(inputData)
outputData = np.array(outputData)
meanStdStress = np.array(meanStdStress)
# maskedIndexTotal = copy.deepcopy(np.array(maskedIndexTotal))
# gridMeshtotal = np.array(gridMeshtotal)

print(inputData.shape, outputData.shape)


np.save('inputData.npy', inputData)
np.save('outputData.npy', outputData)
np.save('meanStdStress.npy', meanStdStress)

maskedIndexTotalFile = open('maskedIndexTotal.pkl', 'wb')

# gridMeshtotalFile = open('gridMeshtotal.pkl', 'wb')

pickle.dump(maskedIndexTotal, maskedIndexTotalFile)
# pickle.dump(gridMeshtotal, gridMeshtotalFile)

maskedIndexTotalFile.close()
# gridMeshtotalFile.close()



