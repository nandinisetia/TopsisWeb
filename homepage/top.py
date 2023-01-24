import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
import sys

def checkDataFrame(df):
    df = df.iloc[:,1:]
    df = df.to_numpy()

    if(not is_numeric_dtype(df)):
        print('All values in dataset from 2nd column to last column must be numeric')
        return False
    
    return True

def checkString(string, size, category):
    string = pd.Series(string.split(','))
    if(len(string) != size):
        print(f'Error, {category}s not in proper format')
        return False
    
    if(category == 'weight'):
        
        if(not string.apply(str.isnumeric).all()):
            
            print('Weights must all be numeric')
            return False
        
    if(category == 'impact'):
        for x in string:
            if x not in ['+', '-']:
                
                print('impacts must be either + or -')
                return False
    return True

def checkArguments(): 
    if(len(sys.argv) < 2):
        print('Please give one input file.')
        return False
    
    if(len(sys.argv) < 5):
        print('Please give 4 parameters in the following form:\n(input_file, weights, impacts, result_file_name')
        return False
    
    if(len(sys.argv) >5):
        print('Ignoring extra parameters')

    return True
    
def checkInputFile():
    inputDF = None
    try:
        inputDF = pd.read_csv(sys.argv[1])
        
    except FileNotFoundError:
        print(f'Error, file \'{sys.argv[1]}\' does not exist.')
        return False
    except:
        print(' Error, file format not correct.')
        return False
    
    # Check Format
    if(inputDF.shape[1] < 3):
        print('Atleast 3 columns required in input file.')
        return False
    
    return True

def Topsis(dataset, weights, impacts):

    #Reading Dataset
    data = pd.read_csv(dataset)
    df = data.copy()
    data = data.iloc[:,1:] #Dropping 1st Column

    data_matrix = data.to_numpy() #Converting into Matrix

    #Vector Normalisation
    numRows = len(data_matrix)
    numCols = len(data_matrix[0])

    rSumSq = [] #Calculating Root of Sum of Squares
    for j in range(numCols):
        sum = 0
        for i in range(numRows):
            sum = sum + (data_matrix[i][j] ** 2)
        res = sum ** 0.5
        rSumSq.append(res)

    for i in range(numRows): #Dividing each entry of data_matrix by rSumSq and updating it
        for j in range(numCols):
            data_matrix[i][j] = float(data_matrix[i][j]) / float(rSumSq[j])


    #Weight Assignmnet
    for j in range(numCols): #Multiplying data_matrix with the weights of the corresponding column
        for i in range(numRows):
            data_matrix[i][j] = data_matrix[i][j] * float(weights[j])

    #Calulating ideal best and worst for each column
    maxx = np.amax(data_matrix, axis=0) #Maximum value of each column
    minn = np.amin(data_matrix, axis=0) #Minimum value of each column
    ibest = [] #ideal best
    iworst = [] #ideal worst

    for i in range(numCols):
        if impacts[i] == '+':
            ibest.append(maxx[i])
            iworst.append(minn[i])
        elif impacts[i] == '-':
            ibest.append(minn[i])
            iworst.append(maxx[i])

    #Calculating Eucledian Distance from ideal best
    ibest_dist = []
    for i in range(numRows):
        sum1 = np.sum(np.square(data_matrix[i] - ibest))
        dist1 = np.sqrt(sum1)
        ibest_dist.append(dist1)

    #Calculating Eucledian Distance from ideal worst
    iworst_dist = []
    for i in range(numRows):
        sum2 = np.sum(np.square(data_matrix[i] - iworst))
        dist2 = np.sqrt(sum2)
        iworst_dist.append(dist2)

    #Calculating Performance Score
    score = []
    for i in range(numRows):
        p = iworst_dist[i] / (iworst_dist[i] + ibest_dist[i])
        score.append(round(p,5))

    df['Topsis Score'] = score

    #Final Result based on Performance Score
    df['Rank'] = (df['Topsis Score'].rank(method='max', ascending=False))
    df = df.astype({"Rank": int})
    
    return df
    
def outputFile(df):
    df.to_csv(sys.argv[4], index=False, header=True)

if (__name__ == "__main__"):
    if(not checkArguments()): 
       sys.exit(0)
    if(not checkInputFile()):
        sys.exit(0)
    
    inputDF = pd.read_csv(sys.argv[1])
    if(not checkDataFrame(inputDF)):
        sys.exit(0)

    if(not (checkString(sys.argv[2], inputDF.shape[1] - 1, 'weight'))):
        sys.exit(0)
    if(not (checkString(sys.argv[3], inputDF.shape[1] - 1, 'impact'))):
        sys.exit(0)
    outputDF = Topsis(sys.argv[1], pd.Series(sys.argv[2].split(',')).astype(int), pd.Series(sys.argv[3].split(',')))
    
    outputFile(outputDF)