from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
import sys
import os
from django.conf import settings
from django.core.mail  import EmailMessage


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
    
def checkInputFile(uploaded_file_url):
    inputDF = None
    try:
        inputDF = pd.read_csv(uploaded_file_url)
        
    except FileNotFoundError:
        print(f'Error, file \'{uploaded_file_url}\' does not exist.')
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
    df.to_csv("./media/result.csv", index=False, header=True)
    

# Create your views here.
def index(request):
    if request.method == 'POST':
        dataset=request.FILES['id_docfile']
        fs = FileSystemStorage()
        filename = fs.save(dataset.name,dataset)
        uploaded_file_url = fs.url(filename)
        uploaded_file_url='.'+uploaded_file_url
        print(filename)
        print(uploaded_file_url)
        email = request.POST['email']
        weights = request.POST['Weights']
        impacts = request.POST['Impacts']
        

        # TANISHQ CODE

        # if(not checkArguments()): 
        #   print("ERROR")
        if(not checkInputFile(uploaded_file_url)):
          print("ERROR")
        inputDF = pd.read_csv(uploaded_file_url)
        if(not checkDataFrame(inputDF)):
          print("ERROR")
        if(not (checkString(weights, inputDF.shape[1] - 1, 'weight'))):
             print("ERROR")
        if(not (checkString(impacts, inputDF.shape[1] - 1, 'impact'))):
             print("ERROR")
        outputDF = Topsis(uploaded_file_url, pd.Series(weights.split(',')).astype(int), pd.Series(impacts.split(',')))
        outputFile(outputDF)

        subject = 'TOPSIS FEEDBACK'
        message = 'Result file for your upload on TOPSIS:'
        email_from = settings.EMAIL_HOST_USER
        recipient_list =[email]
        email = EmailMessage(subject, message, email_from, recipient_list)
        email.attach_file('./media/result.csv')
        email.send()
        print("MAIL SENT")
        os.remove('./media/result.csv')
        os.remove(uploaded_file_url)

        return render(request,'index.html')
    return render(request,'index.html')

