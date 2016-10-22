import numpy as np
import pandas as pd
import datetime

from datetime import datetime
    
def Feature_Generator(Orig_Data, Bool_Train_Test, NumLag = 2, AvgWin = 3):
    # Adding the 1-day and 2-day lags and the moving average of the last three days to the set of features
    
    Columns_Name = Orig_Data.columns
    for column in Columns_Name:
        for lag in range(NumLag):
            newcolumn_shift = column + '_Shifted_' + str(lag+1) + '-day'
            Orig_Data[newcolumn_shift] = Orig_Data[column].shift(lag+1)

    for column in Columns_Name:
        newcolumn_MovAvg = column + '_' + str(AvgWin) + 'day_MovAvg'
        Orig_Data[newcolumn_MovAvg] = Orig_Data[column].rolling(AvgWin).mean()
        
    # Replacing missing values with the next available value
    New_Dataset = Orig_Data.fillna(method = 'bfill')

    # Seperating the dataset to training and testing set
    New_Dataset_Train = New_Dataset[Bool_Train_Test]
    New_Dataset_Test = New_Dataset[~Bool_Train_Test]
    
    return(New_Dataset_Train, New_Dataset_Test)
