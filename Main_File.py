# Importing necessary packages
import numpy as np
import pandas as pd
import seaborn as sb

#from Input_PreProcessor import People_PreProcessor
from Models import *
from Conf_Measure import *
from Feature_Generator import *
from sklearn.preprocessing import scale
from sklearn.cross_validation import train_test_split

# Reading the dataset 
Address_Stock_Return = './data/stock_returns_base150.csv'
Address_test_SVR = './data/Predictions_SVR.csv'
Address_test_KRR = './data/Predictions_KRR.csv'
Address_test_RF = './data/Predictions_RF.csv'

Stock_Base_Data = pd.read_csv(Address_Stock_Return)
Bool_Train_Test = ~Stock_Base_Data['S1'].isnull()
Stock_Base_X = Stock_Base_Data.iloc[:,2:]
Stock_Base_Y = Stock_Base_Data[Bool_Train_Test].S1

# Generating new set of features to capture the dynamics of Stocks
(Stock_Base_Train, Stock_Base_Test) = Feature_Generator(Stock_Base_X, Bool_Train_Test)
# Stock_Base_Train = Stock_Base_X[Bool_Train_Test]
# Stock_Base_Test = Stock_Base_X[~Bool_Train_Test]

Scaled_train_Data = scale(Stock_Base_Train)

Scaled_test_Data = scale(Stock_Base_Test)

# Building the models and Validate them using Leave-one-out CV
################ First Model: Support Vector Regression ###################
print('First Model: Support Vector Regression')
# Building the model
(MeanMSE_SVR, svr_Tuned) = First_Model_SVR(Scaled_train_Data, Stock_Base_Y)

# Predicting the test set using the built model
SVR_Results = SVR_Predictor(svr_Tuned, Scaled_test_Data, Address_test_SVR)

# Measuring the confidence of the model
(SVR_Tau, SVR_Tau_PValue, SVR_R2) = Conf_Measure(svr_Tuned, Scaled_train_Data, Stock_Base_Y, 'SVR')

##################Second Model: Kernel Ridge Regression####################
print('Second Model: Kernel Ridge Regression')
# Building the model
(MeanMSE_KRR, krr_Tuned) = Second_Model_KRR(Scaled_train_Data, Stock_Base_Y)

# Predicting the test set using the built model
KRR_Results = KRR_Predictor(krr_Tuned, Scaled_test_Data, Address_test_KRR)

# Measuring the confidence of the model
(KRR_Tau, KRR_Tau_PValue, KRR_R2) = Conf_Measure(krr_Tuned, Scaled_train_Data, Stock_Base_Y, 'KRR')

##################Third Model: Random Forest ########################
print('Third Model: Random Forest')
# Building the model
(MeanMSE_RF, RFModel) = RF_Model(Scaled_train_Data, Stock_Base_Y)

# Predicting the test set using the built model
(RF_Results, Feature_Importance) = RF_Predictor(RFModel, Scaled_test_Data, Address_test_RF)

# Measuring the confidence of the model
(RF_Tau, RF_Tau_PValue, RF_R2) = Conf_Measure(RFModel, Scaled_train_Data, Stock_Base_Y, 'RF')

# Printing out the importance of the features in the model. 
Sorted_indices = np.argsort(Feature_Importance)[::-1]
print('Features: Their importance')
for f in Sorted_indices:
    print("%s: %f" % (Stock_Base_Train.columns[f], Feature_Importance[f]))

# Plotting the correlation of the three important features (S7, S6 and S2) vs S1
g = sb.jointplot("S1", "S7", Stock_Base_Data[Bool_Train_Test], kind="reg", size = 5)
g = sb.jointplot("S1", "S6", Stock_Base_Data[Bool_Train_Test], kind="reg", size = 5)
g = sb.jointplot("S1", "S2", Stock_Base_Data[Bool_Train_Test], kind="reg", size = 5)
sb.plt.show()



