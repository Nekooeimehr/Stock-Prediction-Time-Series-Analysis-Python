# This function will give an idea of how confident we are about our model
import scipy.stats as stats
from sklearn.metrics import r2_score
def Conf_Measure(RegModel, Train_Data, True_Labels, ModelType):
    Predictions = RegModel.predict(Train_Data)        
    tau, p_value = stats.kendalltau(True_Labels, Predictions)
    R2_Measure = r2_score(True_Labels, Predictions)
    print('The Kindell Coefficient of ', ModelType, ' model is ', tau,' with a p-value of ',p_value)
    print('The R Square of ', ModelType, ' model is ', R2_Measure)
    print('')
    return(tau, p_value, R2_Measure)
