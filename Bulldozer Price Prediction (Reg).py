import random
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split,RandomizedSearchCV,GridSearchCV,cross_val_score
from sklearn.linear_model import LinearRegression,LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,f1_score,accuracy_score, mean_absolute_error,precision_score,confusion_matrix,ConfusionMatrixDisplay,roc_curve,RocCurveDisplay,mean_squared_log_error,r2_score
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler,StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
from joblib import load,dump
from tqdm import tqdm

## predicting the sale price of buildozers using machine learning

## in this notebook we are going to go trhough example machine learning project with a goal of predicting the sale price of bulldozers

## importing training and validation sets

# df = pd.read_csv('TrainAndValid.csv',low_memory=False)

## parsing Dates

## when we work with timeseries data,we want to Enrich the Time and data component as much as possible

## we can do that by teling pandas which of our columns have dates in it using 'parse_date' parameter

## import Data but again this time parse Dates

df = pd.read_csv('TrainAndValid.csv',low_memory=False,parse_dates=['saledate'])

# print(df.saledate[:100])

# print(df.isna().sum())

# fig,ax = plt.subplots()

# ax.scatter(df['saledate'][:1000],df['SalePrice'][:1000])

# plt.show()

## plotting the Sale Price 

# df.SalePrice.plot.hist()

# plt.show()


# print(df.head())


## to display all the columns without truncation use df.T

print(df.T)

## sort dataframe by saledate
## when working with time series data its best to sort by date

## sort datafram in dateorder


df.sort_values('saledate',inplace=True,ascending=True)


# print(df['saledate'].head())

## make copy of the original dataframe

## we make a copy of the original dataframe when we manipulate the original data

df_tmp = df.copy()

## add datetime parameters for "saledate" Column

# print(df_tmp.saledate.dtype)

# print(df_tmp[:2].saledate.dt.date)


# print(df_tmp[:2].saledate.dt.year)

# print(df_tmp[:2].saledate.dt.day)

df_tmp['saleYear'] = df_tmp.saledate.dt.year

df_tmp['saleMonth'] = df_tmp.saledate.dt.month

df_tmp['saleDay'] = df_tmp.saledate.dt.day

df_tmp['saleDayOfWeek'] = df_tmp.saledate.dt.dayofweek

df_tmp['saleDayOfYear'] = df_tmp.saledate.dt.dayofyear


# print(df_tmp.T)

## now we've enriched our dataframe with Datetime feautures we can remove 'saledate'

df_tmp.drop(columns='saledate',axis=1,inplace=True)

# print(df_tmp.head(20))


## check the different states in the dataframe
# print(df_tmp.state.value_counts())


## convert strings to categories

## one way we can turn all of our data into numbers is by converting them to pandas categories

## convert strings to categories

## we can check different data types compatible with pandas here
## https://pandas.pydata.org/pandas-docs/version/1.3/reference/general_utility_functions.html


pd.api.types.is_string_dtype(df_tmp['UsageBand'])

strings = []

## find the columns which contain string

for i in df_tmp.columns.values:
    if pd.api.types.is_string_dtype(df_tmp[i]) is True:
        strings.append(i)
        

# print(len(strings))

## if youre wondering what df.items does heres's an example

# for labels,values in df_tmp.items():
#     print(labels,'\n')
#     print(values)
    
## this will turn all of the string values into categories

for label,content in df_tmp.items():
    if pd.api.types.is_string_dtype(content):
        df_tmp[label] = content.astype('category').cat.as_ordered()
        

# print(df_tmp.info())

## check the list of categories
# print(df_tmp.state.cat.categories)

## check the numerical values of the categories
# print(df_tmp.state.cat.codes)

## thanks to pandas categories we all have the way to access our data in the form of numbers
## but we still have a bunch of missing data

## save preprocessed data

## export current tmp dataframe

df_tmp.to_csv('Bluebook_for_bulldozers.csv',index=False)


## reassing the df_tmp

df_tmp = pd.read_csv('Bluebook_for_bulldozers.csv')

## filling the missing values

## fill numeric missing values first

for label,content in df_tmp.items():
    if pd.api.types.is_numeric_dtype(content):
        print(label)


# print(df_tmp['ModelID'])


## check for which numeric columns have null values

for label,content in df_tmp.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            ## add a binary column which tells us if the data was missing

            df_tmp[label+'is Missing'] = pd.isnull(content)
            ## fill missing numeric values with median
            df_tmp[label] = content.fillna(content.median())
            
## check if there is any null numeric values

# for label,content in df_tmp.items():
#     if pd.api.types.is_numeric_dtype(content):
#         if pd.isnull(content).sum():
#             print(label)


## check how many examples are missing
# print(df_tmp.isnull().sum())         
            
            
# print(df_tmp.columns.values)



## filling and Turning categorical variables into numbers

## check for columns which aren't numeric

# for label,content in df_tmp.items():
#     if not pd.api.types.is_numeric_dtype(content):
        
#         print(label)

## turn categorical variables into numbers and fill missing

for label,content in df_tmp.items():
    if not pd.api.types.is_numeric_dtype(content):
        df_tmp[label+'is Missing'] = pd.isnull(content)
        df_tmp[label] = pd.Categorical(content).codes + 1
       
## checking the function of working for .items() method

# for label,content in df_tmp[['ModelID','saleDay']].items():
#     print(label)
#     print(" ____ ")
#     print(content)
#     print(pd.isnull(content))
    
#     print(pd.Categorical(content))
#     print(pd.Categorical(content).codes)
    

# print(df_tmp.isna().sum())


## now that all our data is numerice and no missing values,we should be able to build the machine learning model


# model = RandomForestRegressor(n_jobs=-1)


## training the model
# model.fit(df_tmp.drop(columns='SalePrice'),df_tmp['SalePrice'])

## fit the model
# print(model.score(df_tmp.drop(columns='SalePrice'),df_tmp['SalePrice']))


## why obove metric is not reliable(or why doesnt it hold water)

## because we evaluated the exact train data

## split the data into valid set after year 2012


df_valid = df_tmp[(df_tmp['saleYear'] == 2012)]

print(df_valid)

df_train = df_tmp[(df_tmp['saleYear'] != 2012)]

print(df_train)

## split data into x and y

x_train,y_train = df_train.drop(columns='SalePrice'),df_train['SalePrice']
x_valid,y_valid = df_valid.drop(columns='SalePrice'),df_valid['SalePrice']

print(x_train)
print(y_train)


## create evaluation function(the competition uses rmsqle)

def rmsle(y_test,y_preds):
    ## calculate root mean squared log error with y_true and y_preds
    return np.sqrt(mean_squared_log_error(y_test,y_preds))


## create function to evaluate model on few different levels

def show_scores_model(model):
    train_predict = model.predict(x_train)
    valid_predict = model.predict(x_valid)

    scores = {'Training MAE':mean_absolute_error(y_train,train_predict),
              'valid MAE':mean_absolute_error(y_valid,valid_predict),
              'Training RMSLE':rmsle(y_train,train_predict),
              'valid RMSLE':rmsle(y_valid,valid_predict),
              'Training_r2':r2_score(y_train,train_predict),
              'valid R2':r2_score(y_valid,valid_predict)}

    return scores
    

## testing our model on a subset(this is mostly to tune hyperparameters)


## change max_Samples values


model1 = RandomForestRegressor(n_jobs=-1,random_state=42,max_samples=10000)

## cutting down on max number of samples each estimator sees improves training time

model1.fit(x_train,y_train)


print(show_scores_model(model1))

## Hyperparameter tuning with RandomizedSearchCV


rf_grid = {'n_estimators':np.arange(10,100,10),
           'max_depth':[None,3,5,10],
           'min_samples_split':np.arange(2,20,2),
           'min_samples_leaf':np.arange(1,20,2),
           'max_features':[0.5,1,'sqrt'],
           'max_samples':[10000]
           }

## instantiate randomized search cv

rs_model = RandomizedSearchCV(RandomForestRegressor(n_jobs=-1,random_state=42),param_distributions=rf_grid,cv=5,n_iter=2,verbose=True)

rs_model.fit(x_train,y_train)

## we can find the best hyperparameters that the randomized search CV Found for us

print(rs_model.best_params_)

## evaluate the randomizedsearch model
print(show_scores_model(rs_model))



## train a model with best hyperparameters note these were found after 100 iterations of RandomizedsearchCV

ideal_model = RandomForestRegressor(n_estimators=40,min_samples_leaf=1,min_samples_split=14,max_features=0.5,n_jobs=-1,max_samples=None,random_state=42)


ideal_model.fit(x_train,y_train)

print(show_scores_model(ideal_model))

## importing the test data

df_test = pd.read_csv('Test.csv',low_memory=False,parse_dates=['saledate'])

print(df_test)

## preprocess the data and get the data into same format as training data

def preprocess(df):
    ## performs transformations and returns the transformed Dataframe

    df['saleYear'] = df.saledate.dt.year 
    df['saleMonth'] = df.saledate.dt.month
    df['saleDay'] = df.saledate.dt.day
    df['saleDayOfWeek'] = df.saledate.dt.dayofweek
    df['saleDayOfYear'] = df.saledate.dt.dayofyear
    
    
    df.drop(columns='saledate',inplace=True)
    
    
    ## fill the numeric rows with median
    for label,content in df.items():
        
        if pd.api.types.is_numeric_dtype(content):
            if pd.isnull(content).sum():
                
                ## add a binary column which tells us if the data was missing

                df[label+'is Missing'] = pd.isnull(content)
                ## fill missing numeric values with median
                df[label] = content.fillna(content.median())
                
           
    ## filled Categorical Missing data and turn categories into numbers
        if not pd.api.types.is_numeric_dtype(content):
           
            df[label+'is Missing'] = pd.isnull(content)
            ## we add +1 to categorie code because pandas enode missing categories as -1
            df[label] = pd.Categorical(content).codes+1
                

    return df


df_test = preprocess(df_test)

print(df_test.shape)

print(df_test.head(20))

print(df_test.isnull().sum())

print(df_test.isna().sum())


##manually adjust df test to have auctioneer id is missing column


df_test['auctioneerIDis Missing'] = False


print(set(x_train.columns) - set(df_test.columns))

print(list(x_train.columns))
print(df_test.columns.values)

## convert the feauture columns of df_test as same as x_Train as we are predicting the df_Test,both x_train and df_Test should have columns with same order



df_test = df_test.loc[:,['SalesID', 'MachineID', 'ModelID', 'datasource', 'auctioneerID', 'YearMade', 'MachineHoursCurrentMeter', 'UsageBand', 'fiModelDesc', 'fiBaseModel', 'fiSecondaryDesc', 'fiModelSeries', 'fiModelDescriptor', 'ProductSize', 'fiProductClassDesc', 'state', 'ProductGroup', 'ProductGroupDesc', 'Drive_System', 'Enclosure', 'Forks', 'Pad_Type', 'Ride_Control', 'Stick', 'Transmission', 'Turbocharged', 'Blade_Extension', 'Blade_Width', 'Enclosure_Type', 'Engine_Horsepower', 'Hydraulics', 'Pushblock', 'Ripper', 'Scarifier', 'Tip_Control', 'Tire_Size', 'Coupler', 'Coupler_System', 'Grouser_Tracks', 'Hydraulics_Flow', 'Track_Type', 'Undercarriage_Pad_Width', 'Stick_Length', 'Thumb', 'Pattern_Changer', 'Grouser_Type', 'Backhoe_Mounting', 'Blade_Type', 'Travel_Controls', 'Differential_Type', 'Steering_Controls', 'saleYear', 'saleMonth', 'saleDay', 'saleDayOfWeek', 'saleDayOfYear', 'auctioneerIDis Missing', 'MachineHoursCurrentMeteris Missing', 'UsageBandis Missing', 'fiModelDescis Missing', 'fiBaseModelis Missing', 'fiSecondaryDescis Missing', 'fiModelSeriesis Missing', 'fiModelDescriptoris Missing', 'ProductSizeis Missing', 'fiProductClassDescis Missing', 'stateis Missing', 'ProductGroupis Missing', 'ProductGroupDescis Missing', 'Drive_Systemis Missing', 'Enclosureis Missing', 'Forksis Missing', 'Pad_Typeis Missing', 'Ride_Controlis Missing', 'Stickis Missing', 'Transmissionis Missing', 'Turbochargedis Missing', 'Blade_Extensionis Missing', 'Blade_Widthis Missing', 'Enclosure_Typeis Missing', 'Engine_Horsepoweris Missing', 'Hydraulicsis Missing', 'Pushblockis Missing', 'Ripperis Missing', 'Scarifieris Missing', 'Tip_Controlis Missing', 'Tire_Sizeis Missing', 'Coupleris Missing', 'Coupler_Systemis Missing', 'Grouser_Tracksis Missing', 'Hydraulics_Flowis Missing', 'Track_Typeis Missing', 'Undercarriage_Pad_Widthis Missing', 'Stick_Lengthis Missing', 'Thumbis Missing', 'Pattern_Changeris Missing', 'Grouser_Typeis Missing', 'Backhoe_Mountingis Missing', 'Blade_Typeis Missing', 'Travel_Controlsis Missing', 'Differential_Typeis Missing', 'Steering_Controlsis Missing']]


# Finally our test dataframe has same features as train data frame

# make prediction on test data


test_preds = ideal_model.predict(df_test)

print(test_preds)

## we made some predictions but they are not in the format the kaggle wants it to be
## create a dataframe with two columns 1.ModelID and 2.Saleprice

df_preds = pd.DataFrame()

df_preds['SalesID'] = df_test['SalesID']

df_preds['SalePrice'] = test_preds


print(df_preds.head(20))


## xport the result to the csv

df_preds.to_csv('Predictions.csv',index=False)


## find the feauture importances of our best model


print(ideal_model.feature_importances_)


plt.bar(df_test.columns.values,ideal_model.feature_importances_,align='center')

plt.show()


## helper function for plotting important feautures 


def plot_feautures(columns,importances,n=20):
    df = pd.DataFrame({'features':columns,'feature_importances':importances}).sort_values('feature_importances',ascending=False).reset_index(drop=True)

    ## plot the dataframe

    fig,ax = plt.subplots()
    ax.barh(df['features'][:n],df['feature_importances'][:n])
    ax.set_xlabel('Feauture Importances')
    ax.set_ylabel('Features')

    plt.show()

plot_feautures(x_train.columns,ideal_model.feature_importances_)

## final challenge:
## what other machine learning models could you try on our dataset?
## hint:https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html
## check out this map or try to use the advanced boosting algorithms like catboost or xgboost









    











        
        
        