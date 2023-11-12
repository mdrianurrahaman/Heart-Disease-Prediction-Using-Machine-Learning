import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


###### data collection and processing 

heart_data = pd.read_csv(r'C:\Users\ramiu\Desktop\Machine Learning Projects\Heart Disease Prediction\heart_disease_data.csv')

###### print first five rows of the dataset 

print(heart_data.head())


####### print last 5 rows ########

print(heart_data.tail())


###getting info of the data ########### 

print(heart_data.info())

### checking missing values ####### 

print(heart_data.isnull().sum())

##### statistical measure about the data ##

print(heart_data.describe())

######## checking the distribution of the target variable ######## 

print(heart_data['target'].value_counts())

######## 1 means disease heart and 0 means no disease in heart ### 

x = heart_data.drop(columns='target',axis=1)
y = heart_data['target']

print(x)
print(y)

##### splitting the data into training data & test data 

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,stratify=y,random_state=2)

print(x.shape,x_train.shape,x_test.shape)


######## logistic regression model training #######

model = LogisticRegression()

## training the logistic regression model #### 

model.fit(x_train,y_train)

######## accuracy on training data ##### 

x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction,y_train)

print('Accuracy on training data:', training_data_accuracy)


######## accuracy on test data ###### 

x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction,y_test)

print('Accuracy on Test Data :',test_data_accuracy)

# building a predictive system ####

#input_data = (56,0,1,140,294,0,0,153,0,1.3,1,0,2)
input_data = (59,1,0,170,326,0,0,140,1,3.4,0,0,3)

### change the input data to a numpy array

input_data_as_numpy_array = np.asarray(input_data)

###### reshape the numpy array as we are predictinh for only on instance ####### 

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)

print(prediction)

if (prediction[0]==0):
    print('The person does not heart disease ')
else:
    print('The person has heart disease')
