import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

sonar_data = pd.read_csv('/workspaces/Sonar-System-Rock-vs-mine-/sonar data.csv', header = None) #reading the csv file 
X = sonar_data.drop(columns = 60,axis=1) #seperating the continous feature values and the the labels
Y = sonar_data[60] #labels are extracted
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=1) #splitting the given examples into training(80%) and test data set
model =LogisticRegression() #preparing the supervised logistic regression model for this classification problem
model.fit(X_train,Y_train) #training the model using the training set
X_train_predictions = model.predict(X_train) #getting the predictions for the training examples
training_data_accuracy = accuracy_score(X_train_predictions,Y_train) #calculating the accuracy for training data
print(f"Accuracy on training data : {training_data_accuracy}") 
X_test_predictions = model.predict(X_test) #evaluating the model for test data set
test_data_accuracy = accuracy_score(X_test_predictions,Y_test) #calculating the test data set error
print(f"Accuracy on test data : {test_data_accuracy}") 
input = (0.0346,0.0509,0.0079,0.0243,0.0432,0.0735,0.0938,0.1134,0.1228,0.1508,0.1809,0.2390,0.2947,0.2866,0.4010,0.5325,0.5486,0.5823,0.6041,0.6749,0.7084,0.7890,0.9284,0.9781,0.9738,1.0000,0.9702,0.9956,0.8235,0.6020,0.5342,0.4867,0.3526,0.1566,0.0946,0.1613,0.2824,0.3390,0.3019,0.2945,0.2978,0.2676,0.2055,0.2069,0.1625,0.1216,0.1013,0.0744,0.0386,0.0050,0.0146,0.0040,0.0122,0.0107,0.0112,0.0102,0.0052,0.0024,0.0079,0.0031) #new examples
X_input = np.asarray(input) #converting the input into an array
X_input = X_input.reshape(1,-1) #reshaping the array
prediction = model.predict(X_input) #making the prediction
if prediction[0] == 'R':
    print("It is a rock")
else:
    print("It is a mine")