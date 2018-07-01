'''
Linear Regressor predictor for FIFA 2018 results.
Author: gudgud96 
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score

# Prepare data - drop some columns manually from results_processed.csv
training_data = pd.read_csv('results_processed_dev.csv').replace(np.NaN, 200)
X = training_data.drop(['home_team', 'away_team', 'home_wins'], axis=1)
Y = training_data['home_wins'].astype('int')

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

encoder = LabelBinarizer()
encoder.fit(Y_train)
Y_train_one_hot = encoder.transform(Y_train)
encoder.fit(Y_test)
Y_test_one_hot = encoder.transform(Y_test)

testing_data = pd.read_csv('results_test_dev.csv').replace(np.NaN, 200)
X_train = training_data.drop(['home_team', 'away_team', 'home_wins'], axis=1)
X_test = testing_data.drop(['home_team', 'away_team', 'home_score', 'away_score','home_wins'], axis=1)
Y_train = training_data['home_wins'].astype('int')
Y_test = testing_data['home_wins'].astype('int')

encoder = LabelBinarizer()
encoder.fit(Y_train)
Y_train_one_hot = encoder.transform(Y_train)
encoder.fit(Y_test)
Y_test_one_hot = encoder.transform(Y_test)

# Build model - a simple linear regressor is used
model = LinearRegression()
model = model.fit(X_train, Y_train_one_hot)
Y_predict = model.predict(X_test)
Y_predict_one_hot = (Y_predict == Y_predict.max(axis=1)[:,None]).astype(int)
print("Accuracy is: " + str(accuracy_score(Y_test_one_hot, Y_predict_one_hot)))
testing_data['predicted_outcome'] = encoder.inverse_transform(Y_predict_one_hot)

# Output testing results to csv file
predicted_winner = []
actual_winner = []
for i in range(len(testing_data)):
    if testing_data['predicted_outcome'][i] == 0:
        predicted_winner.append(testing_data['away_team'][i])
    elif testing_data['predicted_outcome'][i] == 1:
        predicted_winner.append(testing_data['home_team'][i])
    else:
        predicted_winner.append('DRAW')

    if testing_data['home_wins'][i] == 0:
        actual_winner.append(testing_data['away_team'][i])
    elif testing_data['home_wins'][i] == 1:
        actual_winner.append(testing_data['home_team'][i])
    else:
        actual_winner.append('DRAW')

testing_data['predicted_winner'] = predicted_winner
testing_data['actual_winner'] = actual_winner
testing_data['home_winning_proba'] = Y_predict[:, 1]
testing_data['away_winning_proba'] = Y_predict[:, 0]
testing_data['draw_proba'] = Y_predict[:, 2]

testing_data.to_csv('results_test_output.csv')

# Predict match results
predict_data = pd.read_csv('results_predict.csv').replace(np.NaN, 200)
X_prediction = predict_data.drop(['home_team', 'away_team'], axis=1)
print(X_prediction.shape)
Y_prediction = model.predict(X_prediction)
Y_prediction_one_hot = (Y_prediction == Y_prediction.max(axis=1)[:,None]).astype(int)
predict_data['predicted_outcome'] = encoder.inverse_transform(Y_prediction_one_hot)

predicted_winner = []
for i in range(len(predict_data)):
    if predict_data['predicted_outcome'][i] == 0:
        predicted_winner.append(predict_data['away_team'][i])
    elif predict_data['predicted_outcome'][i] == 1:
        predicted_winner.append(predict_data['home_team'][i])
    else:
        predicted_winner.append('DRAW')

predict_data['predicted_winner'] = predicted_winner
predict_data['home_winning_proba'] = Y_prediction[:, 1]
predict_data['away_winning_proba'] = Y_prediction[:, 0]
predict_data['draw_proba'] = Y_prediction[:, 2]

predict_data.to_csv('results_predict_output_final.csv')