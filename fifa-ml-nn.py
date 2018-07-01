'''
Neural network predictor for FIFA 2018 results.
Author: gudgud96
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os.path
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import plot_model, to_categorical
from keras.callbacks import History

def plot_epoch_accuracy_graph(history):
    plt.plot(history.history['acc'], color='red')
    plt.plot(history.history['loss'], color='green')
    plt.ylabel('acc/loss')
    plt.xlabel('epochs')
    legend = plt.legend(loc='upper center', shadow=True)
    plt.show()

# Prepare data - number of examples is the first dimension for Keras
training_data = pd.read_csv('results_processed_dev.csv').replace(np.NaN, 200)
testing_data = pd.read_csv('results_test_dev.csv').replace(np.NaN, 200)
X_train = training_data.drop(['home_team', 'away_team', 'home_wins'], axis=1)
X_test = testing_data.drop(['home_team', 'away_team', 'home_score', 'away_score','home_wins'], axis=1)
Y_train = training_data['home_wins'].astype('int')
Y_test = testing_data['home_wins'].astype('int')

Y_train_one_hot = np.squeeze(to_categorical(Y_train, num_classes=3))
Y_test_one_hot = np.squeeze(to_categorical(Y_test, num_classes=3))

# train or load model
if os.path.isfile('fifa-2-layer-nn-model.h5'):
    print("Loading pre-trained model...")
    model = load_model('fifa-2-layer-nn-model.h5')

else:
    model = Sequential()
    model.add(Dense(5, input_shape=(12,)))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dense(300))
    model.add(Activation('relu'))
    model.add(Dense(500))
    model.add(Activation('relu'))
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation="softmax"))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    plot_model(model, to_file='model.png')
    print(model.summary())

    data_augmented = False
    epochs = 30
    batch_size = 256
    history = model.fit(X_train, Y_train_one_hot,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(X_test, Y_test_one_hot),
                    shuffle=True)

    model.save('fifa-2-layer-nn-model.h5') 
    plot_epoch_accuracy_graph(history)

# Score trained model
scores = model.evaluate(X_train, Y_train_one_hot, verbose=True)
print('Train loss:', scores[0])
print('Train accuracy:', scores[1])
scores = model.evaluate(X_test, Y_test_one_hot, verbose=True)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

# Predict match results
predict_data = pd.read_csv('results_predict_nn.csv').replace(np.NaN, 200)
X_prediction = predict_data.drop(['home_team', 'away_team'], axis=1)
Y_prediction = model.predict(X_prediction)
Y_prediction_one_hot = (Y_prediction == Y_prediction.max(axis=1)[:,None]).astype(int)
predict_data['predicted_outcome'] = np.argmax(Y_prediction_one_hot, axis=1)

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

predict_data.to_csv('results_predict_output_nn_final.csv')