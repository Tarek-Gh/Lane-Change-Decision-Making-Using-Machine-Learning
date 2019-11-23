import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, metrics
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.np_utils import to_categorical

columns_list = ['egoX', 'egoY', 'egoV','egoA',
                'leftFollowerDistanceToEgo', 'leftFollowerY', 'leftFollowerV','leftFollowerA',
                'leftLeaderDistanceToEgo', 'leftLeaderY', 'leftLeaderV','leftLeaderA',
                'rightFollowerDistanceToEgo', 'rightFollowerY', 'rightFollowerV', 'rightFollowerA',
                'rightLeaderDistanceToEgo', 'rightLeaderY', 'rightLeaderV', 'rightLeaderA',
                'LeaderDistanceToEgo', 'LeaderY', 'LeaderV','LeaderA',
                'FollowerDistanceToEgo', 'FollowerY', 'FollowerV', 'FollowerA']

# Read data
df = pd.read_csv("data.csv", index_col=False)

# apply PCA to training data
X = df[columns_list]
sc = StandardScaler()
X = sc.fit_transform(X)
pca = PCA(n_components=11)
X = pca.fit_transform(X)

# Convert string labels into numbers.
le = preprocessing.LabelEncoder()
Y = le.fit_transform(df['action'])

# Split data : 70% training and 30% testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
Y_train = to_categorical(Y_train, num_classes=3)

# initialize ANN model
model = Sequential()
model.add(Dense(20, input_shape=(11,), activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(3, activation='softmax')) # 3 output classes: stay in lane / change lanes to left / right
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# train model
model.fit(X_train, Y_train, epochs=300, batch_size=10)
model.save('ANN_model.h5')

# test model
Y_pred = model.predict_classes(X_test)

# Find accuracy
acc = metrics.accuracy_score(Y_test, Y_pred)    # Find accuracy
print("Accuracy : {}".format(acc))


model2 = Sequential()
model2.add(Dense(200, input_shape=(11,), activation='relu'))
model2.add(Dense(200, activation='relu'))
model2.add(Dense(200, activation='relu'))
model2.add(Dense(200, activation='relu'))
model2.add(Dense(200, activation='relu'))
model2.add(Dense(3, activation='softmax')) # 3 output classes: stay in lane / change lanes to left / right
model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model2.fit(X_train, Y_train, epochs=300, batch_size=10)
model2.save('ANN_model.h5')

# test model
Y_pred = model2.predict_classes(X_test)

# Find accuracy
acc2 = metrics.accuracy_score(Y_test, Y_pred)    # Find accuracy
print("Accuracy : {}".format(acc))


model21 = Sequential()
model21.add(Dense(20, input_shape=(11,), activation='relu'))
model21.add(Dense(20, activation='relu'))
model21.add(Dense(20, activation='relu'))
model21.add(Dense(20, activation='relu'))
model21.add(Dense(20, activation='relu'))
model21.add(Dense(20, activation='relu'))
model21.add(Dense(20, activation='relu'))
model21.add(Dense(20, activation='relu'))
model21.add(Dense(20, activation='relu'))
model21.add(Dense(20, activation='relu'))
model21.add(Dense(20, activation='relu'))
model21.add(Dense(20, activation='relu'))
model21.add(Dense(3, activation='softmax')) # 3 output classes: stay in lane / change lanes to left / right
model21.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model21.fit(X_train, Y_train, epochs=300, batch_size=10)
model21.save('ANN_model.h5')

# test model
Y_pred = model21.predict_classes(X_test)

# Find accuracy
acc21 = metrics.accuracy_score(Y_test, Y_pred)    # Find accuracy
print("Accuracy : {}".format(acc))

accuracy = [acc, acc2, acc21]
print(accuracy)