import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, metrics
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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
print(X)

# Convert string labels into numbers.
le = preprocessing.LabelEncoder()
Y = le.fit_transform(df['action'])
print(Y)

# Split data : 70% training and 30% testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

# Classifier
acc_list = []
for i in range(1, 52, 2):               # odd numbers from 1 to 51
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, Y_train)           # Train the model using the training sets
    Y_pred = knn.predict(X_test)        # Predict the response for test dataset
    
    acc = metrics.accuracy_score(Y_test, Y_pred)    # Find accuracy
    acc_list.append(acc)
    print("Accuracy {} : {}".format(i, acc))

print('\n\n\n')
print(acc_list)