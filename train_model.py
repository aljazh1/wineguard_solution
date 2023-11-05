import os
import pickle
import cv2
import numpy as np
import csv
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier


def get_data():
    rootdir = './dataset'
    X = []
    y = []

    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            img_path = os.path.join(subdir, file)
            image = cv2.imread(img_path)
            pixel_data = image_to_feature_vector(image)
            X.append(pixel_data)
            y.append(subdir[subdir.rfind("\\") + 1:])

    X = np.array(X)
    y = np.array(y)
    df = pd.DataFrame(X)
    df['label'] = y
    df.to_csv('img_data.csv', index=False, encoding='utf-8')


def image_to_feature_vector(image, size=(128, 128)):
    return cv2.resize(image, size).flatten()


# def extract_features(image_path):
#     fixed_width = 128
#     fixed_height = 128
#     image = cv2.imread(image_path)
#     resized_image = cv2.resize(image, (fixed_width, fixed_height))
#     feature_vector = resized_image.reshape(-1, 3)
#     return feature_vector


# def train_decision_tree():
#     df = pd.read_csv('img_data.csv')
#     y = df['label']
#     X = df.drop('label', axis=1)
#     (trainRI, testRI, trainRL, testRL) = train_test_split(
#         X, y, test_size=0.25, random_state=1234)
#     decision_tree = DecisionTreeClassifier()
#     decision_tree.fit(trainRI, trainRL)
#     acc = decision_tree.score(testRI, testRL)
#     pickle.dump(decision_tree, open('decision_tree.pkl', 'wb'))
#     print("Decision tree accuracy: {:.2f}%".format(acc * 100))

#
# def train_neural_network():
#     df = pd.read_csv('img_data.csv')
#     y = df['label']
#     X = df.drop('label', axis=1)
#     (trainRI, testRI, trainRL, testRL) = train_test_split(
#         X, y, test_size=0.25, random_state=1234)
#     model = Sequential()
#     model.add(Dense(64, input_dim=X.shape[1], activation='relu'))
#     model.add(Dense(32, activation='relu'))
#     model.add(Dense(1, activation='sigmoid'))
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#     model.fit(trainRI, trainRL, epochs=10, batch_size=32)
#     acc = model.evaluate(testRI, testRL)
#     model.save('neural_network.h5')
#     print("Neural network accuracy: {:.2f}%".format(acc * 100))

def train_random_forest():
    df = pd.read_csv('img_data.csv')
    y = df['label']
    X = df.drop('label', axis=1)
    (trainRI, testRI, trainRL, testRL) = train_test_split(
        X, y, test_size=0.25, random_state=1234)
    model = RandomForestClassifier(n_estimators=100, random_state=1234)
    model.fit(trainRI, trainRL)
    acc = model.score(testRI, testRL)
    pickle.dump(model, open('random_forest.pkl', 'wb'))
    print("Random forest accuracy: {:.2f}%".format(acc * 100))


def train():
    df = pd.read_csv('img_data.csv')
    y = df['label']
    X = df.drop('label', axis=1)
    (trainRI, testRI, trainRL, testRL) = train_test_split(
        X, y, test_size=0.25, random_state=1234)

    # param_grid = {
    #     'n_neighbors': [3, 5, 7, 9],
    #     'weights': ['uniform', 'distance'],
    #     'p': [1, 2]
    # }

    model = KNeighborsClassifier(n_neighbors=3, p=1, weights='distance')

    # grid_search = GridSearchCV(model, param_grid, cv=5)
    # grid_search.fit(trainRI, trainRL)
    # best_model = grid_search.best_estimator_
    # acc = best_model.score(testRI, testRL)
    # pickle.dump(best_model, open('best_model.pkl', 'wb'))
    # print("Best hyperparameters:", grid_search.best_params_)
    # print("Best accuracy: {:.2f}%".format(acc * 100))

    model.fit(trainRI, trainRL)
    acc = model.score(testRI, testRL)
    pickle.dump(model, open('model.pkl', 'wb'))
    print("Accuracy: {:.2f}%".format(acc * 100))


if __name__ == '__main__':
    get_data()
    train_random_forest()
    # train()
    # train_dt()
    # train_nn()
