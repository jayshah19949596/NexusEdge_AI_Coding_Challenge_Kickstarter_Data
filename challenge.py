# Usage:
# import challenge
# from challenge import Data, MyKNN
# data = Data("data.csv")
# model = MyKNN(data.X, data.y)
# model.train()
# model.predict()
# model.display_acc()
# model.display_confusion_matrix()

import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


class MyKNN(object):
    def __init__(self, data_x, data_y, n=3):
        self.model = KNeighborsClassifier(n_neighbors=n)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data_x, data_y, test_size=0.33)

    def train(self):
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        self.y_pred = self.model.predict(self.X_test)

    def display_acc(self):
        print('Accuracy of K Nearest Neighbor {:.5f}'.format(self.model.score(self.X_test, self.y_test)))
        print("")

    def display_confusion_matrix(self):
        conf_matrix = confusion_matrix(self.y_test, self.y_pred)
        print("Confusion Matrix for K Nearest Neighbor  is as follows :")
        print(conf_matrix)
        print("")


class Data(object):
    def __init__(self, file_path):
        self.data = read_as_pandas(file_path)
        self.pre_process_data()
        self.y = self.processed_data['success_or_not']
        self.X = self.processed_data[['Number of days', 'backers', 'usd_pledged_real', 'usd_goal_real']]

    def pre_process_data(self):
        self.processed_data = pd.concat([self.data[self.data['state'] == 'successful'],
                                         self.data[self.data['state'] == 'failed']])
        self.processed_data['success_or_not'] = self.processed_data.apply(func=success, axis=1)
        for i in ['launched', 'deadline']: self.processed_data[i] = pd.to_datetime(self.processed_data[i])
        self.processed_data['Duration'] = self.processed_data['deadline'] - self.processed_data['launched']
        self.processed_data['Number of days'] = self.processed_data['Duration'].dt.days
        self.processed_data.drop(labels=['usd pledged', 'name'], inplace=True, axis=1)


def success(df):
    if df['state'] == 'successful':
        return 1
    else:
        return 0


def read_as_pandas(file_path):
    return pd.read_csv(file_path)


def main():
    data_file_path = "data.csv"
    data = Data(data_file_path)
    model = MyKNN(data.X, data.y)
    model.train()
    model.predict()
    model.display_acc()
    model.display_confusion_matrix()
