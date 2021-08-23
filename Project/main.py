from skmultiflow.data import DataStream
from skmultiflow.lazy import KNNClassifier
from skmultiflow.meta import OzaBaggingClassifier, OnlineBoostingClassifier

import pandas as pd

column_names=["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "class"]
train_data = pd.read_csv("adult_train.csv", names=column_names)
test_data = pd.read_csv("adult_test.csv", names=column_names)
test_data = test_data.drop(index=0)

for column_name in column_names:
    test_data = test_data.drop(test_data[test_data[column_name] == ' ?'].index)
    train_data = train_data.drop(train_data[train_data[column_name] == ' ?'].index)

train_size = train_data.shape[0]

data = pd.get_dummies(pd.concat([train_data, test_data]))

train_data = data[: train_size]
test_data = data[train_size: ]

train_data_stream = DataStream(train_data)
test_data_stream = DataStream(test_data)


clf = OzaBaggingClassifier(base_estimator=KNNClassifier(n_neighbors=8, max_window_size=2000, leaf_size=30), n_estimators=2)

X, y = train_data_stream.next_sample(13000)
clf = clf.partial_fit(X, y, classes=train_data_stream.target_values)

sample_count = 0
corrects = 0
window_size = 200
chart = []
while test_data_stream.has_more_samples():
    X, y = test_data_stream.next_sample(window_size)
    pred = clf.predict(X)
    if pred is not None:
        for j in range(len(pred)):
            if y[j] == pred[j]:
                corrects += 1
    sample_count += len(y)
    chart.append((sample_count, corrects / sample_count))
    clf = clf.partial_fit(X, y)


