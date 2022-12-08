import pandas as pd
import  json
from sklearn.metrics import accuracy_score


def get_labels(labels, num_labels=3):
    check_labels = range(num_labels)
    
    labels_unique = pd.unique(labels)
    labels_to_cl = dict(zip(labels_unique, check_labels))
    
    labels_cl = list(map(labels_to_cl.get, labels))
    
    return labels_cl

y_true = pd.read_csv('iris.data', header=None).pop(4)
slov = {
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
    'Iris-virginica': 2
}
y_true = y_true.map(slov)

y_pred = pd.read_csv('predict.txt', header=None)[0]
y_pred = get_labels(y_pred)

acc = accuracy_score(y_true, y_pred)

with  open('metrics.json', 'w') as f:
    json.dump(
        {"accuracy": acc}, 
        f
    )
