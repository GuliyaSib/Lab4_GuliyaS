import pandas as pd
import  json
from sklearn.metrics import accuracy_score

y_true = pd.read_csv('iris.data', header=None).pop(4)
y_pred = pd.read_csv('predict.txt', header=None)[0]

slov = {
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
    'Iris-virginica': 2
}

y_true = y_true.map(slov)

acc = accuracy_score(y_true, y_pred)

with  open('metrics.json', 'w') as f:
    json.dump(
        {"accuracy": acc}, 
        f
    )