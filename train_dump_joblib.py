import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import  RandomForestClassifier

#load dataset
iris = load_iris()
X, y = iris.data, iris.target

#train model
model = RandomForestClassifier()
model.fit(X, y)

#save model with joblib
joblib.dump(model, 'model.joblib')