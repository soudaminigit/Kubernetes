import pickle
import os
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

X, y = load_iris(return_X_y=True)

model = LogisticRegression(max_iter=200)
model.fit(X, y)

os.makedirs("/models", exist_ok=True)

with open("/models/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved to /models/model.pkl")
