from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split

# Importing dataset
df = pd.read_csv("./prostate-cancer-prediction.csv")
columns = ["id", "diagnosis_result", "radius", "texture", "perimeter", "area", "smoothness",
           "compactness", "symmetry", "fractal_dimension"]

df.columns = columns

# changing result M and B to 1 and 0 respectively
i = 0
for result in df["diagnosis_result"]:
    if result == "M":
        df.diagnosis_result.iloc[i] = 1
    else:
        df.diagnosis_result.iloc[i] = 0
    i += 1
    
# dont need the ID column!
df.drop("id", inplace=True, axis=1)

# scale using MaxAbsScaler to maintain spatial relationships
MaxAbsScaler(df)
print(df)

x = df[["radius", "texture", "perimeter", "area", "smoothness",
        "compactness", "symmetry", "fractal_dimension"]]
y = df["diagnosis_result"]

# split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)

# create model
model = KNeighborsClassifier(n_neighbors=5, leaf_size=5)

# train model
model.fit(x_train, y_train)

print("Score:", model.score(x_test, y_test))
