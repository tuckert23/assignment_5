import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import scatterplotmatrix
import numpy as np
from mlxtend.plotting import heatmap

# Importing dataset
df = pd.read_csv("./prostate-cancer-prediction.csv")
columns = ["id", "diagnosis_result", "radius", "texture", "perimeter", "area", "smoothness",
           "compactness", "symmetry", "fractal_dimension"]

df.columns = columns


df.diagnosis_result = df.diagnosis_result.replace("M", 1)
df.diagnosis_result = df.diagnosis_result.replace("B", 0)

# creating a scatter plot matrix
scatterplotmatrix(df[columns].values, figsize=(30, 30), names=columns, alpha=0.8)
plt.tight_layout()
plt.savefig("./figs/scatter_plot_matrix.png")

# creating the correlating heatmap
corrmap = np.corrcoef(df[columns].values.T)
hmap = heatmap(corrmap, row_names=columns, column_names=columns, figsize=(20, 20))
plt.savefig("./figs/heatmap.png")
