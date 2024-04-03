#%% 
import numpy as np
import pandas as pd
import pickle 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import string

nr_to_letter = {k:v.upper() for k,v in enumerate(list(string.ascii_lowercase))}
df=pd.read_csv(r"C:\Users\ferid\Downloads\archive2\handwritten_data_785.csv")
X=df.values[:,1:]
y=df.values[:,0]
# X=X.reshape(len(X),28,28)
X=X/255.0

# y = np.eye(len(np.unique(y)))[y]

X, y = X[:100000], y[:100000]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# model=SVC().fit(X_train,y_train)
# pickle.dump(model,open("model.pkl","wb"))
# model.score(X_test,y_test)

# %%
