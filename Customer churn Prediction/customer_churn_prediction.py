import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
data = pd.read_csv(r"D:\datasets\Churn_Modelling.csv")
data.drop(columns=["RowNumber","CustomerId"],inplace=True)
df= pd.get_dummies(data,columns=["Geography","Gender"],drop_first=True,dtype=int)
x = df.drop(columns=["Exited","Surname"])
y = df["Exited"]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train) 
x_test_scaled = scaler.transform(x_test)
model = Sequential()
model.add(Dense(11,activation="relu",input_dim = 11))
model.add(Dense(11,activation="relu"))
model.add(Dense(1,activation = "sigmoid"))
print(model.summary())
model.compile(loss="binary_crossentropy",optimizer = "Adam",metrics=["accuracy"])
model.fit(x_train_scaled,y_train,epochs=100,validation_split=0.2)
y_pred = model.predict(x_test_scaled)
y_pred = np.where(y_pred>0.5,1,0)
accuracy = accuracy_score(y_test,y_pred)
print(accuracy)





