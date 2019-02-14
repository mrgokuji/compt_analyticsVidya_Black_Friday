import os
print(os.listdir("../input"))

import pandas as pd
import numpy as np

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train["Product_Category_2"].replace(np.NaN,0,inplace = True)
train["Product_Category_3"].replace(np.NaN,0,inplace = True)

train["dummy"] = 1
test["dummy"] = 0
combined = pd.concat([train,test])

df = pd.get_dummies(combined["Gender"])

combined = pd.concat([combined,df],axis=1)
df = pd.get_dummies(combined["City_Category"])
combined = pd.concat([combined,df],axis=1)

combined.head()
#combined.Age[combined.Age=="0-17"]="16"
#combined.Age[combined.Age=="55+"]="60"
combined.Age.replace("0-17",16, inplace= True)
combined.Age.replace("55+",60,inplace=True)
combined.Age[combined.Age=="26-35"] = 30
combined.Age[combined.Age=="46-50"] = 48
combined.Age.replace("51-55",53, inplace= True)
combined.Age[combined.Age=="36-45"] = 40
combined.Age[combined.Age=="18-25"] = 22
combined.Stay_In_Current_City_Years[combined.Stay_In_Current_City_Years=="4+"] = 6

combined.pop("Gender")
combined.pop("City_Category")

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
combined.Product_ID = le.fit_transform(combined.Product_ID)

combined.Age=pd.to_numeric(combined.Age)
combined.Product_ID = pd.to_numeric(combined.Product_ID)
combined.Stay_In_Current_City_Years = pd.to_numeric(combined.Stay_In_Current_City_Years)

train = combined[combined["dummy"]==1]
test = combined[combined["dummy"]==0]

test.pop("dummy")
train.pop("dummy")

X = train.drop(labels="Purchase",axis = 1)
y = train.Purchase

from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential()

model.add(Dense(44, input_dim=14))
model.add(Activation('relu'))

model.add(Dense(70))
model.add(Activation('relu'))

model.add(Dense(200))
model.add(Activation('relu'))

model.add(Dense(60))
model.add(Activation('relu'))

model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X,y,epochs=100,verbose=0)

test.pop("Purchase")

sub = pd.read_csv("sub.csv")
sub["Purchase"]=model.predict(test)

sub["User_ID"] = test["User_ID"]

sub["Product_ID"] = test["Product_ID"]

sub.to_csv("sub.csv")
