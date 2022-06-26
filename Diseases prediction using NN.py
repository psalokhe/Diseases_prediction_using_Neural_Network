import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

#importing the test data 
df=pd.read_csv('training_data.csv') 
col=df.columns

#cleaning the data and preparing the data for the neural network
df.dropna()
class_name=df['prognosis'].unique()
dummies = pd.get_dummies(df['prognosis'])
tdata=pd.concat([df,dummies],axis=1)
tdata.drop(['prognosis'],axis=1,inplace = True)
fdata=tdata.sample(frac=1)
inputdata = fdata.iloc[:,:132]
outputdata = fdata.iloc[:, -41:]

#converting the data into numpy array 
X = np.array(inputdata)
X=np.expand_dims(X,axis=1)
Y= np.array(outputdata)

#spliting the data into test and train dataset 
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.1)

#building the neural network 
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(1,132)),
    keras.layers.Dense(8,activation='relu',kernel_regularizer=keras.regularizers.l1(0.01)), #implementing regularization to reduce overfitting 
    keras.layers.Dense(8,activation='relu',kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.Dense(41,activation="softmax") #data has 41 unique diseases 
    ])

#compiling the model to our dataset
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=['accuracy']) 

#training the model
model.fit(x_train,y_train,epochs=300) 

#importing test data for the prediction 
df2=pd.read_csv('test_data.csv')
numdf2 = np.array(df2)
numdf2 = np.expand_dims(numdf2,axis=1)

#calculating the test loss and test accuracy of the model
test_loss,test_acc = model.evaluate(x_test,y_test)
print("test_acc: ", test_acc)

#making the prediction with the model
prediction = model.predict(numdf2)
print("prediction: "+class_name[np.argmax(prediction[0])])

    



