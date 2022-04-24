# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 15:33:02 2022

@author: ANEH
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


#1. Read CSV data
diamond_data = pd.read_csv(r"C:\Users\ANEH\Documents\Deep Learning Class\TensorFlow Deep Learning\Datasets\diamond\diamonds.csv")


#2. Remove unnecessary column
diamond_data = diamond_data.drop('Unnamed: 0', axis=1)


#3. Split data into features and label
diamond_features = diamond_data.copy()
diamond_labels = diamond_features.pop('price')

#4. Check the data
print("------------------Features-------------------------")
print(diamond_features.head())
print("-----------------Label----------------------")
print(diamond_label.head())

#%%
#5. Ordinal encode categorical features
from sklearn.preprocessing import OrdinalEncoder

cut_category = ['Fair','Good','Very Good','Premium','Ideal']
color_category = ['J','I','H','G','F','E','D']
clarity_category = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

ordinal_encoder = OrdinalEncoder(categories=[cut_category,color_category,clarity_category])
diamond_features[['cut','color','clarity']] = ordinal_encoder.fit_transform(diamond_features[['cut','color','clarity']])

#%%
#6. Split the data into train-validation-test sets, with a ratio of 60:20:20
SEED = 12345
x_train,x_iter,y_train,y_iter = train_test_split(diamond_features,diamond_label,test_size=0.4,random_state=SEED)
x_val,x_test,y_val,y_test = train_test_split(x_iter,y_iter,test_size=0.5,random_state=SEED)

#7. Perform feature scaling, using training data for fitting
from sklearn.preprocessing import StandardScaler

standard_scaler = StandardScaler()
standard_scaler.fit(x_train)
x_train = standard_scaler.transform(x_train)
x_val = standard_scaler.transform(x_val)
x_test = standard_scaler.transform(x_test)

#Data preparation is completed 

#%%
#8. Create a feedforward neural network using TensorFlow Keras
number_input = x_train.shape[-1]

model = tf.keras.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=number_input))
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dense(64,activation='relu'))
model.add(tf.keras.layers.Dense(32,activation='relu'))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(1))

model.summary()

#9.Compile the model
model.compile(optimizer='adam',loss='mse',metrics=['mae','mse'])

#%%
#10. Train and evaluate the model with validation data
#Define callback functions: EarlyStopping and Tensorboard
import datetime
import os

base_log_path = r"C:\Users\ANEH\Documents\Deep Learning Class\TensorFlow Deep Learning\Tensorboard\p2_log"
log_path = os.path.join(base_log_path, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path)
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5,verbose=2)
EPOCHS = 100
BATCH_SIZE=64
history = model.fit(x_train,y_train,validation_data=(x_val,y_val),batch_size=BATCH_SIZE,epochs=EPOCHS,callbacks=[tb_callback,es_callback])


#%%
#11. Evaluate with test data for wild testing
import matplotlib.pyplot as plt

test_result = model.evaluate(x_test,y_test,batch_size=BATCH_SIZE)
print(f"Test loss = {test_result[0]}")
print(f"Test MAE = {test_result[1]}")
print(f"Test MSE = {test_result[2]}")

#12. Plot a graph of prediction vs label on test data
predictions = np.squeeze(model.predict(x_test))
labels = np.squeeze(y_test)
plt.plot(predictions,labels,".")
plt.xlabel("Predictions")
plt.ylabel("Labels")
plt.title("Graph of Predictions vs Labels with Test Data")
save_path = r"C:\Users\ANEH\Documents\Deep Learning Class\AI05_repo_2\img"
plt.savefig(os.path.join(save_path,"result.png"),bbox_inches='tight')
plt.show()
