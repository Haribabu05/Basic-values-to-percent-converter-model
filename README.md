# Basic-values-to-percent-converter-model
#this source code creates a single layer model which converts values to percentile of 120.
#code
import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf
values = np.array([45,67,92,32,53],dtype = float)
percent = np.array([57,80.4,110.4,38.4,63.6],dtype = float)
l= tf.keras.layers.Dense(units =1,input_shape= [1])
model = tf.keras.Sequential(l)
model.compile(loss='mean_absolute_error',optimizer=tf.keras.optimizers.Adam(0.01))
calculate = model.fit(values,percent,epochs= 700,verbose = False)
print(model.predict([56.0]))
