
# coding: utf-8

# In[33]:

import pandas as pd
import numpy as np


# In[34]:

#preprocessed_df=pd.read_csv('/home/rootuser/arj/ticket/Data/preprocessed_data.csv')
preprocessed_df=pd.read_csv('C:/Users/ar393556/Documents/Utilization-prediction-RNN-LSTM/Data/preprocessed_data.csv')
preprocessed_df.drop(['Unnamed: 0'],1,inplace=True)


# In[35]:

preprocessed_df.head()


# In[36]:

application_name=['dns','https']


# In[37]:

# for i in application_name:
    
df=preprocessed_df.loc[preprocessed_df['Application'] == 'dns']
df=df[['Timestamp','Utilization(bps)']]
training_set=df[0:len(df)-150]
print(training_set.shape)
training_set_sliced = training_set.iloc[:, 1:2].values


# In[ ]:




# In[38]:

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set_scaled = sc.fit_transform(training_set_sliced)


# In[39]:

# Creating a data structure with 100 timesteps and 1 output
X_train = []
y_train = []
for i in range(12, len(training_set)):
    X_train.append(training_set_scaled[i-12:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# In[40]:

X_train.shape


# In[41]:

y_train.shape


# In[42]:


# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Activation

# Initialising the RNN
regressor = Sequential()


# In[43]:

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 100, activation = 'relu',return_sequences=True,input_shape = (X_train.shape[1], 1)))

regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, activation = 'relu'))
# Adding the output layer
regressor.add(Dense(units = 1))
regressor.add(Activation("linear"))
# Compiling the RNN
#regressor.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['accuracy'])
regressor.compile(loss="mse", optimizer="adam",metrics=['accuracy'])


# In[44]:

regressor.summary()


# In[ ]:

print(5)


# In[ ]:

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 1,shuffle=False,verbose=1, batch_size = 16)


# In[ ]:

print('model built successfully')
#regressor.save('/home/rootuser/arj/ticket/model/dns-relu(1000 epo- 100 neuron-288 time).h5') 
regressor.save('C:/Users/ar393556/Documents/Utilization-prediction-RNN-LSTM/model/service-dns-relu-new(1000 epo- 100*50 neuron-12 time).h5') 


# In[26]:


real_pattern_prediction=df[len(df)-150:]


# In[27]:

real_pattern_prediction.head()


# In[28]:

training_set.head()


# In[29]:


# Getting the Random predictiond data
dataset_total = pd.concat((training_set['Utilization(bps)'], real_pattern_prediction['Utilization(bps)']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(real_pattern_prediction) - 12:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(12, len(real_pattern_prediction)+12):
    X_test.append(inputs[i-12:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_pattern = regressor.predict(X_test)
predicted_pattern = sc.inverse_transform(predicted_pattern)


# In[30]:

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
real_pattern_prediction=real_pattern_prediction.iloc[:, 1:2].values
# Visualising the results
plt.plot(real_pattern_prediction, color = 'red', label = 'Real Utilization Number')
plt.plot(predicted_pattern, color = 'blue', label = 'Predicted Utilization Number')
plt.title('12 timeskips, 100*50 neurons, 1000 epochs-DNS Utilization Prediction')
plt.xlabel('record no.')
plt.ylabel('Utilization')
plt.legend()
plt.show()


# In[ ]:



