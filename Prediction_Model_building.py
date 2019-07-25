
# coding: utf-8

# In[25]:

import pandas as pd
import numpy as np


# In[26]:

preprocessed_df=pd.read_csv('/home/rootuser/arj/ticket/Data/preprocessed_data.csv')
preprocessed_df.drop(['Unnamed: 0'],1,inplace=True)


# In[27]:

preprocessed_df.head()


# In[28]:

application_name=['dns','https']


# In[29]:

# for i in application_name:
    
df=preprocessed_df.loc[preprocessed_df['Application'] == 'dns']
df=df[['Timestamp','Utilization(bps)']]



# In[30]:

df=df[df['Utilization(bps)']!= 0]


# In[31]:

df.shape


# In[32]:

training_set=df[0:len(df)-3000]
print(training_set.shape)
training_set_sliced = training_set.iloc[:, 1:2].values


# In[33]:

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set_sliced)


# In[34]:

# Creating a data structure with 100 timesteps and 1 output
X_train = []
y_train = []
for i in range(150, len(training_set)):
    X_train.append(training_set_scaled[i-150:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# In[35]:


# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()


# In[36]:


# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 80, return_sequences = True, input_shape = (X_train.shape[1], 1)))


# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 80))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'sgd', loss = 'mean_squared_error')





# In[37]:

regressor.summary()


# In[38]:

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 200, batch_size = 32)


# In[98]:

print('model built successfully')
regressor.save('/home/rootuser/arj/ticket/model/dns-sgd(200 epo- 80 neuron-200 time).h5') 


# In[99]:


real_pattern_prediction=df[len(df)-3000:]


# In[100]:

real_pattern_prediction.head()


# In[101]:

training_set.head()


# In[102]:


# Getting the Random predictiond data
dataset_total = pd.concat((training_set['Utilization(bps)'], real_pattern_prediction['Utilization(bps)']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(real_pattern_prediction) - 150:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(150, len(real_pattern_prediction)+150):
    X_test.append(inputs[i-150:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_pattern = regressor.predict(X_test)
predicted_pattern = sc.inverse_transform(predicted_pattern)


# In[103]:

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
real_pattern_prediction=real_pattern_prediction.iloc[:, 1:2].values
# Visualising the results
plt.plot(real_pattern_prediction, color = 'red', label = 'Real Pattern Number')
plt.plot(predicted_pattern, color = 'blue', label = 'Predicted Pattern Number')
plt.title('Pattern Number Prediction')
plt.xlabel('record')
plt.ylabel('Pattern Number')
plt.legend()
plt.show()


# In[ ]:



