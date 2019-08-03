
# coding: utf-8

# In[25]:

import pandas as pd
import numpy as np


# In[26]:

preprocessed_df=pd.read_csv('C:/Users/ar393556/Documents/Utilization-prediction-RNN-LSTM/Data/preprocessed_data.csv')
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

training_set=df[0:len(df)-150]
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
for i in range(15, len(training_set)):
    X_train.append(training_set_scaled[i-15:i, 0])
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
# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 100, activation = 'relu',return_sequences=True,input_shape = (X_train.shape[1], 1)))

regressor.add(Dropout(0.1))
regressor.add(LSTM(units = 50, activation = 'relu'))
# Adding the output layer
regressor.add(Dense(units = 1))
#regressor.add(Activation("linear"))
# Compiling the RNN
#regressor.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['accuracy'])
regressor.compile(loss="mse", optimizer="adam",metrics=['accuracy'])






# In[37]:

regressor.summary()


# In[38]:

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 1000, shuffle=False,verbose=1, batch_size = 16)


# In[98]:

print('model built successfully')
regressor.save('C:/Users/ar393556/Documents/Utilization-prediction-RNN-LSTM/model/task-scheduler.h5') 



print('model built successfully')
