
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
# coding: utf-8




# In[26]:
preprocessed_df=pd.read_csv('/home/rootuser/arj/ticket/Data/preprocessed_data.csv')
preprocessed_df.drop(['Unnamed: 0'],1,inplace=True)



    
df=preprocessed_df.loc[preprocessed_df['Application'] == 'salesforce']
df=df[['Timestamp','Utilization(kb)']]



# In[30]:

df=df[df['Utilization(kb)']!= 0]


# In[31]:

df.shape


# In[32]:

training_set=df[0:len(df)-150]
print(training_set.shape)
training_set_sliced = training_set.iloc[:, 1:2].values


# In[33]:

# # Feature Scaling
# from sklearn.preprocessing import MinMaxScaler
# sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = training_set_sliced


# In[34]:

# Creating a data structure with 100 timesteps and 1 output
X_train = []
y_train = []
for i in range(12, len(training_set)):
    X_train.append(training_set_scaled[i-12:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# In[35]:


# Importing the Keras libraries and packages

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense , BatchNormalization , Dropout , Activation
from keras.layers import LSTM , GRU
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.optimizers import Adam , SGD , RMSprop


# Initialising the RNN
regressor = Sequential()


# In[36]:


# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 128,return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))


# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 128))
regressor.add(Dropout(0.1))


# Adding the output layer
regressor.add(Dense(units = 1))
regressor.add(Activation("linear"))
# Compiling the RNN
regressor.compile(optimizer = "adam", loss = 'mse')



#model.compile(loss='mean_squared_error', optimizer=Adam(lr = 0.0005) , metrics = ['accuracy'])

history = regressor.fit(X_train, y_train, epochs=1000 , batch_size = 16, shuffle=False,verbose=1 )



regressor.save('/home/rootuser/arj/ticket/model/sales-force1.h5') 

print('Svaed Successfully')
