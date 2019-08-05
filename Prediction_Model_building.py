
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

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set_sliced)


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

filepath="stock_weights.hdf5"
from keras.callbacks import ReduceLROnPlateau , ModelCheckpoint
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.1, epsilon=0.0001, patience=1, verbose=1)
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='max')

# Initialising the RNN
regressor = Sequential()


# In[36]:

model = Sequential()
model.add(GRU(256 , input_shape = (X_train.shape[1], 1), return_sequences=True))
model.add(Dropout(0.4))
model.add(LSTM(256))
model.add(Dropout(0.4))
model.add(Dense(64 ,  activation = 'relu'))
model.add(Dense(1))
print(model.summary())

model.compile(loss='mean_squared_error', optimizer=Adam(lr = 0.0005) , metrics = ['accuracy'])


history = model.fit(X_train, y_train, epochs=2 , batch_size = 128 , 
          callbacks = [checkpoint , lr_reduce], shuffle=False,verbose=1 )




# In[98]:

print('model built successfully')
model.save('C:/Users/ar393556/Documents/Utilization-prediction-RNN-LSTM/model/new-dns-fine-tuning-task-scheduler.h5') 



print('model built successfully')
