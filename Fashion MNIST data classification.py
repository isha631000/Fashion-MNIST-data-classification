#!/usr/bin/env python
# coding: utf-8

# In[25]:


import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sms
import matplotlib.pyplot as plt
from tensorflow.keras import datasets,models,layers
import keras


# In[3]:


(training_x,training_y),(test_x,test_y)=datasets.fashion_mnist.load_data()


# In[5]:


training_x[0]


# In[6]:


class_labels=["T-shirts/top","Trouser","Pullover","Dress",'Coat','Sandal','Shirt','Sneaker','Bag',"Ankle boot"]
class_labels


# In[7]:


plt.imshow(training_x[0],cmap='Greys')


# In[8]:


plt.figure(figsize=(16,16))
j=1
for i in np.random.randint(0,1000,25):
    plt.subplot(5,5,j);j+=1
    plt.imshow(training_x[i],cmap="Greys")
    plt.axis("off")
    plt.title('{} / {}'.format(class_labels[training_y[i]],training_y[i]))


# In[10]:


training_x.ndim


# In[11]:


training_x=np.expand_dims(training_x,-1)


# In[12]:


training_x.ndim


# In[13]:


test_x=np.expand_dims(test_x,-1)


# In[14]:


training_x=training_x/255
test_x=test_x/255
from sklearn.model_selection import train_test_split
training_x,x_validation,training_y,y_validation=train_test_split(training_x,training_y,test_size=0.2,random_state=2020)


# In[15]:


training_x.shape,x_validation.shape,training_y.shape,y_validation.shape


# In[17]:


model=keras.models.Sequential([
    keras.layers.Conv2D(filters=32,kernel_size=3,strides=(1,1),padding='valid',activation='relu',input_shape=[28,28,1]),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(units=128,activation='relu'),
    keras.layers.Dense(units=10,activation='softmax')
])
model.summary()


# In[18]:


model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])


# In[19]:


model.fit(training_x,training_y,epochs=10,batch_size=512,verbose=1,validation_data=(x_validation,y_validation))


# In[20]:


y_predict=model.predict(test_x)


# In[21]:


y_predict.round(2)


# In[22]:


model.evaluate(test_x,test_y)


# In[23]:


plt.figure(figsize=(16,16))
j=1
for i in np.random.randint(0,1000,25):
    plt.subplot(5,5,j);j+=1
    plt.imshow(test_x[i].reshape(28,28),cmap="Greys")
    plt.axis("off")
    plt.title('actual= {} / {}\n predicted={} / {}'.format(class_labels[test_y[i]],test_y[i],class_labels[np.argmax(y_predict[i])],np.argmax(y_predict[i])))


# In[24]:


from sklearn.metrics import confusion_matrix
plt.figure(figsize=(16,9))
y_predict_labels=[np.argmax(label) for label in y_predict]
cm=confusion_matrix(test_y,y_predict_labels)


# In[26]:


sms.heatmap(cm,annot=True,fmt='d',xticklabels=class_labels,yticklabels=class_labels)
from sklearn.metrics import classification_report
cr=classification_report(test_y,y_predict_labels,target_names=class_labels)
print(cr)


# In[27]:


model.save('mnist_fashion_cnn')


# In[ ]:




