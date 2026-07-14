#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv("DateFruit_Dataset.csv")


# In[3]:


df.head()


# In[4]:


df.isnull().sum()


# In[5]:


df.shape


# In[6]:


X = df.drop("Class",axis = 1)
y = df["Class"]


# In[7]:


X.head()


# In[8]:


y.head()


# In[9]:


df["Class"].unique()


# In[10]:


from sklearn.preprocessing import StandardScaler, LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
y


# In[11]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size = 0.2, random_state = 42
)


# In[12]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ### ANN

# In[14]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset


# In[15]:


X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)


# In[16]:


train_dataset = TensorDataset(X_train_tensor,y_train_tensor)
test_dataset = TensorDataset(X_test_tensor,y_test_tensor)


# In[17]:


train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = 32)


# In[18]:


# Build Our Model
class ANN(nn.Module):
    def __init__(self):
     super(ANN,self).__init__()
     self.model = nn.Sequential(
         nn.Linear(X.shape[1],64),
         nn.ReLU(),
         nn.Linear(64,64),
         nn.ReLU(),
         nn.Linear(64,7)
     )
    def forward(self,x):
        return self.model(x)   


# In[19]:


model = ANN()
criteria = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())


# In[20]:


# Training the NN
epochs = 100
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for xb,yb in train_loader:
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criteria(outputs,yb)
        loss.backward()
        optimizer.step()
        running_loss += loss
        train_loss =   running_loss/len(train_loader)  
        print(f"epoch = {epoch+1}/{epochs}, loss = {train_loss}")


# In[23]:


# Evaluate
model.eval()
total = 0
correct = 0
with torch.no_grad():
    for xb,yb in test_loader:
        outputs = model(xb)
        _, predicted = torch.max(outputs,1)

        correct += (predicted == yb).sum().item()
        total += yb.size(0)
print("Total Values:",total)      
print("Correct Values:",correct)    


# In[25]:


print("Accuracy:", correct/total)


# In[ ]:




