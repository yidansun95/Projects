#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Name:Yidan Sun
#HW_6:K-NN classifier


# In[11]:


import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import math
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# In[5]:


df=pd.read_csv('SBUX_weekly_return_volatility_1.csv')


# In[6]:


df = df[(df['Year'] == 2020) | (df['Year'] == 2021)]


# In[7]:


df['k=1']=''
df['k=3']=''
df['k=5']=''
df['k=7']=''
df['k=9']=''
df['k=11']=''


# In[9]:


df.loc[106,'k=3']='green'
df.loc[158,'k=3']='green'


# In[13]:


X = df[['Week_Number']]
y = df['True Label']

knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
knn.fit(X, y)

prediction= knn.predict(X)


# In[20]:


#Implement Strategy:starts with 100, ends with -96.698355
df.loc[159,'Trading Strategy k=1']=100*(1+(-0.22660))

i=159
while i>=159 and i<=209:
    i+=1
    if df.loc[i,'k=1']=='red':
        df.loc[i,'Trading Strategy k=1']=df.loc[i-1,'Trading Strategy k=1']
    else:
        df.loc[i,'Trading Strategy k=1']=df.loc[i-1,'Trading Strategy k=1']*(1+df.loc[i,'mean_return'])


# In[22]:


acc=accuracy_score(y,prediction)
cf=confusion_matrix(y,prediction)
print(acc)
print(cf)


# In[29]:


X = df[['Week_Number']]
y = df['True Label']

knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn.fit(X, y)
df['k=3'] = knn.predict(X)

prediction= knn.predict(X)


# In[30]:


acc=accuracy_score(y,prediction)
cf=confusion_matrix(y,prediction)
print(acc)
print(cf)


# In[34]:


X = df[['Week_Number']]
y = df['True Label']

knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X, y)

df['k=5'] = knn.predict(X)
prediction=knn.predict(X)


# In[28]:


acc=accuracy_score(y,prediction)
cf=confusion_matrix(y,prediction)
print(acc)
print(cf)


# In[35]:


#Implement Strategy:starts with 100, ends with -96.698355
df.loc[159,'Trading Strategy k=5']=100*(1+(-0.22660))

i=159
while i>=159 and i<=209:
    i+=1
    if df.loc[i,'k=5']=='red':
        df.loc[i,'Trading Strategy k=5']=df.loc[i-1,'Trading Strategy k=5']
    else:
        df.loc[i,'Trading Strategy k=5']=df.loc[i-1,'Trading Strategy k=5']*(1+df.loc[i,'mean_return'])


# In[36]:


df


# In[9]:


X = df[['Week_Number']]
y = df['True Label']

knn = KNeighborsClassifier(n_neighbors=7, metric='euclidean')
knn.fit(X, y)

df['k=7'] = knn.predict(X)


# In[10]:


X = df[['Week_Number']]
y = df['True Label']

knn = KNeighborsClassifier(n_neighbors=9, metric='euclidean')
knn.fit(X, y)

df['k=9'] = knn.predict(X)


# In[11]:


X = df[['Week_Number']]
y = df['True Label']

knn = KNeighborsClassifier(n_neighbors=11, metric='euclidean')
knn.fit(X, y)

df['k=11'] = knn.predict(X)


# In[12]:


df.loc[df['Year'] == 2021, 'k=5'] = ''
df.loc[df['Year'] == 2021, 'k=7'] = ''
df.loc[df['Year'] == 2021, 'k=9'] = ''
df.loc[df['Year'] == 2021, 'k=11'] = ''


# In[13]:


#Question1
i=105
n_correct=0
n=0
while i>=105 and i <= 157:
    i+=1
    n+=1
    if df.loc[i,'k=3']==df.loc[i,'True Label']:
        n_correct+=1
    else:
        continue
print(f'The accuracy for k=3 is: {(n_correct/n)*100}%')


# In[14]:


#Question1
i=105
n_correct=0
n=0
while i>=105 and i <= 157:
    i+=1
    n+=1
    if df.loc[i,'k=5']==df.loc[i,'True Label']:
        n_correct+=1
    else:
        continue
print(f'The accuracy for k=5 is: {(n_correct/n)*100}%')


# In[15]:


#Question1
i=105
n_correct=0
n=0
while i>=105 and i <= 157:
    i+=1
    n+=1
    if df.loc[i,'k=7']==df.loc[i,'True Label']:
        n_correct+=1
    else:
        continue
print(f'The accuracy for k=7 is: {(n_correct/n)*100}%')


# In[16]:


#Question1
i=105
n_correct=0
n=0
while i>=105 and i <= 157:
    i+=1
    n+=1
    if df.loc[i,'k=9']==df.loc[i,'True Label']:
        n_correct+=1
    else:
        continue
print(f'The accuracy for k=9 is: {(n_correct/n)*100}%')


# In[17]:


#Question1
i=105
n_correct=0
n=0
while i>=105 and i <= 157:
    i+=1
    n+=1
    if df.loc[i,'k=11']==df.loc[i,'True Label']:
        n_correct+=1
    else:
        continue
print(f'The accuracy for k=11 is: {(n_correct/n)*100}%')


# In[18]:


#Question1
#Answer: K=3 is optimal value with accuracy of 71.7%

x = [3,5,7,9,11]

y = [71.70,56.60,54.72,62.26,56.60]

# Create a bar chart
plt.bar(x, y)

# Customize the plot
plt.title('K Accuracy')
plt.xlabel('K Value')
plt.ylabel('Frequency(%) ')
plt.xticks(x)

# Show the histogram
plt.show()


# In[21]:


df[df['Year']==2021]


# In[23]:


#Question 2:
#The accuracy of using year 1 optimal value(k=3) is 78.85%
i=158
n_correct=0
n=0
while i>=158 and i <= 209:
    i+=1
    n+=1
    if df.loc[i,'k=3']==df.loc[i,'True Label']:
        n_correct+=1
    else:
        continue
print(f'The 2021 accuracy for k=3 is: {(n_correct/n)*100}%')


# In[24]:


#Question 3&4

#2021 True Positive is: 19
#2021 True Negative is: 22
#2021 False Positive is: 9
#2021 False Negative is: 2

i=158
TP=0
TN=0
FN=0
FP=0
while i>=158 and i <= 209:
    i+=1
    n+=1
    if df.loc[i,'k=3']==df.loc[i,'True Label'] and df.loc[i,'k=3']=='green':
        TP+=1
    elif df.loc[i,'k=3']==df.loc[i,'True Label'] and df.loc[i,'k=3']=='red':
        TN+=1
    elif df.loc[i,'k=3']!=df.loc[i,'True Label'] and df.loc[i,'k=3']=='green':
        FP+=1
    elif df.loc[i,'k=3']!=df.loc[i,'True Label'] and df.loc[i,'k=3']=='red':
        FN+=1
    else:
        continue
print(f'2021 True Positive is: {TP}')
print(f'2021 True Negative is: {TN}')
print(f'2021 False Positive is: {FP}')
print(f'2021 False Negative is: {FN}')


# In[43]:


#Question 5
#Buy and Hold Strategy in 2021 starts with 100 dollars and ends with 0.051270 dollars
#While Implementing k=3 Trading Strategy in 2021, ends with -96.698355 dollars 
#It turns out Buy and Hold strategy has larger amount in end of year.

df.loc[159,'buy and hold']=100*(1+(-0.22660))
i=159
while i>=159 and i<=209:
    i+=1
    df.loc[i,'buy and hold']=df.loc[i-1,'buy and hold']*(1+df.loc[i,'mean_return'])


# In[31]:


#Implement Strategy:starts with 100, ends with -96.698355
df.loc[159,'Trading Strategy k=3']=100*(1+(-0.22660))

i=159
while i>=159 and i<=209:
    i+=1
    if df.loc[i,'k=3']=='red':
        df.loc[i,'Trading Strategy k=3']=df.loc[i-1,'Trading Strategy k=3']
    else:
        df.loc[i,'Trading Strategy k=3']=df.loc[i-1,'Trading Strategy k=3']*(1+df.loc[i,'mean_return'])


# In[33]:


df[df['Year']==2021]


# In[ ]:




