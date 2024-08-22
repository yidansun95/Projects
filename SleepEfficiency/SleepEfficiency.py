#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Sleep and Efficiency
#Name: Yidan Sun
#Date: 10/5/2023


# In[2]:


import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import math
import os


# In[3]:


#import csv into Pandas
df=pd.read_csv('Sleep_Efficiency.csv')


# In[4]:


# print out the results of different entries in each column
# ID:452 / Age:61 / Bedtime:424/ Wakeup time:434/ Sleep Duration:9/ Sleep Efficiency:50/ 
# REM sleep percentage:11/Deep sleep percentage:18/Light sleep percentage:21/ Awakenings:5/Caffeine consumption:6
# Alcohol consumption:6/Smoking status:2/Exercise frequency:6
df.nunique()


# In[5]:


# [Awakenings]:20 [Caffeine consumption]:25 [Alcohol consumption]:16
df.isnull().sum()


# In[6]:


# Missing data in Awakenings
# Female 48,33,37,40,34,41,44,35,28 
# Male 52,53,19,55,25,24,61

df[df['Awakenings'].isnull()] # filtered by awakenings column missing data


# In[7]:


# 52 Male data ,pick first 5 data
df[(df['Age']==52) & (df['Gender']=='Male')]


# In[8]:


# View52 Male, calculate average and assign value in table
aw_52m=(1+1+0+1+2)/5
print(aw_52m)

# Assign value to #52 Male Awakenings
df.loc[19,'Awakenings']=aw_52m
df.loc[337,'Awakenings']=aw_52m
df.loc[352,'Awakenings']=aw_52m


# In[9]:


# View 53 Male data, pick 5 data
df[(df['Age']==53) & (df['Gender']=='Male')]


# In[10]:


# Calculate average and assign value in table
aw_53m=(0+1+4+3+3)/5
print(aw_53m)
df.loc[85,'Awakenings']=aw_53m


# In[11]:


# 19 Male data, pick 5 
df[(df['Age']<=21) & (df['Gender']=='Male')]


# In[12]:


#Calculate average and assign value in table
aw_19m=(3+2+4+1+2)/5
print(aw_19m)
df.loc[123,'Awakenings']=aw_19m


# In[13]:


# View 48 Female data, pick 5
df[(df['Age']==48) & (df['Gender']=='Female')]


# In[14]:


#Calculate average and assign value in table
aw_48f=(1+0+1+0+2)/5
print(aw_48f)
df.loc[135,'Awakenings']=aw_48f


# In[15]:


#View 55 Female data, pick 5
df[(df['Age']==55) & (df['Gender']=='Male')]


# In[16]:


#Calculate average and assign value in table
aw_55m=(0+1+1+1+1)/5
print(aw_55m)
df.loc[138,'Awakenings']=aw_55m


# In[17]:


#View 25 Male data, pick 5
df[(df['Age']==25) & (df['Gender']=='Male')]


# In[18]:


#Calculate average and assign value in table
aw_25m=(4+0+2+2+0)/5
print(aw_25m)
df.loc[143,'Awakenings']=aw_25m


# In[19]:


# 33 Male data, pick 5
df[(df['Age']==33)  & (df['Gender']=='Female')| (df['Age']==32)& (df['Gender']=='Female')]


# In[20]:


#Calculate average and assign value in table
aw_33f=(2+1+4+0+1)/5
print(aw_33f)
df.loc[149,'Awakenings']=aw_33f


# In[21]:


# View 24 Male data, pick 5
df[(df['Age']==24) & (df['Gender']=='Male')]


# In[22]:


#Calculate average and assign value in table
aw_24m=(3+1+3+0+0)/5
print(aw_24m)
df.loc[170,'Awakenings']=aw_24m
df.loc[404,'Awakenings']=aw_24m


# In[23]:


# View 37 Female data, pick 5
df[(df['Age']==37) & (df['Gender']=='Female')]


# In[24]:


#Calculate average and assign value in table
aw_37f=(0+1+1+2+3)/5
print(aw_37f)
df.loc[244,'Awakenings']=aw_37f


# In[25]:


# View 61 Male data, pick 5 
df[(df['Age']==61) & (df['Gender']=='Male')]


# In[26]:


#Calculate average and assign value in table
aw_61m=(1+3+1+0+1)/5
print(aw_61m)
df.loc[249,'Awakenings']=aw_61m


# In[27]:


# View 40 Female data, pick 5 
df[(df['Age']==40) & (df['Gender']=='Female')]


# In[28]:


#Calculate average and assign value in table
aw_40f=(1+3+3+1+4)/5
print(aw_40f)
df.loc[287,'Awakenings']=aw_40f
df.loc[315,'Awakenings']=aw_40f


# In[29]:


# View 34 Female data, pick 5 
# View 35 Female data, pick 5 
df[(df['Age']==34) & (df['Gender']=='Female')|(df['Age']==35) & (df['Gender']=='Female')]


# In[30]:


#Calculate average and assign value in table
aw_35f=(1+0+2+1+3)/5
print(aw_35f)
df.loc[407,'Awakenings']=aw_35f

aw_34f=(3+3+4+1+0)/5
print(aw_34f)
df.loc[288,'Awakenings']=aw_34f


# In[31]:


# View 41 Female data, pick 5 
df[(df['Age']==41) & (df['Gender']=='Female')]


# In[32]:


#Calculate average and assign value in table
aw_41f=(3+1+0+4+4)/5
print(aw_41f)
df.loc[307,'Awakenings']=aw_41f


# In[33]:


# View 44 Female data, pick 5 
df[(df['Age']==44) & (df['Gender']=='Female')]


# In[34]:


#Calculate average and assign value in table
aw_44f=(4+0+1+1+4)/5
print(aw_44f)
df.loc[327,'Awakenings']=aw_44f


# In[35]:


# View 28 Female data, pick 5 
df[(df['Age']==28) & (df['Gender']=='Female')]


# In[36]:


#Calculate average and assign value in table
aw_28f=(1+2+1+2+1)/5
print(aw_28f)
df.loc[434,'Awakenings']=aw_28f


# In[37]:


# All missing data is filled for [Awakenings]
df[df['Awakenings'].isnull()] 


# In[38]:


df[df['Caffeine consumption'].isnull()]
# Missing data in Caffeine consumption

# Female 36,20,32,35,55,37,22,44,27,40
# Male 24,43,30,32,52,48,42,38,25,39,21


# In[39]:


# View 36 Female data, pick 5 
df[(df['Age']==36) & (df['Gender']=='Female')]


# In[40]:


#Calculate average and assign value in table
ca_36f=(25+50+50+50+25)/5
print(ca_36f)
df.loc[26,'Caffeine consumption']=ca_36f
df.loc[5,'Caffeine consumption']=ca_36f


# In[41]:


# View 20 Female data, pick 5 
df[(df['Age']<=21) & (df['Gender']=='Female')]


# In[42]:


#Calculate average and assign value in table
ca_20f=(25+25+0+0+25)/5
print(ca_20f)
df.loc[37,'Caffeine consumption']=ca_20f


# In[43]:


# View 32 Female data, pick 5 
df[(df['Age']==32) & (df['Gender']=='Female')]


# In[44]:


#Calculate average and assign value in table
ca_32f=(25+25+25+50+50)/5
print(ca_32f)
df.loc[60,'Caffeine consumption']=ca_32f


# In[45]:


# View 35 Female data, pick 5 
df[(df['Age']==35) & (df['Gender']=='Female')|(df['Age']==34)&(df['Gender']=='Female')]


# In[46]:


#Calculate average and assign value in table
ca_35f=(25+25+25+25+25)/5
print(ca_35f)
df.loc[63,'Caffeine consumption']=ca_35f


# In[47]:


# View 55 Female data, pick 5 
df[(df['Age'] == 55) & (df['Gender'] == 'Female')|(df['Age'] == 56) & (df['Gender'] == 'Female')]


# In[48]:


#Calculate average and assign value in table
ca_55f=(25+50+0+25+50)/5
print(ca_55f)
df.loc[169,'Caffeine consumption']=ca_55f


# In[49]:


# View 37 Female data, pick 5 
df[(df['Age'] == 37) & (df['Gender'] == 'Female')]


# In[50]:


#Calculate average and assign value in table
ca_37f=(25+0+0+25+50)/5
print(ca_37f)
df.loc[186,'Caffeine consumption']=ca_37f
df.loc[392,'Caffeine consumption']=ca_37f


# In[51]:


# View 22 Female data, pick 5 
df[(df['Age'] == 22) & (df['Gender'] == 'Female')|(df['Age'] == 23) & (df['Gender'] == 'Female')]


# In[52]:


#Calculate average and assign value in table
ca_22f=(25+0+50+25+50)/5
print(ca_22f)
df.loc[215,'Caffeine consumption']=ca_22f


# In[53]:


# View 44 Female data, pick 5 
df[(df['Age'] == 44) & (df['Gender'] == 'Female')]


# In[54]:


#Calculate average and assign value in table
ca_44f=(25+0+50+50+50)/5
print(ca_44f)
df.loc[270,'Caffeine consumption']=ca_44f
df.loc[324,'Caffeine consumption']=ca_44f


# In[55]:


# View 27 Female data, pick 5 
df[(df['Age'] == 27) & (df['Gender'] == 'Female')]


# In[56]:


#Calculate average and assign value in table
ca_27f=(25+75+50+50+50)/5
print(ca_27f)
df.loc[442,'Caffeine consumption']=ca_27f


# In[57]:


# View 40 Female data, pick 5 
df[(df['Age'] == 40) & (df['Gender'] == 'Female')]


# In[58]:


#Calculate average and assign value in table
ca_40f=(25+0+50+50+50)/5
print(ca_40f)
df.loc[449,'Caffeine consumption']=ca_40f


# In[59]:


# View 24 Male data, pick 5 
df[(df['Age'] == 24) & (df['Gender'] == 'Male')]


# In[60]:


#Calculate average and assign value in table
ca_24m=(0+0+50+50+0)/5
print(ca_24m)
df.loc[24,'Caffeine consumption']=ca_24m
df.loc[404,'Caffeine consumption']=ca_24m


# In[61]:


# View 43 Male data, pick 5 
df[(df['Age'] == 43) & (df['Gender'] == 'Male')|(df['Age'] == 44) & (df['Gender'] == 'Male')]


# In[62]:


#Calculate average and assign value in table
ca_43m=(0+0+0+0+0)/5
print(ca_43m)
df.loc[57,'Caffeine consumption']=ca_43m


# In[63]:


# View 30 Male data, pick 5 
df[(df['Age'] == 30) & (df['Gender'] == 'Male')|(df['Age'] == 29) & (df['Gender'] == 'Male')|(df['Age'] == 31) & (df['Gender'] == 'Male')]


# In[64]:


#Calculate average and assign value in table
ca_30m=(0+0+0+50+200)/5
print(ca_30m)
df.loc[64,'Caffeine consumption']=ca_30m


# In[65]:


# View 32 Male data, pick 5 
df[(df['Age'] == 32) & (df['Gender'] == 'Male')|(df['Age'] == 31) & (df['Gender'] == 'Male')|(df['Age'] == 30) & (df['Gender'] == 'Male')]


# In[66]:


#Calculate average and assign value in table
ca_32m=(50+50+0+50+0)/5
print(ca_32m)
df.loc[114,'Caffeine consumption']=ca_32m


# In[67]:


# View 52 Male data, pick 5 
df[(df['Age'] == 52) & (df['Gender'] == 'Male')]


# In[68]:


#Calculate average and assign value in table
ca_52m=(50+0+0+25+25)/5
print(ca_52m)
df.loc[136,'Caffeine consumption']=ca_52m


# In[69]:


# View 48 Male data, pick 5 
df[(df['Age'] == 48) & (df['Gender'] == 'Male')]


# In[70]:


#Calculate average and assign value in table
ca_48m=(0+0+0+0+0)/5
print(ca_48m)
df.loc[164,'Caffeine consumption']=ca_48m


# In[71]:


# View 42 Male data, pick 5 
df[(df['Age'] == 42) & (df['Gender'] == 'Male')|(df['Age'] == 43) & (df['Gender'] == 'Male')]


# In[72]:


#Calculate average and assign value in table
ca_42m=(0+0+0+200+0)/5
print(ca_42m)
df.loc[175,'Caffeine consumption']=ca_42m


# In[73]:


# View 38 Male data, pick 5 
df[(df['Age'] == 38) & (df['Gender'] == 'Male')|(df['Age'] == 37) & (df['Gender'] == 'Male')]


# In[74]:


#Calculate average and assign value in table
ca_38m=(0+0+75+75+0)/5
print(ca_38m)
df.loc[203,'Caffeine consumption']=ca_38m


# In[75]:


# View 25 Male data, pick 5 
df[(df['Age'] == 25) & (df['Gender'] == 'Male')]


# In[76]:


#Calculate average and assign value in table
ca_25m=(0+100+25+50+0)/5
print(ca_25m)
df.loc[321,'Caffeine consumption']=ca_25m


# In[77]:


# View 39 Male data, pick 5 
df[(df['Age'] == 39) & (df['Gender'] == 'Male')|(df['Age'] == 40) & (df['Gender'] == 'Male')]


# In[78]:


#Calculate average and assign value in table
ca_39m=(0+0+0+0+0)/5
print(ca_39m)
df.loc[355,'Caffeine consumption']=ca_39m


# In[79]:


# View 21 Male data, pick 5 
df[(df['Age'] == 21) & (df['Gender'] == 'Male')|(df['Age'] == 22) & (df['Gender'] == 'Male')]


# In[80]:


#Calculate average and assign value in table
ca_21m=(0+50+50+0+0)/5
print(ca_21m)
df.loc[390,'Caffeine consumption']=ca_21m


# In[81]:


# Caffeine consumption missing values are filled 
df[df['Caffeine consumption'].isnull()]


# In[82]:


#Check the missing value in Alcohol consumption

# Male: 24,44,25,27,56,55,26
# Female: 37,26,14,36,13,50,28
df[df['Alcohol consumption'].isnull()]


# In[83]:


# View 24 Male data, pick 5 
df[(df['Age'] == 24) & (df['Gender'] == 'Male')]


# In[84]:


#Calculate average and assign value in table
al_24m=(5+1+0+3+3)/5
print(al_24m)
df.loc[20,'Alcohol consumption']=al_24m
df.loc[192,'Alcohol consumption']=al_24m


# In[85]:


# View 44 Male data, pick 5 
df[(df['Age'] == 44) & (df['Gender'] == 'Male')|(df['Age'] == 45) & (df['Gender'] == 'Male')]


# In[86]:


#Calculate average and assign value in table
al_44m=(0+0+0+0+3)/5
print(al_44m)
df.loc[75,'Alcohol consumption']=al_44m


# In[87]:


# View 25 Male data, pick 5 
df[(df['Age'] == 25) & (df['Gender'] == 'Male')]


# In[88]:


#Calculate average and assign value in table
al_25m=(4+0+0+3+0)/5
print(al_25m)
df.loc[129,'Alcohol consumption']=al_25m


# In[89]:


# View 27 Male data, pick 5 
df[(df['Age'] == 27) & (df['Gender'] == 'Male')]


# In[90]:


#Calculate average and assign value in table
al_27m=(2+0+2+0+5)/5
print(al_27m)
df.loc[187,'Alcohol consumption']=al_27m


# In[91]:


# View 56 Male data, pick 5 
df[(df['Age'] == 56) & (df['Gender'] == 'Male')]


# In[92]:


#Calculate average and assign value in table
al_56m=(2+0+0+0+0)/5
print(al_56m)
df.loc[395,'Alcohol consumption']=al_56m
df.loc[405,'Alcohol consumption']=al_56m


# In[93]:


# View 55 Male data, pick 5 
df[(df['Age'] == 55) & (df['Gender'] == 'Male')]


# In[94]:


#Calculate average and assign value in table
al_55m=(3+0+0+4+0)/5
print(al_55m)
df.loc[420,'Alcohol consumption']=al_55m


# In[95]:


# View 26 Male data, pick 5 
df[(df['Age'] == 26) & (df['Gender'] == 'Male')|(df['Age'] == 25) & (df['Gender'] == 'Male')]


# In[96]:


#Calculate average and assign value in table
al_26m=(4+0+0+4+0)/5
print(al_26m)
df.loc[426,'Alcohol consumption']=al_26m


# In[97]:


# View 37 Female data, pick 5 
df[(df['Age'] == 37) & (df['Gender'] == 'Female')]


# In[98]:


#Calculate average and assign value in table
al_37f=(1+5+1+3+0)/5
print(al_37f)
df.loc[140,'Alcohol consumption']=al_37f


# In[99]:


# View 26 Female data, pick 5 
df[(df['Age'] == 26) & (df['Gender'] == 'Female')|(df['Age'] == 27) & (df['Gender'] == 'Female')]


# In[100]:


#Calculate average and assign value in table
al_26f=(0+2+0+0+1)/5
print(al_26f)
df.loc[148,'Alcohol consumption']=al_26f


# In[101]:


# View 14 Female data, pick 5 
# View 13 Female data, pick 5 

df[(df['Age'] <= 16) & (df['Gender'] == 'Female')]


# In[102]:


#Calculate average and assign value in table
al_14f=(5+4+2+1+3)/5
print(al_14f)
df.loc[158,'Alcohol consumption']=al_14f


# In[103]:


#Calculate average and assign value in table
al_13f=(3+2+3+5+1)/5
print(al_13f)
df.loc[219,'Alcohol consumption']=al_13f


# In[104]:


# View 36 Female data, pick 5 

df[(df['Age'] == 36) & (df['Gender'] == 'Female')]


# In[105]:


#Calculate average and assign value in table
al_36f=(0+1+0+0+2)/5
print(al_36f)
df.loc[185,'Alcohol consumption']=al_36f


# In[106]:


# View 50 Female data, pick 5 

df[(df['Age'] == 50) & (df['Gender'] == 'Female')|(df['Age'] == 49) & (df['Gender'] == 'Female')]


# In[107]:


#Calculate average and assign value in table
al_50f=(0+1+0+0+2)/5
print(al_50f)
df.loc[336,'Alcohol consumption']=al_50f


# In[108]:


# View 28 Female data, pick 5 

df[(df['Age'] == 28) & (df['Gender'] == 'Female')]


# In[109]:


#Calculate average and assign value in table
al_28f=(0+5+0+0+4)/5
print(al_28f)
df.loc[440,'Alcohol consumption']=al_28f


# In[110]:


#The missing values in Alcohol consumption are filled
df[df['Alcohol consumption'].isnull()]


# In[111]:


#View missing values in Excercise Frequency
#Female: 37,31
#Male: 29,25,52,23
df[df['Exercise frequency'].isnull()] 


# In[112]:


# View 37 Female data, pick 5 

df[(df['Age'] == 37) & (df['Gender'] == 'Female')]


# In[113]:


#Calculate average and assign value in table
ex_37f=(4+3+0+1+4)/5
print(ex_37f)
df.loc[33,'Exercise frequency']=ex_37f


# In[114]:


# View 31 Female data, pick 5 

df[(df['Age'] == 31) & (df['Gender'] == 'Female')|(df['Age'] == 32) & (df['Gender'] == 'Female')]


# In[115]:


#Calculate average and assign value in table
ex_31f=(0+5+4+1+1)/5
print(ex_31f)
df.loc[262,'Exercise frequency']=ex_31f


# In[116]:


# View 29 Male data, pick 5 

df[(df['Age'] == 29) & (df['Gender'] == 'Male')|(df['Age'] == 30) & (df['Gender'] == 'Male')]


# In[117]:


#Calculate average and assign value in table
ex_29f=(3+2+0+1+2)/5
print(ex_29f)
df.loc[62,'Exercise frequency']=ex_29f


# In[118]:


# View 25 Male data, pick 5 

df[(df['Age'] == 25) & (df['Gender'] == 'Male')]


# In[119]:


#Calculate average and assign value in table
ex_25f=(3+2+3+3+3)/5
print(ex_25f)
df.loc[304,'Exercise frequency']=ex_25f


# In[120]:


# View 52 Male data, pick 5 

df[(df['Age'] == 52) & (df['Gender'] == 'Male')]


# In[121]:


#Calculate average and assign value in table
ex_52f=(3+2+1+1+3)/5
print(ex_52f)
df.loc[366,'Exercise frequency']=ex_52f


# In[122]:


# View 23 Male data, pick 5 

df[(df['Age'] == 23) & (df['Gender'] == 'Male')|(df['Age'] == 24) & (df['Gender'] == 'Male')]


# In[123]:


#Calculate average and assign value in table
ex_23f=(3+2+3+2+2)/5
print(ex_23f)
df.loc[446,'Exercise frequency']=ex_23f


# In[124]:


#The missing values in Exercise frequency are filled
df[df['Exercise frequency'].isnull()]


# In[125]:


# All missing values are filled
df.isnull().sum()


# In[126]:


i=-1
for e in df['Smoking status']:
    i+=1
    if df.loc[i,'Smoking status']=='Yes':
        df.loc[i,'Smoking status']=1
    elif df.loc[i,'Smoking status']=='No':
        df.loc[i,'Smoking status']=0


# In[143]:


df['Smoking status'] = df['Smoking status'].astype('int64')  # Convert to int64


# In[128]:


#• Group 1: children (1-12)
#• Group 2: teenagers (13-17)
#• Group 3: young adults (18-30)
#• Group 4: adults (31-60)
#• Group 5: older adults (65+)


# In[172]:


# Assign Combined Group variable
children=df[df['Age']<=12]
teenagers=df[(df['Age']>=13) & (df['Age']<=17)]
young_adults=df[(df['Age']>=18) & (df['Age']<=30)]
adults=df[(df['Age']>=31) & (df['Age']<=60)]
older_adults=df[df['Age']>=65]


# In[173]:


# Assign Female Group variable
f_children=df[(df['Age']<=12)& (df['Gender']=='Female')]
f_teenagers=df[(df['Age']>=13) & (df['Age']<=17)&(df['Gender']=='Female')]
f_young_adults=df[(df['Age']>=18) & (df['Age']<=30)&(df['Gender']=='Female')]
f_adults=df[(df['Age']>=31) &(df['Age']<=60)&(df['Gender']=='Female')]
f_older_adults=df[(df['Age']>=65) &(df['Gender']=='Female')]


# In[187]:


# Assign Male Group variable
m_children=df[(df['Age']<=12)&df['Gender']=='Male']
m_teenagers=df[(df['Age']>=13) & (df['Age']<=17)& (df['Gender']=='Male')]
m_young_adults=df[(df['Age']>=18) & (df['Age']<=30)&(df['Gender']=='Male')]
m_adults=df[(df['Age']>=31)&(df['Age']<=60)&(df['Gender']=='Male')]
m_older_adults=df[(df['Age']>=65) &(df['Gender']=='Male')]


# In[154]:


#Create Combined Mean and Standard Deviation table
data = {
    'METRIC': ['age','duration','efficiency','REM%','deep sleep%','light sleep%','#awake','smoking','exercise'],
    'Group1 (1~12)':  ['μ=10.5,σ=1.29','μ=8.38,σ=0.95','μ=0.54,σ=0.02','μ=18,σ=0','μ=35,σ=0','μ=45,σ=0','μ=2.5,σ=1.29','N/A','μ=0,σ=0'],
    'Group2 (13~17)': ['μ=15,σ=1.58','μ=7.7,σ=0.67','μ=0.63,σ=0.04','μ=18.8,σ=1.79','μ=32.6,σ=5.37','μ=46.8,σ=4.02','μ=2,σ=1','μ=0.2,σ=0.45','μ=0,σ=0'],
    'Group3 (18~30)': ['μ=25.50,σ=3.27','μ=7.50,σ=0.78','μ=0.77,σ=0.14','μ=23.67,σ=4.18','μ=52.39,σ=16.79','μ=25.98,σ=15.75','μ=1.67,σ=1.37','μ=0.39,σ=0.49','μ=1.78,σ=1.25'],
    'Group4 (31~60)': ['μ=45.20,σ=8.05','μ=7.42,σ=0.89','μ=0.81,σ=0.13','μ=22.86,σ=3.88','μ=54.20,σ=14.52','μ=23.27,σ=14.21','μ=1.57,σ=1.31','μ=0.34,σ=0.47','μ=1.92,σ=1.47'],
    'Group5 (65+)':   ['μ=65.67,σ=1.29','μ=7.43,σ=1.00','μ=0.76,σ=0.14','μ=23.07,σ=3.70','μ=50.60,σ=17.03','μ=27.27,σ=17.30','μ=2.13,σ=1.60','μ=0.47,σ=0.52','μ=1.73,σ=1.44'],
}
Combined_Table= pd.DataFrame(data)


# In[182]:


#Create Female Mean and Standard Deviation table
data = {
    'METRIC': ['age','duration','efficiency','REM%','deep sleep%','light sleep%','#awake','smoking','exercise'],
    'Group1 (1~12)':  ['μ=10.5,σ=1.29','μ=8.38,σ=0.95','μ=0.54,σ=0.02','μ=18,σ=0','μ=35,σ=0','μ=45,σ=0','μ=2.5,σ=1.29','N/A','μ=0,σ=0'],
    'Group2 (13~17)': ['μ=15,σ=1.58','μ=7.7,σ=0.67','μ=0.63,σ=0.04','μ=18.8,σ=1.79','μ=32.6,σ=5.37','μ=46.8,σ=4.02','μ=2,σ=1','μ=0.2,σ=0.45','μ=0,σ=0'],
    'Group3 (18~30)': ['μ=26.38,σ=3.26','μ=7.56,σ=0.87','μ=0.79,σ=0.14','μ=25.07,σ=3.28','μ=53.68,σ=14.55','μ=24.13,σ=14.32','μ=1.45,σ=1.33','μ=0.41,σ=0.50','μ=1.47,σ=1.33'],
    'Group4 (31~60)': ['μ=41.20,σ=7.40','μ=7.44,σ=0.87','μ=0.81,σ=0.14','μ=23.39,σ=3.92','μ=53.11,σ=15.19','μ=24.08,σ=15.28','μ=1.52,σ=1.30','μ=0.23,σ=0.43','μ=1.69,σ=1.65'],
    'Group5 (65+)':   ['μ=65,σ=0','μ=7.42,σ=1.02','μ=0.79,σ=0.17','μ=22,σ=3.58','μ=50.83,σ=19.49','μ=27,σ=19.00','μ=1.33,σ=1.63','μ=0.5,σ=0.55','μ=0.5,σ=1.22'],
}
Female_Table= pd.DataFrame(data)


# In[190]:


m_children


# In[195]:


#Create Male Mean and Standard Deviation table  
data = {
    'METRIC': ['age','duration','efficiency','REM%','deep sleep%','light sleep%','#awake','smoking','exercise'],
    'Group1 (1~12)':  ['N/A','N/A','N/A','N/A','N/A','N/A','N/A','N/A','N/A'],
    'Group2 (13~17)': ['N/A','N/A','N/A','N/A','N/A','N/A','N/A','N/A','N/A'],
    'Group3 (18~30)': ['μ=24.47,σ=3.00','μ=7.45,σ=0.66','μ=0.75,σ=0.14','μ=22.05,σ=4.54','μ=50.90,σ=19.08','μ=28.12,σ=17.13','μ=1.93,σ=1.38','μ=0.36,σ=0.48','μ=2.13,σ=1.06'],
    'Group4 (31~60)': ['μ=48.65,σ=6.95','μ=7.40,σ=0.91','μ=0.80,σ=0.12','μ=22.40,σ=3.79','μ=55.13,σ=13.89','μ=22.56,σ=13.23','μ=1.62,σ=1.32','μ=0.43,σ=0.50','μ=2.12,σ=1.27'],
    'Group5 (65+)':   ['μ=66.11,σ=1.54','μ=7.44,σ=1.04','μ=0.74,σ=0.11','μ=23.78,σ=3.80','μ=50.44,σ=16.43','μ=27.44,σ=17.26','μ=2.67,σ=1.41','μ=0.44,σ=0.53','μ=2.56,σ=0.88'],
}
Male_Table= pd.DataFrame(data)

# There is no Male Teenagers and Children in the data!!


# In[199]:


#View Combined Table data
Combined_Table


# In[200]:


#View Male Table data
Male_Table


# In[201]:


#View Female Table data
Female_Table


# In[208]:


#Question: Which group (age range and gender) sleeps the most? the least? wakes up the most? the least?
#Answer: 
#Group1 Female sleeps the most.
#Group4 Male sleeps the least.
#Group5 Male wakes up the most.
#Group5 Female wakes up the least.


# In[ ]:


#Question: which group has the max and min sleep efficiency? Deep/Light sleep percentage
#Answer: 
#Group 4 Female has max sleep efficiency, Group1 Female has min sleep efficiency.
#Group 4 Male has max deep sleep percentage, Group 2 Female has min deep sleep percentage
#Group 2 Female has max light sleep percentage, Group 4 Male has min light sleep percentage


# In[ ]:


#Question: do people sleep more or less if they exercise?
#Answer:
#Yes, For male, Group 5 Male tends to have higher exercise frequency, and as well sleep more in the whole stage of sleep time.
#For female, Group 4 Female tends to have highest exercise frequency, even though not the group sleep most ,but rank as 2nd among all female groups


# In[ ]:


#Question： do smokers sleep more or less
#Answer: Smokers in both female and male table sleep more.


# In[ ]:


#Question： examine your tables carefully and tell us what you see (2-3 paragraphs)
#Answer:
#Female tends to have more sleep in their early age and their sleep duration continuously falls down as the age increases.While Male relatively keeps sleep duration the same in each period.
#Famale has higher sleep efficiency than men in general during the lifetime.
#While Male tends to sleep less in golden age but more in the later retiring age(65+) with lower sleep efficiency.
#Female typically has less exercise than male, and they tend to extremely shorten exercise time in their retiring age(65+), while male cares more in exercise when they are getting older into 65+
# Male would have more awake time during life than woman.

#Male tends to have more deep sleep and REM sleep in 31~60 years old while their light sleep largely shortened with less awake time. 
#they also exercise more in this period, male probably has better health status during 31~60 because deep sleep helps recovering more dead body cells. 

#While at the same time, Female tends to have less deep sleep and REM sleep as the age increases, and they also exercise less as they are getting older.
#Female might easier to have less quality sleep when they are in retiring age, and we can see some correlation between exercise time and sleep time.
#but we don't know the exact reason why female has less exercise in the late age.

#Smoke seems to have little negative affect in sleep efficiency and time.

