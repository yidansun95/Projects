#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import pandas as pd
import math
import numpy as np


# In[2]:


df= pd.read_csv('SBUX.csv')
sp500= pd.read_csv('SPY.csv')


# In[3]:


df["Year"].dtype


# In[4]:


#Question1_1

#Starbucks

# make true label column and mark for it
df["True Label"]='Unchange'
df.loc[df["Return"] > 0, "True Label"] = "+"
df.loc[df["Return"] < 0, "True Label"] = "-"


# In[5]:


#Spy

# make true label column and mark for it
sp500["True Label"]='Unchange'
sp500.loc[sp500["Return"] > 0, "True Label"] = "+"
sp500.loc[sp500["Return"] < 0, "True Label"] = "-"


# In[6]:


df_first3year=df.loc[df['Year']<=2020]


# In[7]:


sp500_first3year=sp500.loc[sp500['Year']<=2020]


# In[8]:


df_first3year.count()


# In[9]:


sp500_first3year.count()


# In[10]:


df_first3year.head()


# In[11]:


#starbucks

i=-1
for e in df['Return']:
    i+=1
    if e>0:
        df.loc[i,"Decision"]="Buy"
    elif e<0:
        df.loc[i,"Decision"]='Sell'
    elif e==0:
        df.loc[i,"Decision"]='DT'
    


# In[12]:


#Spy

i=-1
for e in sp500['Return']:
    i+=1
    if e>0:
        sp500.loc[i,"Decision"]="Buy"
    elif e<0:
        sp500.loc[i,"Decision"]='Sell'
    elif e==0:
        sp500.loc[i,"Decision"]='DT'
    


# In[13]:


sp500.head()


# In[14]:


#starbucks

# count the total trading days:745 days
i=df_first3year[df_first3year["Decision"]!='DT']
i['Decision'].value_counts()


# In[15]:


#spy

# count the total trading days:753 days
i=sp500_first3year[sp500_first3year["Decision"]!='DT']
i['Decision'].value_counts()


# In[16]:


#starbucks

# count the trading days w/ "+": 403 days
t_plus=df_first3year.loc[(df_first3year["True Label"]=="+")&(df_first3year["Decision"]!="DT")]
print(t_plus['Decision'].value_counts())
print("")
print(t_plus.shape)


# In[17]:


#spy

# count the trading days w/ "+": 428 days
t_plus=sp500_first3year.loc[(sp500_first3year["True Label"]=="+")&(sp500_first3year["Decision"]!="DT")]
print(t_plus['Decision'].value_counts())
print("")
print(t_plus.shape)


# In[18]:


#starbucks

# count the trading days w/ "-": 335 days
t_minus=df_first3year.loc[(df_first3year["True Label"]=="-")&(df_first3year["Decision"]!="DT")]
print(t_minus['Decision'].value_counts())
print("")
print(t_minus.shape)


# In[19]:


#spy

# count the trading days w/ "-": 325 days
t_minus=sp500_first3year.loc[(sp500_first3year["True Label"]=="-")&(sp500_first3year["Decision"]!="DT")]
print(t_minus['Decision'].value_counts())
print("")
print(t_minus.shape)


# In[20]:


#Question1_1
#starbucks

df.head()


# In[21]:


#Question1_1
#spy

sp500.head()


# In[22]:


#Question1_2
#starbucks

#probability of you will be up next day
print(f'the starbucks "up" probability is: {(403/738)*100}%')


#probability of you will be down next day
print(f'the starbucks "down" probability is {(335/738)*100}%')


# In[23]:


#Question1_2
#spy


#probability of you will be up next day
print(f'the sp500 "up" probability is: {(428/753)*100}%')


#probability of you will be down next day
print(f'the sp500 "down" probability is {(325/753)*100}%')


# In[24]:


#starbucks
df_first3year["Decision"].value_counts()


# In[25]:


#spy
sp500_first3year["Decision"].value_counts()


# In[26]:


##Question1_3 
#starbucks

#3 consecutive ups: 119
#3 consecutive downs: 63
i=-1
count_3u=0
count_3d=0
for e in df_first3year['True Label']:
    i+=1
    if e=="-" and(df_first3year.loc[i+1,'True Label']=='-') and (df_first3year.loc[i+2,'True Label']=='-'):
            print(e,df_first3year.loc[i,'Date'])
            count_3d+=1
            print(count_3d)
    elif e=="+" and (df_first3year.loc[i+1,'True Label']=='+') and (df_first3year.loc[i+2,'True Label']=='+'):
            print(e,df_first3year.loc[i,'Date'])
            count_3u+=1
            print(count_3u)
    else:
        continue


# In[ ]:


##Question1_3 
#spy

#3 consecutive ups: 133
#3 consecutive downs: 63


i=-1
sp500_count_3u=0
sp500_count_3d=0
for e in sp500_first3year['True Label']:
    i+=1
    if e=="-" and(sp500_first3year.loc[i+1,'True Label']=='-') and (sp500_first3year.loc[i+2,'True Label']=='-'):
            print(e,sp500_first3year.loc[i,'Date'])
            sp500_count_3d+=1
            print(sp500_count_3d)
    elif e=="+" and (sp500_first3year.loc[i+1,'True Label']=='+') and (sp500_first3year.loc[i+2,'True Label']=='+'):
            print(e,sp500_first3year.loc[i,'Date'])
            sp500_count_3u+=1
            print(sp500_count_3u)
    else:
        continue


# In[27]:


#Question1_3 & #Question1_4
#Starbucks  K=3 total counts after a consecutive 3

i=-1
count_dddd=0 #this is -/-/-/-
count_dddu=0 #this is -/-/-/+
count_uuuu=0 #this is +/+/+/+
count_uuud=0 #this is +/+/+/-

for e in df_first3year['True Label']:
    i+=1
    if e=='-'and (df_first3year.loc[i+1,'True Label'])=='-'and (df_first3year.loc[i+2,'True Label']=='-')and (df_first3year.loc[i+3,'True Label']=='-'):
        print(e,df_first3year.loc[i,'Date'])
        count_dddd+=1
        print(f'this is -/-/-/-  {count_dddd}\n')  #result of 26
        
    elif e=='-'and (df_first3year.loc[i+1,'True Label'])=='-'and (df_first3year.loc[i+2,'True Label']=='-')and (df_first3year.loc[i+3,'True Label']=='+'):
        print(e,df_first3year.loc[i,'Date'])
        count_dddu+=1
        print(f'this is -/-/-/+  {count_dddu}\n')  #result of 35
    
    elif e=='+' and (df_first3year.loc[i+1,'True Label']=='+')and (df_first3year.loc[i+2,'True Label']=='+'and (df_first3year.loc[i+3,'True Label']=='+')):
        print(e,df_first3year.loc[i,'Date'])
        count_uuuu+=1
        print(f'this is +/+/+/+  {count_uuuu}\n') #result of 59
        
    elif e=='+' and (df_first3year.loc[i+1,'True Label']=='+')and (df_first3year.loc[i+2,'True Label']=='+'and (df_first3year.loc[i+3,'True Label']=='-')):
        print(e,df_first3year.loc[i,'Date'],)
        count_uuud+=1
        print(f'this is +/+/+/-  {count_uuud}\n')  #result of 57

# we can find that 43 "3 consecutive ups" and 29 of "3 consecutive downs"
# k=3, after 3consecutive down days: Probability（-/-/-/+）= 35/63        Probability（-/-/-/-）= 26/63
# k=3, after 3consecutive up days: Probability（+/+/+/+）= 59/119         Probability（+/+/+/-）= 57/119


# In[28]:


##Question1_3 
#spy K=3 total counts after a consecutive 3

i=-1
count_dddd=0 #this is -/-/-/-
count_dddu=0 #this is -/-/-/+
count_uuuu=0 #this is +/+/+/+
count_uuud=0 #this is +/+/+/-

for e in sp500_first3year['True Label']:
    i+=1
    if e=='-'and (sp500_first3year.loc[i+1,'True Label'])=='-'and (sp500_first3year.loc[i+2,'True Label']=='-')and (sp500_first3year.loc[i+3,'True Label']=='-'):
        print(e,sp500_first3year.loc[i,'Date'])
        count_dddd+=1
        print(f'this is -/-/-/-  {count_dddd}\n')  #result of 30
        
    elif e=='-'and (sp500_first3year.loc[i+1,'True Label'])=='-'and (sp500_first3year.loc[i+2,'True Label']=='-')and (sp500_first3year.loc[i+3,'True Label']=='+'):
        print(e,sp500_first3year.loc[i,'Date'])
        count_dddu+=1
        print(f'this is -/-/-/+  {count_dddu}\n')  #result of 33
    
    elif e=='+' and (sp500_first3year.loc[i+1,'True Label']=='+')and (sp500_first3year.loc[i+2,'True Label']=='+'and (sp500_first3year.loc[i+3,'True Label']=='+')):
        print(e,sp500_first3year.loc[i,'Date'])
        count_uuuu+=1
        print(f'this is +/+/+/+  {count_uuuu}\n') #result of 61
        
    elif e=='+' and (sp500_first3year.loc[i+1,'True Label']=='+')and (sp500_first3year.loc[i+2,'True Label']=='+'and (sp500_first3year.loc[i+3,'True Label']=='-')):
        print(e,sp500_first3year.loc[i,'Date'],)
        count_uuud+=1
        print(f'this is +/+/+/-  {count_uuud}\n')  #result of 72

# we can find that 133 "3 consecutive ups" and 63 of "3 consecutive downs"
# k=3, after 3consecutive down days: Probability（-/-/-/+）= 33/63        Probability（-/-/-/-）= 30/63
# k=3, after 3consecutive up days: Probability（+/+/+/+）= 61/133         Probability（+/+/+/-）= 72/133


# In[29]:


#Question1_3 & #Question1_4
#Starbucks  K=2 Count consecutive2

i=-1
count_2down=0 
count_2up=0 

for e in df_first3year['True Label']:
    i+=1
    if e=='-'and (df_first3year.loc[i+1,'True Label'])=='-':
        print(e,df_first3year.loc[i,'Date'])
        count_2down+=1 #result of 146
        print(count_2down) 
    elif e=='+' and (df_first3year.loc[i+1,'True Label']=='+'):
        print(e,df_first3year.loc[i,'Date'])
        count_2up+=1 #result of 221
        print(count_2up)
# k=2, two consecutive downs total is  146
# k=2, two consecutive ups total is  221


# In[30]:


#Question1_3 & #Question1_4
#Starbucks  K=2  Total account after a consecutive2

i=-1
count_ddd=0 #this is -/-/-
count_ddu=0 #this is -/-/+
count_uuu=0 #this is +/+/+
count_uud=0 #this is +/+/-

for e in df_first3year['True Label']:
    i+=1
    if e=='-'and (df_first3year.loc[i+1,'True Label'])=='-'and (df_first3year.loc[i+2,'True Label']=='-'):
        print(e,df_first3year.loc[i,'Date'])
        count_ddd+=1
        print(f'this is -/-/-  {count_ddd}\n')  #result of 
        
    elif e=='-'and (df_first3year.loc[i+1,'True Label'])=='-'and (df_first3year.loc[i+2,'True Label']=='+'):
        print(e,df_first3year.loc[i,'Date'])
        count_ddu+=1
        print(f'this is -/-/+  {count_ddu}\n')  #result of 
    
    elif e=='+' and (df_first3year.loc[i+1,'True Label']=='+')and (df_first3year.loc[i+2,'True Label']=='+'):
        print(e,df_first3year.loc[i,'Date'])
        count_uuu+=1
        print(f'this is +/+/+  {count_uuu}\n') #result of 
        
    elif e=='+' and (df_first3year.loc[i+1,'True Label']=='+') and (df_first3year.loc[i+2,'True Label']=='-'):
        print(e,df_first3year.loc[i,'Date'],)
        count_uud+=1
        print(f'this is +/+/-  {count_uud}\n')  #result of 

# k=2, after 2consecutive down days: Probability（-/-/+）=    81/146      Probability（-/-/-）= 63/146
# k=2, after 2consecutive up days: Probability（+/+/+）=     119/221      Probability（+/+/-）= 99/221


# In[31]:


#Question1_3 & #Question1_4
#Spy  K=2 Count consecutive2

i=-1
sp500_count_2down=0 
sp500_count_2up=0 

for e in sp500_first3year['True Label']:
    i+=1
    if e=='-'and (sp500_first3year.loc[i+1,'True Label'])=='-':
        print(e,sp500_first3year.loc[i,'Date'])
        sp500_count_2down+=1 #result of 146
        print(sp500_count_2down) 
    elif e=='+' and (sp500_first3year.loc[i+1,'True Label']=='+'):
        print(e,sp500_first3year.loc[i,'Date'])
        sp500_count_2up+=1 #result of 221
        print(sp500_count_2up)
# k=2, two consecutive downs total is  137
# k=2, two consecutive ups total is  239


# In[32]:


#Question1_3 & #Question1_4
#Spy K=2  Total account after a consecutive2

i=-1
sp500_count_ddd=0 #this is -/-/-
sp500_count_ddu=0 #this is -/-/+
sp500_count_uuu=0 #this is +/+/+
sp500_count_uud=0 #this is +/+/-

for e in sp500_first3year['True Label']:
    i+=1
    if e=='-'and (sp500_first3year.loc[i+1,'True Label'])=='-'and (sp500_first3year.loc[i+2,'True Label']=='-'):
        print(e,sp500_first3year.loc[i,'Date'])
        sp500_count_ddd+=1
        print(f'this is -/-/-  {sp500_count_ddd}\n')  #result of 63
        
    elif e=='-'and (sp500_first3year.loc[i+1,'True Label'])=='-'and (sp500_first3year.loc[i+2,'True Label']=='+'):
        print(e,sp500_first3year.loc[i,'Date'])
        sp500_count_ddu+=1
        print(f'this is -/-/+  {sp500_count_ddu}\n')  #result of 73
    
    elif e=='+' and (sp500_first3year.loc[i+1,'True Label']=='+')and (sp500_first3year.loc[i+2,'True Label']=='+'):
        print(e,sp500_first3year.loc[i,'Date'])
        sp500_count_uuu+=1
        print(f'this is +/+/+  {sp500_count_uuu}\n') #result of 133
        
    elif e=='+' and (sp500_first3year.loc[i+1,'True Label']=='+') and (sp500_first3year.loc[i+2,'True Label']=='-'):
        print(e,sp500_first3year.loc[i,'Date'],)
        sp500_count_uud+=1
        print(f'this is +/+/-  {sp500_count_uud}\n')  #result of 104

# k=2, after 2consecutive down days: Probability（-/-/+）=    73/137      Probability（-/-/-）= 63/137
# k=2, after 2consecutive up days: Probability（+/+/+）=     133/239      Probability（+/+/-）= 104/239


# In[33]:


#Question1_3 & #Question1_4
#Starbucks  K=1 Count consecutive1

i=-1
count_1down=0 
count_1up=0 

for e in df_first3year['True Label']:
    i+=1
    if e=='-':
        print(e,df_first3year.loc[i,'Date'])
        count_1down+=1 #result of 336
        print(count_1down) 
    elif e=='+':
        print(e,df_first3year.loc[i,'Date'])
        count_1up+=1 #result of 412
        print(count_1up)
        
# k=1, 1 down total is  336
# k=1, 1 up total is  412


# In[34]:


#Question1_3 & #Question1_4
#Starbucks  K=1  #Total account after a consecutive1


i=-1
count_dd=0 #this is -/-
count_du=0 #this is -/+
count_uu=0 #this is +/+
count_ud=0 #this is +/-

for e in df_first3year['True Label']:
    i+=1
    if e=='-'and (df_first3year.loc[i+1,'True Label'])=='-':
        print(e,df_first3year.loc[i,'Date'])
        count_dd+=1
        print(f'this is -/-  {count_dd}\n')  #result of 
        
    elif e=='-'and (df_first3year.loc[i+1,'True Label']=='+'):
        print(e,df_first3year.loc[i,'Date'])
        count_du+=1
        print(f'this is -/+  {count_du}\n')  #result of 
    
    elif e=='+' and (df_first3year.loc[i+1,'True Label']=='+'):
        print(e,df_first3year.loc[i,'Date'])
        count_uu+=1
        print(f'this is +/+  {count_uu}\n') #result of 
        
    elif e=='+'and (df_first3year.loc[i+1,'True Label']=='-'):
        print(e,df_first3year.loc[i,'Date'],)
        count_ud+=1
        print(f'this is +/-  {count_ud}\n')  #result of 

# k=1, after 1 down day: Probability（-/+）=    186/336      Probability（-/-）= 146/336
# k=1, after 1 up day: Probability（+/+）=     221/412      Probability（+/-）= 187/412


# In[35]:


#Question1_3 & #Question1_4
#Spy  K=1 Count consecutive1

i=-1
sp500_count_1down=0 
sp500_count_1up=0 

for e in sp500_first3year['True Label']:
    i+=1
    if e=='-':
        print(e,sp500_first3year.loc[i,'Date'])
        sp500_count_1down+=1 #result of 325
        print(sp500_count_1down) 
    elif e=='+':
        print(e,sp500_first3year.loc[i,'Date'])
        sp500_count_1up+=1 #result of 428
        print(sp500_count_1up)
        
# k=1, 1 down total is 325 
# k=1, 1 up total is  428


# In[36]:


#Question1_3 & #Question1_4
#Spy  K=1  #Total account after a consecutive1


i=-1
sp500_count_dd=0 #this is -/-
sp500_count_du=0 #this is -/+
sp500_count_uu=0 #this is +/+
sp500_count_ud=0 #this is +/-

for e in sp500_first3year['True Label']:
    i+=1
    if e=='-'and (sp500_first3year.loc[i+1,'True Label'])=='-':
        print(e,sp500_first3year.loc[i,'Date'])
        sp500_count_dd+=1
        print(f'this is -/-  {sp500_count_dd}\n')  #result of 137
        
    elif e=='-'and (sp500_first3year.loc[i+1,'True Label']=='+'):
        print(e,sp500_first3year.loc[i,'Date'])
        sp500_count_du+=1
        print(f'this is -/+  {sp500_count_du}\n')  #result of 187
    
    elif e=='+' and (sp500_first3year.loc[i+1,'True Label']=='+'):
        print(e,sp500_first3year.loc[i,'Date'])
        sp500_count_uu+=1
        print(f'this is +/+  {sp500_count_uu}\n') #result of 239
        
    elif e=='+'and (sp500_first3year.loc[i+1,'True Label']=='-'):
        print(e,sp500_first3year.loc[i,'Date'],)
        sp500_count_ud+=1
        print(f'this is +/-  {sp500_count_ud}\n')  #result of 187

# k=1, after 1 down day: Probability（-/+）=    187/325      Probability（-/-）= 137/325
# k=1, after 1 up day: Probability（+/+）=     239/428      Probability（+/-）= 187/428


# In[37]:


df["Predict Label_w=2"]="Unchange"


# In[38]:


sp500["Predict Label_w=2"]="Unchange"


# In[39]:


#Question 2_1
# calculate other probabilities total
c1=0
c2=0
c3=0
c4=0
c5=0
c6=0

i=-1
for e in df_first3year['True Label']:
    i+=1
    if e=="+"and(df_first3year.loc[i+1,'True Label']=='+')and(df_first3year.loc[i+2,'True Label']=='-'):
        print("+/+/-: ")
        print(df_first3year.loc[i,'Date'])
        c1+=1
        print(f'{c1}\n')
    elif e=="+"and (df_first3year.loc[i+1,'True Label']=='-')and(df_first3year.loc[i+2,'True Label']=='+'):
        print("+/-/+: ")
        print(df_first3year.loc[i,'Date'])
        c2+=1
        print(f'{c2}\n')
            
    elif e=="+" and (df_first3year.loc[i+1,'True Label']=='-')and(df_first3year.loc[i+2,'True Label']=='-'):
        print("+/-/-: ")
        print(df_first3year.loc[i,'Date'])
        c3+=1
        print(f'{c3}\n')
    elif e=="-" and (df_first3year.loc[i+1,'True Label']=='+')and(df_first3year.loc[i+2,'True Label']=='+'):
        print("-/+/+: ")
        print(df_first3year.loc[i,'Date'])
        c4+=1
        print(f'{c4}\n')
    elif e=="-" and (df_first3year.loc[i+1,'True Label']=='+')and(df_first3year.loc[i+2,'True Label']=='-'):
        print("-/+/-: ")
        print(df_first3year.loc[i,'Date'])
        c5+=1
        print(f'{c5}\n')
    elif e=="-" and (df_first3year.loc[i+1,'True Label']=='-')and(df_first3year.loc[i+2,'True Label']=='+'):
        print("-/-/+: ")
        print(df_first3year.loc[i,'Date'])
        c6+=1
        print(f'{c6}\n')
    else:
        continue

#  +/+/- total:99
# +/-/+ total:103
# +/-/- total:82
# -/+/+ total:99
# -/+/- total:86
# -/-/+ total:81

#we already know: +/+/+ is 119       -/-/- is 63


# In[40]:


#Question 2_1
#Starbucks W=3

i=-1

#  +/+/- total:99
count_uudd=0 #this is +/+/-/-
count_uudu=0 #this is +/+/-/+

# +/-/+ total:103
count_uduu=0 #this is +/-/+/+
count_udud=0 #this is +/-/+/-

# +/-/- total:82
count_uddu=0 #this is +/-/-/+
count_uddd=0 #this is +/-/-/-

# -/+/+ total:99
count_duuu=0 #this is -/+/+/+
count_duud=0 #this is -/+/+/-

#-/+/- total:86
count_dudu=0 #this is -/+/-/+
count_dudd=0 #this is -/+/-/-

#-/-/+ total:81
count_dduu=0 #this is -/-/+/+
count_ddud=0 #this is -/-/+/-

count_uuud=0
count_dddu=0



for e in df_first3year['True Label']:
    i+=1
    #Compare +/+/-/- with +/+/-/+
    if e=='+'and (df_first3year.loc[i+1,'True Label'])=='+'and (df_first3year.loc[i+2,'True Label']=='-')and (df_first3year.loc[i+3,'True Label']=='-'):
        print(e,df_first3year.loc[i,'Date'])
        count_uudd+=1
        print(f'this is +/+/-/-  {count_uudd}\n')  
        
    if e=='+'and (df_first3year.loc[i+1,'True Label'])=='+'and (df_first3year.loc[i+2,'True Label']=='-')and (df_first3year.loc[i+3,'True Label']=='+'):
        print(e,df_first3year.loc[i,'Date'])
        count_uudu+=1
        print(f'this is +/+/-/+  {count_uudu}\n')   
        
        
    #Compare +/-/+/+ with +/-/+/-  
    elif e=='+'and (df_first3year.loc[i+1,'True Label'])=='-'and (df_first3year.loc[i+2,'True Label']=='+')and (df_first3year.loc[i+3,'True Label']=='+'):
        print(e,df_first3year.loc[i,'Date'])
        count_uduu+=1
        print(f'this is +/-/+/+  {count_uduu}\n') 
    
    elif e=='+' and (df_first3year.loc[i+1,'True Label']=='-')and (df_first3year.loc[i+2,'True Label']=='+'and (df_first3year.loc[i+3,'True Label']=='-')):
        print(e,df_first3year.loc[i,'Date'])
        count_udud+=1
        print(f'this is +/-/+/-  {count_udud}\n') 
        
        
    #Compare +/-/-/+ with +/-/-/-
    elif e=='+' and (df_first3year.loc[i+1,'True Label']=='-')and (df_first3year.loc[i+2,'True Label']=='-'and (df_first3year.loc[i+3,'True Label']=='+')):
        print(e,df_first3year.loc[i,'Date'],)
        count_uddu+=1
        print(f'this is +/-/-/+  {count_uddu}\n') 
        
    elif e=='+' and (df_first3year.loc[i+1,'True Label']=='-')and (df_first3year.loc[i+2,'True Label']=='-'and (df_first3year.loc[i+3,'True Label']=='-')):
        print(e,df_first3year.loc[i,'Date'],)
        count_uddd+=1
        print(f'this is +/-/-/-  {count_uddd}\n')       
        
        
    #Compare -/+/+/+ with -/+/+/-
    elif e=='-' and (df_first3year.loc[i+1,'True Label']=='+')and (df_first3year.loc[i+2,'True Label']=='+'and (df_first3year.loc[i+3,'True Label']=='+')):
        print(e,df_first3year.loc[i,'Date'],)
        count_duuu+=1
        print(f'this is -/+/+/+  {count_duuu}\n')     
        
    elif e=='-' and (df_first3year.loc[i+1,'True Label']=='+')and (df_first3year.loc[i+2,'True Label']=='+'and (df_first3year.loc[i+3,'True Label']=='-')):
        print(e,df_first3year.loc[i,'Date'],)
        count_duud+=1
        print(f'this is -/+/+/-  {count_duud}\n') 

        
    #Compare -/+/-/+  with  -/+/-/- 
    elif e=='-' and (df_first3year.loc[i+1,'True Label']=='+')and (df_first3year.loc[i+2,'True Label']=='-'and (df_first3year.loc[i+3,'True Label']=='+')):
        print(e,df_first3year.loc[i,'Date'],)
        count_dudu+=1
        print(f'this is -/+/-/+  {count_dudu}\n')     

    elif e=='-' and (df_first3year.loc[i+1,'True Label']=='+')and (df_first3year.loc[i+2,'True Label']=='-'and (df_first3year.loc[i+3,'True Label']=='-')):
        print(e,df_first3year.loc[i,'Date'],)
        count_dudd+=1
        print(f'this is -/+/-/-  {count_dudd}\n')     
        
    #Compare -/-/+/+  with -/-/+/-
    elif e=='-' and (df_first3year.loc[i+1,'True Label']=='-')and (df_first3year.loc[i+2,'True Label']=='+'and (df_first3year.loc[i+3,'True Label']=='+')):
        print(e,df_first3year.loc[i,'Date'],)
        count_dduu+=1
        print(f'this is -/-/+/+  {count_dduu}\n')     
        
    elif e=='-' and (df_first3year.loc[i+1,'True Label']=='-')and (df_first3year.loc[i+2,'True Label']=='+'and (df_first3year.loc[i+3,'True Label']=='-')):
        print(e,df_first3year.loc[i,'Date'],)
        count_ddud+=1
        print(f'this is -/-/+/-  {count_ddud}\n')  
        
        
    #Compare +/+/+/-  with -/-/-/+   
    if e=='+'and (df_first3year.loc[i+1,'True Label'])=='+'and (df_first3year.loc[i+2,'True Label']=='+')and (df_first3year.loc[i+3,'True Label']=='-'):
        print(e,df_first3year.loc[i,'Date'])
        count_uuud+=1
        print(f'this is +/+/+/-  {count_uuud}\n')  
        
    if e=='-'and (df_first3year.loc[i+1,'True Label'])=='-'and (df_first3year.loc[i+2,'True Label']=='-')and (df_first3year.loc[i+3,'True Label']=='+'):
        print(e,df_first3year.loc[i,'Date'])
        count_dddu+=1
        print(f'this is -/-/-/+  {count_dddu}\n')  
    

#Use training data to analyze next one probability 
#P() represents probability

#########P(+/+/-/+)wins!!!!!!###########################
#P(+/+/-/-)= 48/99
#P(+/+/-/+)= 50/99   


########P(+/-/+/+)wins!!!!!!############################
#P(+/-/+/+)= 57/103
#P(+/-/+/-)= 46/103


########P(+/-/-/+) wins!!!!!!############################
#P(+/-/-/+)= 45/82
#P(+/-/-/-)= 37/82


#########P(-/+/+/+)wins!!!!!!############################
#P(-/+/+/+)= 57/99
#P(-/+/+/-)= 42/99


#######P(-/+/-/+)wins!!!!!################################
#P(-/+/-/+)= 52/86
#P(-/+/-/-)= 33/86


########P(-/-/+/+)wins!!!!!###############################
#P(-/-/+/+)= 41/81
#P(-/-/+/-)= 39/81


########P(+/+/+/+)wins!!!!!###############################
#P(+/+/+/-)=  57/116
#P(+/+/+/+)=  59/116

########P(-/-/-/+)wins!!!!!###############################
#P(-/-/-/-)=26/61
#P(-/-/-/+)= 35/61


# In[41]:


df["Predict Label_w=3"]='Unchange'


# In[42]:


#Question 2_1
#Starbucks W=3

#assign value to the predict label when w=3
i=753
while i>=753 and i <=1259:
    if df.loc[i,"True Label"]=="+" and df.loc[i+1,"True Label"]=="+" and df.loc[i+2,"True Label"]=="+":
        df.loc[i+3,"Predict Label_w=3"]="+"
    elif df.loc[i,"True Label"]=="+" and df.loc[i+1,"True Label"]=="+" and df.loc[i+2,"True Label"]=="-":
        df.loc[i+3,"Predict Label_w=3"]="+"
    elif df.loc[i,"True Label"]=="+" and df.loc[i+1,"True Label"]=="-" and df.loc[i+2,"True Label"]=="+":
        df.loc[i+3,"Predict Label_w=3"]="+"
    elif df.loc[i,"True Label"]=="+" and df.loc[i+1,"True Label"]=="-" and df.loc[i+2,"True Label"]=="-":
        df.loc[i+3,"Predict Label_w=3"]="+"
    elif df.loc[i,"True Label"]=="-" and df.loc[i+1,"True Label"]=="+" and df.loc[i+2,"True Label"]=="+":
        df.loc[i+3,"Predict Label_w=3"]="+"
    elif df.loc[i,"True Label"]=="-" and df.loc[i+1,"True Label"]=="+" and df.loc[i+2,"True Label"]=="-":
        df.loc[i+3,"Predict Label_w=3"]="+"
    elif df.loc[i,"True Label"]=="-" and df.loc[i+1,"True Label"]=="-" and df.loc[i+2,"True Label"]=="+":
        df.loc[i+3,"Predict Label_w=3"]="+"
    elif df.loc[i,"True Label"]=="-" and df.loc[i+1,"True Label"]=="-" and df.loc[i+2,"True Label"]=="-":
        df.loc[i+3,"Predict Label_w=3"]="+"
    i+=1


# In[43]:


df[df['Year']==2022]


# In[44]:


#Question 2_1
#Starbucks W=2

i=-1

#  +/+ total:

#  +/- total:187
count_ud=0 #this is +/-

# -/+ total:186
count_du=0 #this is -/+

# -/- total:

for e in df_first3year['True Label']:
    i+=1
    
    if e=='+'and (df_first3year.loc[i+1,'True Label'])=='-':
        print(e,df_first3year.loc[i,'Date'])
        count_ud+=1
        print(f'this is +/-  {count_ud}\n')  
        
    if e=='-'and (df_first3year.loc[i+1,'True Label'])=='+':
        print(e,df_first3year.loc[i,'Date'])
        count_du+=1
        print(f'this is -/+  {count_du}\n')  
        


# In[45]:


#Question 2_1

#Starbucks W=2

i=-1

count_udu=0
count_udd=0
count_duu=0
count_dud=0
#count_uuu= 119
#count_ddd= 63

for e in df_first3year['True Label']:
    i+=1
    #Compare +/-/+ with +/-/-
    if e=='+'and (df_first3year.loc[i+1,'True Label'])=='-'and (df_first3year.loc[i+2,'True Label']=='+'):
        print(e,df_first3year.loc[i,'Date'])
        count_udu+=1
        print(f'this is +/-/+  {count_udu}\n')  #result of 
        
    if e=='+'and (df_first3year.loc[i+1,'True Label'])=='-'and (df_first3year.loc[i+2,'True Label']=='-'):
        print(e,df_first3year.loc[i,'Date'])
        count_udd+=1
        print(f'this is +/-/-  {count_udd}\n')  #result of 
        
        
    #Compare -/+/+ with -/+/-  
    elif e=='-'and (df_first3year.loc[i+1,'True Label'])=='+'and (df_first3year.loc[i+2,'True Label']=='+'):
        print(e,df_first3year.loc[i,'Date'])
        count_duu+=1
        print(f'this is -/+/+  {count_duu}\n')  #result of 
    
    elif e=='-' and (df_first3year.loc[i+1,'True Label']=='+')and (df_first3year.loc[i+3,'True Label']=='-'):
        print(e,df_first3year.loc[i,'Date'])
        count_dud+=1
        print(f'this is -/+/-  {count_dud}\n') #result of 
        

#k=3
#P() represents probability


#（-/-/+）wins; (+/+/+)wins
#P（-/-/+）=    81/146      Probability（-/-/-）= 63/146
#P（+/+/+）=    119/221      Probability（+/+/-）= 99/221

#########(+/-/+)wins!!!!!!###########################
#P(+/-/+)= 103/187
#P(+/-/-)= 82/187


########(-/+/+)wins!!!!!!############################
#P(-/+/+)= 99/186
#P(-/+/-)= 33/186



# In[46]:


#Question 2_1

#Starbucks W=2

#assign value to the predict label when w=2
i=753
while i>=753 and i <=1255:
    i+=1
    if df.loc[i,"True Label"]=="+" and df.loc[i+1,"True Label"]=="+" :
        df.loc[i+2,"Predict Label_w=2"]="+"
    elif df.loc[i,"True Label"]=="+" and df.loc[i+1,"True Label"]=="-" :
        df.loc[i+2,"Predict Label_w=2"]="+"
    elif df.loc[i,"True Label"]=="-" and df.loc[i+1,"True Label"]=="+" :
        df.loc[i+2,"Predict Label_w=2"]="+"
    elif df.loc[i,"True Label"]=="-" and df.loc[i+1,"True Label"]=="-" :
        df.loc[i+2,"Predict Label_w=2"]="+"


# In[47]:


df[df['Year']==2022]


# In[48]:


#Question 2_1

#Starbucks W=4, it is pretty obviously that the next one will be "+"



#  +/+/+/+/ total:59
count_uuuud=0 #this is +/+/+/+/-
count_uuuuu=0 #this is +/+/+/+/+


for e in df_first3year['True Label']:
    i+=1
    #Compare +/+/+/+/+ with +/+/+/+/-
    if e=='+'and (df_first3year.loc[i+1,'True Label'])=='+'and (df_first3year.loc[i+2,'True Label']=='+')and (df_first3year.loc[i+3,'True Label']=='+')and (df_first3year.loc[i+3,'True Label']=='+'):
        print(e,df_first3year.loc[i,'Date'])
        count_uuuuu+=1
        print(f'this is +/+/+/+/+  {count_uuuuu}\n')  #result of 59
        
    if e=='+'and (df_first3year.loc[i+1,'True Label'])=='+'and (df_first3year.loc[i+2,'True Label']=='+')and (df_first3year.loc[i+3,'True Label']=='+')and (df_first3year.loc[i+3,'True Label']=='-'):
        print(e,df_first3year.loc[i,'Date'])
        count_uuuud+=1
        print(f'this is +/+/+/+/-  {count_uuuud}\n')  #result of 0

#probability for +/+/+/+/- is 0 ; so definitely go for +/+/+/+/+


# In[49]:


#Question 2_1

#Starbucks W=4

#assign value to the predict label when w=4
df['Predict Label_w=4']="+"
i=756
while i>=756 and i <=1259:
    df.loc[i,"Predict Label_w=4"]="+"
    i+=1


# In[50]:


##Question 2_2 & Question 2_3

#Starbucks W=2,3,4

#My w=2,3,4 goes the same, there is no highest probability. the next one always goes with "+" as we have analyzed

#total "+" in w: 500
#count accuracy for w
i=-1
accuracy_w3=0
wrong_w3=0
accuracy_w3_u=0
accuracy_w3_d=0
while i>=-1 and i<=1257:
    i+=1
    if df.loc[i, 'True Label']==df.loc[i,'Predict Label_w=3']:
        accuracy_w3+=1
        if df.loc[i, 'True Label']=="+":
            accuracy_w3_u+=1
        elif df.loc[i, 'True Label']=="-":
            accuracy_w3_d=0
            
    else:
        wrong_w3+=1
print(f'Right predict Probability= {accuracy_w3/(accuracy_w3+wrong_w3)*100}%\nWrong predict Probability= {wrong_w3/(accuracy_w3+wrong_w3)*100}%')
print(f'\nRight predict Probability of "+"is: {(accuracy_w3_u/500)*100}%' )
print('\nRight predict Probability of "-"is: 0')
print(f'accuracy is : {(accuracy_w3_u/500)*100}'  )


# In[51]:


#Question 2_1
#Spy

# calculate other probabilities total
c1=0
c2=0
c3=0
c4=0
c5=0
c6=0

i=-1
for e in sp500_first3year['True Label']:
    i+=1
    if e=="+"and(sp500_first3year.loc[i+1,'True Label']=='+')and(sp500_first3year.loc[i+2,'True Label']=='-'):
        print("+/+/-: ")
        print(sp500_first3year.loc[i,'Date'])
        c1+=1
        print(f'{c1}\n')
    elif e=="+"and (sp500_first3year.loc[i+1,'True Label']=='-')and(sp500_first3year.loc[i+2,'True Label']=='+'):
        print("+/-/+: ")
        print(sp500_first3year.loc[i,'Date'])
        c2+=1
        print(f'{c2}\n')
            
    elif e=="+" and (sp500_first3year.loc[i+1,'True Label']=='-')and(sp500_first3year.loc[i+2,'True Label']=='-'):
        print("+/-/-: ")
        print(sp500_first3year.loc[i,'Date'])
        c3+=1
        print(f'{c3}\n')
    elif e=="-" and (sp500_first3year.loc[i+1,'True Label']=='+')and(sp500_first3year.loc[i+2,'True Label']=='+'):
        print("-/+/+: ")
        print(sp500_first3year.loc[i,'Date'])
        c4+=1
        print(f'{c4}\n')
    elif e=="-" and (sp500_first3year.loc[i+1,'True Label']=='+')and(sp500_first3year.loc[i+2,'True Label']=='-'):
        print("-/+/-: ")
        print(sp500_first3year.loc[i,'Date'])
        c5+=1
        print(f'{c5}\n')
    elif e=="-" and (sp500_first3year.loc[i+1,'True Label']=='-')and(sp500_first3year.loc[i+2,'True Label']=='+'):
        print("-/-/+: ")
        print(sp500_first3year.loc[i,'Date'])
        c6+=1
        print(f'{c6}\n')
    else:
        continue
        
# -/+/+ : 104
# +/-/+ : 114
# +/+/- : 104
# -/-/+ : 73
# +/-/- : 73
# -/+/- : 83


# In[91]:


#Question 2_1
#Spy W=3

# -/+/+ : 104
# +/-/+ : 114
# +/+/- : 104
# -/-/+ : 73
# +/-/- : 73
# -/+/- : 83



#  +/+/- total:99
count_uudd=0 #this is +/+/-/-
count_uudu=0 #this is +/+/-/+

# +/-/+ total:103
count_uduu=0 #this is +/-/+/+
count_udud=0 #this is +/-/+/-

# +/-/- total:82
count_uddu=0 #this is +/-/-/+
count_uddd=0 #this is +/-/-/-

# -/+/+ total:99
count_duuu=0 #this is -/+/+/+
count_duud=0 #this is -/+/+/-

#-/+/- total:86
count_dudu=0 #this is -/+/-/+
count_dudd=0 #this is -/+/-/-

#-/-/+ total:81
count_dduu=0 #this is -/-/+/+
count_ddud=0 #this is -/-/+/-

count_uuud=0
count_dddu=0


i=-1
for e in sp500_first3year['True Label']:
    i+=1
    #Compare +/+/-/- with +/+/-/+
    if e=='+'and (sp500_first3year.loc[i+1,'True Label'])=='+'and (sp500_first3year.loc[i+2,'True Label']=='-')and (sp500_first3year.loc[i+3,'True Label']=='-'):
        print(e,sp500_first3year.loc[i,'Date'])
        count_uudd+=1
        print(f'this is +/+/-/-  {count_uudd}\n')  
        
    if e=='+'and (sp500_first3year.loc[i+1,'True Label'])=='+'and (sp500_first3year.loc[i+2,'True Label']=='-')and (sp500_first3year.loc[i+3,'True Label']=='+'):
        print(e,sp500_first3year.loc[i,'Date'])
        count_uudu+=1
        print(f'this is +/+/-/+  {count_uudu}\n')   
        
        
    #Compare +/-/+/+ with +/-/+/-  
    elif e=='+'and (sp500_first3year.loc[i+1,'True Label'])=='-'and (sp500_first3year.loc[i+2,'True Label']=='+')and (sp500_first3year.loc[i+3,'True Label']=='+'):
        print(e,sp500_first3year.loc[i,'Date'])
        count_uduu+=1
        print(f'this is +/-/+/+  {count_uduu}\n') 
    
    elif e=='+' and (sp500_first3year.loc[i+1,'True Label']=='-')and (sp500_first3year.loc[i+2,'True Label']=='+'and (sp500_first3year.loc[i+3,'True Label']=='-')):
        print(e,sp500_first3year.loc[i,'Date'])
        count_udud+=1
        print(f'this is +/-/+/-  {count_udud}\n') 
        
        
    #Compare +/-/-/+ with +/-/-/-
    elif e=='+' and (sp500_first3year.loc[i+1,'True Label']=='-')and (sp500_first3year.loc[i+2,'True Label']=='-'and (sp500_first3year.loc[i+3,'True Label']=='+')):
        print(e,sp500_first3year.loc[i,'Date'],)
        count_uddu+=1
        print(f'this is +/-/-/+  {count_uddu}\n') 
        
    elif e=='+' and (sp500_first3year.loc[i+1,'True Label']=='-')and (sp500_first3year.loc[i+2,'True Label']=='-'and (sp500_first3year.loc[i+3,'True Label']=='-')):
        print(e,sp500_first3year.loc[i,'Date'],)
        count_uddd+=1
        print(f'this is +/-/-/-  {count_uddd}\n')       
        
        
    #Compare -/+/+/+ with -/+/+/-
    elif e=='-' and (sp500_first3year.loc[i+1,'True Label']=='+')and (sp500_first3year.loc[i+2,'True Label']=='+'and (sp500_first3year.loc[i+3,'True Label']=='+')):
        print(e,sp500_first3year.loc[i,'Date'],)
        count_duuu+=1
        print(f'this is -/+/+/+  {count_duuu}\n')     
        
    elif e=='-' and (sp500_first3year.loc[i+1,'True Label']=='+')and (sp500_first3year.loc[i+2,'True Label']=='+'and (sp500_first3year.loc[i+3,'True Label']=='-')):
        print(e,sp500_first3year.loc[i,'Date'],)
        count_duud+=1
        print(f'this is -/+/+/-  {count_duud}\n') 

        
    #Compare -/+/-/+  with  -/+/-/- 
    elif e=='-' and (sp500_first3year.loc[i+1,'True Label']=='+')and (sp500_first3year.loc[i+2,'True Label']=='-'and (sp500_first3year.loc[i+3,'True Label']=='+')):
        print(e,sp500_first3year.loc[i,'Date'],)
        count_dudu+=1
        print(f'this is -/+/-/+  {count_dudu}\n')     

    elif e=='-' and (sp500_first3year.loc[i+1,'True Label']=='+')and (sp500_first3year.loc[i+2,'True Label']=='-'and (sp500_first3year.loc[i+3,'True Label']=='-')):
        print(e,sp500_first3year.loc[i,'Date'],)
        count_dudd+=1
        print(f'this is -/+/-/-  {count_dudd}\n')     
        
    #Compare -/-/+/+  with -/-/+/-
    elif e=='-' and (sp500_first3year.loc[i+1,'True Label']=='-')and (sp500_first3year.loc[i+2,'True Label']=='+'and (sp500_first3year.loc[i+3,'True Label']=='+')):
        print(e,sp500_first3year.loc[i,'Date'],)
        count_dduu+=1
        print(f'this is -/-/+/+  {count_dduu}\n')     
        
    elif e=='-' and (sp500_first3year.loc[i+1,'True Label']=='-')and (sp500_first3year.loc[i+2,'True Label']=='+'and (sp500_first3year.loc[i+3,'True Label']=='-')):
        print(e,sp500_first3year.loc[i,'Date'],)
        count_ddud+=1
        print(f'this is -/-/+/-  {count_ddud}\n')  
        
        
    #Compare +/+/+/-  with -/-/-/+   
    if e=='+'and (sp500_first3year.loc[i+1,'True Label'])=='+'and (sp500_first3year.loc[i+2,'True Label']=='+')and (sp500_first3year.loc[i+3,'True Label']=='-'):
        print(e,sp500_first3year.loc[i,'Date'])
        count_uuud+=1
        print(f'this is +/+/+/-  {count_uuud}\n')  
        
    if e=='-'and (sp500_first3year.loc[i+1,'True Label'])=='-'and (sp500_first3year.loc[i+2,'True Label']=='-')and (sp500_first3year.loc[i+3,'True Label']=='+'):
        print(e,sp500_first3year.loc[i,'Date'])
        count_dddu+=1
        print(f'this is -/-/-/+  {count_dddu}\n')  
        
        

#Use training data to analyze next one probability 
#P() represents probability

#########P(+/+/-/+)wins!!!!!!###########################
#P(+/+/-/-)= 45/104
#P(+/+/-/+)= 59/104   


########P(+/-/+/+)wins!!!!!!############################
#P(+/-/+/+)= 69/114
#P(+/-/+/-)= 45/114


########P(+/-/-/+) wins!!!!!!############################
#P(+/-/-/+)= 40/73
#P(+/-/-/-)= 32/73


#########P(-/+/+/+)wins!!!!!!############################
#P(-/+/+/+)= 70/104
#P(-/+/+/-)= 32/104


#######P(-/+/-/+)wins!!!!!################################
#P(-/+/-/+)= 55/83
#P(-/+/-/-)= 28/83


########P(-/-/+/-)wins!!!!!###############################
#P(-/-/+/+)= 35/73
#P(-/-/+/-)= 38/73


########P(+/+/+/-)wins!!!!!###############################
#P(+/+/+/-)= 72/133
#P(+/+/+/+)= 61/133

########P(-/-/-/+)wins!!!!!###############################
#P(-/-/-/-)= 30/63
#P(-/-/-/+)= 33/63


# In[92]:


sp500["Predict Label_w=3"]='Unchange'


# In[102]:


sp500.tail()


# In[94]:


#Question 2_1
#Spy W=3

#assign value to the predict label when w=3
i=753
while i>=753 and i <=1255:
    if sp500.loc[i,"True Label"]=="+" and sp500.loc[i+1,"True Label"]=="+" and sp500.loc[i+2,"True Label"]=="+":
        sp500.loc[i+3,"Predict Label_w=3"]="-"
    elif sp500.loc[i,"True Label"]=="+" and sp500.loc[i+1,"True Label"]=="+" and sp500.loc[i+2,"True Label"]=="-":
        sp500.loc[i+3,"Predict Label_w=3"]="+"
    elif sp500.loc[i,"True Label"]=="+" and sp500.loc[i+1,"True Label"]=="-" and sp500.loc[i+2,"True Label"]=="+":
        sp500.loc[i+3,"Predict Label_w=3"]="+"
    elif sp500.loc[i,"True Label"]=="+" and sp500.loc[i+1,"True Label"]=="-" and sp500.loc[i+2,"True Label"]=="-":
        sp500.loc[i+3,"Predict Label_w=3"]="+"
    elif sp500.loc[i,"True Label"]=="-" and sp500.loc[i+1,"True Label"]=="+" and sp500.loc[i+2,"True Label"]=="+":
        sp500.loc[i+3,"Predict Label_w=3"]="+"
    elif sp500.loc[i,"True Label"]=="-" and sp500.loc[i+1,"True Label"]=="+" and sp500.loc[i+2,"True Label"]=="-":
        sp500.loc[i+3,"Predict Label_w=3"]="+"
    elif sp500.loc[i,"True Label"]=="-" and sp500.loc[i+1,"True Label"]=="-" and sp500.loc[i+2,"True Label"]=="+":
        sp500.loc[i+3,"Predict Label_w=3"]="-"
    elif sp500.loc[i,"True Label"]=="-" and sp500.loc[i+1,"True Label"]=="-" and sp500.loc[i+2,"True Label"]=="-":
        sp500.loc[i+3,"Predict Label_w=3"]="+"
    i+=1


# In[95]:


#Question 2_1
#Spy W=2


#  +/+ total: 239
#  +/- total: 187
# -/+ total: 187
# -/- total: 137

i=-1
count_udu=0
count_udd=0
count_duu=0
count_dud=0
#count_uuu= 133
#count_ddd= 63

for e in sp500_first3year['True Label']:
    i+=1
    #Compare +/-/+ with +/-/-
    if e=='+'and (sp500_first3year.loc[i+1,'True Label'])=='-'and (sp500_first3year.loc[i+2,'True Label']=='+'):
        print(e,sp500_first3year.loc[i,'Date'])
        count_udu+=1
        print(f'this is +/-/+  {count_udu}\n')  
        
    if e=='+'and (sp500_first3year.loc[i+1,'True Label'])=='-'and (sp500_first3year.loc[i+2,'True Label']=='-'):
        print(e,sp500_first3year.loc[i,'Date'])
        count_udd+=1
        print(f'this is +/-/-  {count_udd}\n')  
        
        
    #Compare -/+/+ with -/+/-  
    elif e=='-'and (sp500_first3year.loc[i+1,'True Label'])=='+'and (sp500_first3year.loc[i+2,'True Label']=='+'):
        print(e,sp500_first3year.loc[i,'Date'])
        count_duu+=1
        print(f'this is -/+/+  {count_duu}\n') 
    
    elif e=='-' and (sp500_first3year.loc[i+1,'True Label']=='+')and (sp500_first3year.loc[i+3,'True Label']=='-'):
        print(e,sp500_first3year.loc[i,'Date'])
        count_dud+=1
        print(f'this is -/+/-  {count_dud}\n') 
        

        
#P() represents probability

#########(-/-/+)wins!!!!!!###########################
#P（-/-/+）=  81/137     Probability（-/-/-）= 63/137


#########(+/+/+)wins!!!!!!###########################
#P（+/+/+）= 119/239     Probability（+/+/-）= 99/239



#########(+/-/+)wins!!!!!!###########################
#P(+/-/+)= 114/187
#P(+/-/-)= 73/187


########(-/+/+)wins!!!!!!############################
#P(-/+/+)= 104/187
#P(-/+/-)= 28/187

#we can tell that w=2, it always go with "+"


# In[96]:


#Question2_1

#Spy W=2

#assign value to the predict label when w=2
i=753
while i>=753 and i <=1255:
    i+=1
    if sp500.loc[i,"True Label"]=="+" and sp500.loc[i+1,"True Label"]=="+" :
        sp500.loc[i+2,"Predict Label_w=2"]="+"
    elif sp500.loc[i,"True Label"]=="+" and sp500.loc[i+1,"True Label"]=="-" :
        sp500.loc[i+2,"Predict Label_w=2"]="+"
    elif sp500.loc[i,"True Label"]=="-" and sp500.loc[i+1,"True Label"]=="+" :
        sp500.loc[i+2,"Predict Label_w=2"]="+"
    elif sp500.loc[i,"True Label"]=="-" and sp500.loc[i+1,"True Label"]=="-" :
        sp500.loc[i+2,"Predict Label_w=2"]="+"


# In[97]:


#Question 2_1
#Spy W=4

#  +/+/-/- total:45
c1=0 #this is +/+/-/-/+ 
c2=0 #this is +/+/-/-/-

# +/+/-/+ total:59
c3=0 #this is +/+/-/+/+
c4=0 #this is +/+/-/+/-

# +/-/+/+ total:69
c5=0 #this is +/-/+/+/+
c6=0 #this is +/-/+/+/-

# +/-/+/- total:45
c7=0 #this is +/-/+/-/+
c8=0 #this is +/-/+/-/-

#+/-/-/+ total:40
c9=0 #this is +/-/-/+/+
c10=0 #this is +/-/-/+/-

#+/-/-/- total:32
c11=0 #this is +/-/-/-/+
c12=0 #this is +/-/-/-

#-/+/+/+ total:70
c13=0 #this is -/+/+/+/+
c14=0 #this is -/+/+/+/-

#-/+/+/- total:32
c15=0 #this is -/+/+/-/+
c16=0 #this is -/+/+/-/-

#-/+/-/+ total:55
c17=0 #this is -/+/-/+/+
c18=0 #this is -/+/-/+/-

#-/+/-/- total:28
c19=0 #this is -/+/-/-/+
c20=0 #this is -/+/-/-/-

#-/-/+/+ total:35
c21=0 #this is -/-/+/+/+
c22=0 #this is -/-/+/+/-

#-/-/+/- total:38
c23=0 #this is -/-/+/-/+
c24=0 #this is -/-/+/-/-

#+/+/+/- total:72
c25=0 #this is +/+/+/-/+
c26=0 #this is +/+/+/-/-

#+/+/+/+ total:61
c27=0 #this is +/+/+/+/+ 
c28=0 #this is +/+/+/+/- 


#-/-/-/- total:30
c29=0 #this is -/-/-/-/+
c30=0 #this is -/-/-/-/-



#-/-/-/+ total:33
c31=0 #this is -/-/-/+/+ 
c32=0 #this is -/-/-/+/- 


i=-1
for e in sp500_first3year['True Label']:
    i+=1
    #Compare +/+/-/-/+ with +/+/-/-/-
    if e=='+'and (sp500_first3year.loc[i+1,'True Label'])=='+'and (sp500_first3year.loc[i+2,'True Label']=='-')and (sp500_first3year.loc[i+3,'True Label']=='-')and (sp500_first3year.loc[i+4,'True Label']=='+'):
        print(e,sp500_first3year.loc[i,'Date'])
        c1+=1
        print(f'this is +/+/-/-/+  {c1}\n')  
        
    if e=='+'and (sp500_first3year.loc[i+1,'True Label'])=='+'and (sp500_first3year.loc[i+2,'True Label']=='-')and (sp500_first3year.loc[i+3,'True Label']=='-')and (sp500_first3year.loc[i+3,'True Label']=='-'):
        print(e,sp500_first3year.loc[i,'Date'])
        c2+=1
        print(f'this is +/+/-/-/-  {c2}\n')   
        
        
    #Compare +/+/-/+/+ with +/+/-/+/-
    elif e=='+'and (sp500_first3year.loc[i+1,'True Label'])=='+'and (sp500_first3year.loc[i+2,'True Label']=='-')and (sp500_first3year.loc[i+3,'True Label']=='+')and (sp500_first3year.loc[i+4,'True Label']=='+'):
        print(e,sp500_first3year.loc[i,'Date'])
        c3+=1
        print(f'this is +/+/-/+/+  {c3}\n') 
    
    elif e=='+' and (sp500_first3year.loc[i+1,'True Label']=='+')and (sp500_first3year.loc[i+2,'True Label']=='-')and (sp500_first3year.loc[i+3,'True Label']=='+')and (sp500_first3year.loc[i+4,'True Label']=='-'):
        print(e,sp500_first3year.loc[i,'Date'])
        c4+=1
        print(f'this is +/+/-/+/-  {c4}\n') 
        
        
    #Compare +/-/+/+/+ with +/-/+/+/-
    elif e=='+' and (sp500_first3year.loc[i+1,'True Label']=='-')and (sp500_first3year.loc[i+2,'True Label']=='+')and (sp500_first3year.loc[i+3,'True Label']=='+')and (sp500_first3year.loc[i+4,'True Label']=='+'):
        print(e,sp500_first3year.loc[i,'Date'],)
        c5+=1
        print(f'this is +/-/+/+/+  {c5}\n') 
        
    elif e=='+' and (sp500_first3year.loc[i+1,'True Label']=='-')and (sp500_first3year.loc[i+2,'True Label']=='+')and (sp500_first3year.loc[i+3,'True Label']=='+')and (sp500_first3year.loc[i+4,'True Label']=='-'):
        print(e,sp500_first3year.loc[i,'Date'],)
        c6+=1
        print(f'this is +/-/+/+/-  {c6}\n')       
        
        
    #Compare +/-/+/-/+ with +/-/+/-/-
    elif e=='+' and (sp500_first3year.loc[i+1,'True Label']=='-')and (sp500_first3year.loc[i+2,'True Label']=='+')and (sp500_first3year.loc[i+3,'True Label']=='-')and (sp500_first3year.loc[i+4,'True Label']=='+'):
        print(e,sp500_first3year.loc[i,'Date'],)
        c7+=1
        print(f'this is +/-/+/-/+  {c7}\n')     
        
    elif e=='+' and (sp500_first3year.loc[i+1,'True Label']=='-')and (sp500_first3year.loc[i+2,'True Label']=='+')and (sp500_first3year.loc[i+3,'True Label']=='-')and (sp500_first3year.loc[i+4,'True Label']=='-'):
        print(e,sp500_first3year.loc[i,'Date'],)
        c8+=1
        print(f'this is +/-/+/-/-  {c8}\n') 

        
    #Compare +/-/-/+/+  with  +/-/-/+/-
    elif e=='+' and (sp500_first3year.loc[i+1,'True Label']=='-')and (sp500_first3year.loc[i+2,'True Label']=='-')and (sp500_first3year.loc[i+3,'True Label']=='+')and (sp500_first3year.loc[i+4,'True Label']=='+'):
        print(e,sp500_first3year.loc[i,'Date'],)
        c9+=1
        print(f'this is +/-/-/+/+  {c9}\n')     

    elif e=='+' and (sp500_first3year.loc[i+1,'True Label']=='-')and (sp500_first3year.loc[i+2,'True Label']=='-')and (sp500_first3year.loc[i+3,'True Label']=='+')and (sp500_first3year.loc[i+4,'True Label']=='-'):
        print(e,sp500_first3year.loc[i,'Date'],)
        c10+=1
        print(f'this is +/-/-/+/-  {c10}\n')     
        
    #Compare +/-/-/-/+ with +/-/-/-/-
    elif e=='+' and (sp500_first3year.loc[i+1,'True Label']=='-')and (sp500_first3year.loc[i+2,'True Label']=='-')and (sp500_first3year.loc[i+3,'True Label']=='-')and (sp500_first3year.loc[i+4,'True Label']=='+'):
        print(e,sp500_first3year.loc[i,'Date'],)
        c11+=1
        print(f'this is +/-/-/-/+  {c11}\n')     
        
    elif e=='+' and (sp500_first3year.loc[i+1,'True Label']=='-')and (sp500_first3year.loc[i+2,'True Label']=='-')and (sp500_first3year.loc[i+3,'True Label']=='-')and (sp500_first3year.loc[i+4,'True Label']=='-'):
        print(e,sp500_first3year.loc[i,'Date'],)
        c12+=1
        print(f'this is +/-/-/-/-  {c12}\n')  
        
        
    #Compare -/+/+/+/+  with -/+/+/+/-  
    if e=='-'and (sp500_first3year.loc[i+1,'True Label'])=='+'and (sp500_first3year.loc[i+2,'True Label']=='+')and (sp500_first3year.loc[i+3,'True Label']=='+')and (sp500_first3year.loc[i+4,'True Label']=='+'):
        print(e,sp500_first3year.loc[i,'Date'])
        c13+=1
        print(f'this is -/+/+/+/+  {c13}\n')  
        
    if e=='-'and (sp500_first3year.loc[i+1,'True Label'])=='+'and (sp500_first3year.loc[i+2,'True Label']=='+')and (sp500_first3year.loc[i+3,'True Label']=='+')and (sp500_first3year.loc[i+4,'True Label']=='-'):
        print(e,sp500_first3year.loc[i,'Date'])
        c14+=1
        print(f'this is -/+/+/+/-  {c14}\n')  
        
    
        #Compare -/+/+/-/+  with -/+/+/-/-  
    if e=='-'and (sp500_first3year.loc[i+1,'True Label'])=='+'and (sp500_first3year.loc[i+2,'True Label']=='+')and (sp500_first3year.loc[i+3,'True Label']=='-')and (sp500_first3year.loc[i+4,'True Label']=='+'):
        print(e,sp500_first3year.loc[i,'Date'])
        c15+=1
        print(f'this is -/+/+/-/+   {c15}\n')  
        
    if e=='-'and (sp500_first3year.loc[i+1,'True Label'])=='+'and (sp500_first3year.loc[i+2,'True Label']=='+')and (sp500_first3year.loc[i+3,'True Label']=='-')and (sp500_first3year.loc[i+4,'True Label']=='-'):
        print(e,sp500_first3year.loc[i,'Date'])
        c16+=1
        print(f'this is -/+/+/-/-  {c16}\n')  
        
        
        #Compare -/+/-/+/+  with -/+/-/+/-  
    if e=='-'and (sp500_first3year.loc[i+1,'True Label'])=='+'and (sp500_first3year.loc[i+2,'True Label']=='-')and (sp500_first3year.loc[i+3,'True Label']=='+')and (sp500_first3year.loc[i+4,'True Label']=='+'):
        print(e,sp500_first3year.loc[i,'Date'])
        c17+=1
        print(f'this is -/+/-/+/+ {c17}\n')  
        
    if e=='-'and (sp500_first3year.loc[i+1,'True Label'])=='+'and (sp500_first3year.loc[i+2,'True Label']=='-')and (sp500_first3year.loc[i+3,'True Label']=='+')and (sp500_first3year.loc[i+4,'True Label']=='-'):
        print(e,sp500_first3year.loc[i,'Date'])
        c18+=1
        print(f'this is -/+/-/+/-  {c18}\n')  

        
        #Compare -/+/-/-/+  with -/+/-/-/-  
    if e=='-'and (sp500_first3year.loc[i+1,'True Label'])=='+'and (sp500_first3year.loc[i+2,'True Label']=='-')and (sp500_first3year.loc[i+3,'True Label']=='-')and (sp500_first3year.loc[i+4,'True Label']=='+'):
        print(e,sp500_first3year.loc[i,'Date'])
        c19+=1
        print(f'this is -/+/-/-/+   {c19}\n')  
        
    if e=='-'and (sp500_first3year.loc[i+1,'True Label'])=='+'and (sp500_first3year.loc[i+2,'True Label']=='-')and (sp500_first3year.loc[i+3,'True Label']=='-')and (sp500_first3year.loc[i+4,'True Label']=='-'):
        print(e,sp500_first3year.loc[i,'Date'])
        c20+=1
        print(f'this is -/+/-/-/-  {c20}\n')  
        
        
            #Compare -/-/+/+/+  with -/-/+/+/-  
    if e=='-'and (sp500_first3year.loc[i+1,'True Label'])=='-'and (sp500_first3year.loc[i+2,'True Label']=='+')and (sp500_first3year.loc[i+3,'True Label']=='+')and (sp500_first3year.loc[i+4,'True Label']=='+'):
        print(e,sp500_first3year.loc[i,'Date'])
        c21+=1
        print(f'this is -/-/+/+/+  {c21}\n')  
        
    if e=='-'and (sp500_first3year.loc[i+1,'True Label'])=='-'and (sp500_first3year.loc[i+2,'True Label']=='+')and (sp500_first3year.loc[i+3,'True Label']=='+')and (sp500_first3year.loc[i+4,'True Label']=='-'):
        print(e,sp500_first3year.loc[i,'Date'])
        c22+=1
        print(f'this is -/-/+/+/-  {c22}\n')  
        
        
            #Compare -/-/+/-/+  with -/-/+/-/-  
    if e=='-'and (sp500_first3year.loc[i+1,'True Label'])=='-'and (sp500_first3year.loc[i+2,'True Label']=='+')and (sp500_first3year.loc[i+3,'True Label']=='-')and (sp500_first3year.loc[i+4,'True Label']=='+'):
        print(e,sp500_first3year.loc[i,'Date'])
        c23+=1
        print(f'this is -/-/+/-/+  {c23}\n')  
        
    if e=='-'and (sp500_first3year.loc[i+1,'True Label'])=='-'and (sp500_first3year.loc[i+2,'True Label']=='+')and (sp500_first3year.loc[i+3,'True Label']=='-')and (sp500_first3year.loc[i+4,'True Label']=='-'):
        print(e,sp500_first3year.loc[i,'Date'])
        c24+=1
        print(f'this is -/-/+/-/-  {c24}\n')  
        
        
            #Compare +/+/+/-/+  with +/+/+/-/-  
    if e=='+'and (sp500_first3year.loc[i+1,'True Label'])=='+'and (sp500_first3year.loc[i+2,'True Label']=='+')and (sp500_first3year.loc[i+3,'True Label']=='-')and (sp500_first3year.loc[i+4,'True Label']=='+'):
        print(e,sp500_first3year.loc[i,'Date'])
        c25+=1
        print(f'this is +/+/+/-/+  {c25}\n')  
        
    if e=='+'and (sp500_first3year.loc[i+1,'True Label'])=='+'and (sp500_first3year.loc[i+2,'True Label']=='+')and (sp500_first3year.loc[i+3,'True Label']=='-')and (sp500_first3year.loc[i+4,'True Label']=='-'):
        print(e,sp500_first3year.loc[i,'Date'])
        c26+=1
        print(f'this is +/+/+/-/-   {c26}\n')  
        
        
            #Compare +/+/+/+/+  with +/+/+/+/-  
    if e=='+'and (sp500_first3year.loc[i+1,'True Label'])=='+'and (sp500_first3year.loc[i+2,'True Label']=='+')and (sp500_first3year.loc[i+3,'True Label']=='+')and (sp500_first3year.loc[i+4,'True Label']=='+'):
        print(e,sp500_first3year.loc[i,'Date'])
        c27+=1
        print(f'this is +/+/+/+/+   {c27}\n')  
        
    if e=='+'and (sp500_first3year.loc[i+1,'True Label'])=='+'and (sp500_first3year.loc[i+2,'True Label']=='+')and (sp500_first3year.loc[i+3,'True Label']=='+')and (sp500_first3year.loc[i+4,'True Label']=='-'):
        print(e,sp500_first3year.loc[i,'Date'])
        c28+=1
        print(f'this is +/+/+/+/-  {c28}\n')  
        
        
            #Compare -/-/-/-/+  with -/-/-/-/-  
    if e=='-'and (sp500_first3year.loc[i+1,'True Label'])=='-'and (sp500_first3year.loc[i+2,'True Label']=='-')and (sp500_first3year.loc[i+3,'True Label']=='-')and (sp500_first3year.loc[i+4,'True Label']=='+'):
        print(e,sp500_first3year.loc[i,'Date'])
        c29+=1
        print(f'this is -/-/-/-/+  {c29}\n')  
        
    if e=='-'and (sp500_first3year.loc[i+1,'True Label'])=='-'and (sp500_first3year.loc[i+2,'True Label']=='-')and (sp500_first3year.loc[i+3,'True Label']=='-')and (sp500_first3year.loc[i+4,'True Label']=='-'):
        print(e,sp500_first3year.loc[i,'Date'])
        c30+=1
        print(f'this is -/-/-/-/-  {c30}\n')  
        
        
            #Compare -/-/-/+/+  with -/-/-/+/-  
    if e=='-'and (sp500_first3year.loc[i+1,'True Label'])=='-'and (sp500_first3year.loc[i+2,'True Label']=='-')and (sp500_first3year.loc[i+3,'True Label']=='+')and (sp500_first3year.loc[i+4,'True Label']=='+'):
        print(e,sp500_first3year.loc[i,'Date'])
        c31+=1
        print(f'this is -/-/-/+/+   {c31}\n')  
        
    if e=='-'and (sp500_first3year.loc[i+1,'True Label'])=='-'and (sp500_first3year.loc[i+2,'True Label']=='-')and (sp500_first3year.loc[i+3,'True Label']=='+')and (sp500_first3year.loc[i+4,'True Label']=='-'):
        print(e,sp500_first3year.loc[i,'Date'])
        c32+=1
        print(f'this is -/-/-/+/-   {c32}\n')  
        
        
#Use training data to analyze next one probability 
#P() represents probability

#Comparison frequencies are followings
#(+/+/-/+/+): 31            
#(+/+/-/+/-): 28

#(+/+/+/-/+): 44   
#(+/+/+/-/-): 28


#(-/+/+/+/-): 40
#(-/+/+/+/+): 30


#(-/-/-/+/+): 15
#(-/-/-/+/-): 18

#(+/+/-/-/- ):45
#(+/+/-/-/+ ):24


#(+/-/-/-/+ ): 15 
#(+/-/-/-/- ): 17


#(-/-/-/-/+):  17
#(-/-/-/-/-):13


#(+/-/+/-/-): 16
#(+/-/+/-/+) :29

#(-/+/-/+/-):17
#(-/+/-/+/+):38


#(-/+/+/-/+):  15
#(-/+/+/-/-):  17

#(+/-/+/+/-):  26
#(+/-/+/+/+): 42

#(+/-/-/+/-):20
#(+/-/-/+/+):20

#(+/+/+/+/-):32
#(+/+/+/+/+):29


#(-/-/+/+/+):28
#(-/-/+/+/-):6

#(-/-/+/-/+): 26
#(-/-/+/-/-):12

#(-/+/-/-/-): 11
#(-/+/-/-/+):16



# In[98]:


#Spy W=4

#assign value to the predict label when w=4. 6 patterns next one more likely to go with "-" ; otherwise "+"
i=751
while i>=751 and i <=1253:
    i+=1
    if sp500.loc[i,"True Label"]=="-" and sp500.loc[i+1,"True Label"]=="+" and sp500.loc[i+2,"True Label"]=="+" and sp500.loc[i+3,"True Label"]=="+":
        sp500.loc[i+4,"Predict Label_w=4"]="-"
    elif sp500.loc[i,"True Label"]=="-" and sp500.loc[i+1,"True Label"]=="-" and sp500.loc[i+2,"True Label"]=="-" and sp500.loc[i+3,"True Label"]=="+":
        sp500.loc[i+4,"Predict Label_w=4"]="-"
    elif sp500.loc[i,"True Label"]=="+" and sp500.loc[i+1,"True Label"]=="+" and sp500.loc[i+2,"True Label"]=="-" and sp500.loc[i+3,"True Label"]=="-" :
        sp500.loc[i+4,"Predict Label_w=4"]="-"
    elif sp500.loc[i,"True Label"]=="+" and sp500.loc[i+1,"True Label"]=="-" and sp500.loc[i+2,"True Label"]=="-" and sp500.loc[i+3,"True Label"]=="-" :
        sp500.loc[i+4,"Predict Label_w=4"]="-"
    elif sp500.loc[i,"True Label"]=="-" and sp500.loc[i+1,"True Label"]=="+" and sp500.loc[i+2,"True Label"]=="+" and sp500.loc[i+3,"True Label"]=="-" :
        sp500.loc[i+4,"Predict Label_w=4"]="-"
    elif sp500.loc[i,"True Label"]=="+" and sp500.loc[i+1,"True Label"]=="+" and sp500.loc[i+2,"True Label"]=="+" and sp500.loc[i+3,"True Label"]=="+" :
        sp500.loc[i+4,"Predict Label_w=4"]="-"
    else:
        sp500.loc[i+4,"Predict Label_w=4"]="+"


# In[99]:


#Question2_2&3
#Spy W2 accuracy

a=0
b=0
i=755

for e in sp500[sp500['Year']>=2021]['True Label']:
    i+=1
    if e ==sp500.loc[i,'Predict Label_w=2']:
        a+=1
    else:
        b+=1
print(f'Spy Accuracy for W2 is: {(a/(a+b))*100}%')


# In[104]:


#Question2_2&3
#Spy W3 accuracy

a=0
b=0
i=755

for e in sp500[sp500['Year']>=2021]['True Label']:
    i+=1
    if e ==sp500.loc[i,'Predict Label_w=3']:
        a+=1
    else:
        b+=1
print(f'Spy Accuracy for W3 is: {(a/(a+b))*100}%')


# In[105]:


#Question2_2&3
#Spy W4 accuracy

a=0
b=0
i=755

for e in sp500[sp500['Year']>=2021]['True Label']:
    i+=1
    if e ==sp500.loc[i,'Predict Label_w=4']:
        a+=1
    else:
        b+=1
print(f'Spy Accuracy for W4 is: {(a/(a+b))*100}%')


# In[ ]:


#Question2_2&3
# W4 and W3 have higher accuracy


# In[114]:


#Spy #Calculating each W2 accuracy for predicting either "+" or "-"
a_u=0
a_d=0
b_u=0
b_d=0
i=755

for e in sp500[sp500['Year']>=2021]['True Label']:
    i+=1
    if (sp500.loc[i,'Predict Label_w=2']=='+') and (e =='+'):
        a_u+=1
    elif(sp500.loc[i,'Predict Label_w=2']=='-') and (e =='-'):
        a_d+=1
    elif (sp500.loc[i,'Predict Label_w=2']=='+') and (e =='-'):
        b_u+=1
    elif (sp500.loc[i,'Predict Label_w=2']=='-') and (e =='+'):
        b_d+=1

print(f'Spy W2 TP value: {a_u}')
print(f'Spy W2 TN value: {a_d}')
print(f'Spy W2 FP value: {b_u}')
print(f'Spy W2 FN value: {b_d}')
print(f'W2 Accuracy for predicting "+"  is: {(a_u/(a_u+b_u))*100}%')
print(a_d)
print(f'W2 Accuracy for predicting "-"  is: 0')

print('SPY W2 Predicting "+"  has higher accuracy')


# In[113]:


#Question2_2&3
#Spy #Calculating each W3 accuracy for predicting either "+" or "-"
a_u=0
a_d=0
b_u=0
b_d=0
i=755

for e in sp500[sp500['Year']>=2021]['True Label']:
    i+=1
    if (sp500.loc[i,'Predict Label_w=3']=='+') and (e =='+'):
        a_u+=1
    elif(sp500.loc[i,'Predict Label_w=3']=='-') and (e =='-'):
        a_d+=1
    elif (sp500.loc[i,'Predict Label_w=3']=='+') and (e =='-'):
        b_u+=1
    elif (sp500.loc[i,'Predict Label_w=3']=='-') and (e =='+'):
        b_d+=1

print(f'Spy W3 TP value: {a_u}')
print(f'Spy W3 TN value: {a_d}')
print(f'Spy W3 FP value: {b_u}')
print(f'Spy W3 FN value: {b_d}')        
print(f'W3 Accuracy for predicting "+"  is: {(a_u/(a_u+b_u))*100}%')
print(f'W3 Accuracy for predicting "-"  is: {(a_d/(a_d+b_d))*100}%')
print('SPY W3 Predicting "+"  has higher accuracy')


# In[112]:


#Question2_2&3
#Spy #Calculating each W2 accuracy for predicting either "+" or "-"

a_u=0
a_d=0
b_u=0
b_d=0
i=755

for e in sp500[sp500['Year']>=2021]['True Label']:
    i+=1
    if (sp500.loc[i,'Predict Label_w=4']=='+') and (e =='+'):
        a_u+=1
    elif(sp500.loc[i,'Predict Label_w=4']=='-') and (e =='-'):
        a_d+=1
    elif (sp500.loc[i,'Predict Label_w=4']=='+') and (e =='-'):
        b_u+=1
    elif (sp500.loc[i,'Predict Label_w=4']=='-') and (e =='+'):
        b_d+=1

print(f'Spy W4 TP value: {a_u}')
print(f'Spy W4 TN value: {a_d}')
print(f'Spy W4 FP value: {b_u}')
print(f'Spy W4 FN value: {b_d}')
print(f'W4 Accuracy for predicting "+"  is: {(a_u/(a_u+b_u))*100}%')
print(f'W4 Accuracy for predicting "-"  is: {(a_d/(a_d+b_d))*100}%')
print('SPY W4 Predicting "+"  has higher accuracy')


# In[67]:


sp500['Ensemble']=''


# In[70]:


#Question3_2&3
i=-1
for e in sp500['Predict Label_w=2']:
    i+=1
    if (e==sp500.loc[i,'Predict Label_w=3']) or (e==sp500.loc[i,'Predict Label_w=4']):
        sp500.loc[i,'Ensemble']=e
    else:
        sp500.loc[i,'Ensemble']=sp500.loc[i,'Predict Label_w=3']


# In[82]:


#Question3_2&3
#Spy

a=0
b=0
i=755

for e in sp500[sp500['Year']>=2021]['True Label']:
    i+=1
    if e ==sp500.loc[i,'Ensemble']:
        a+=1
    else:
        b+=1
print(f'Spy Accuracy for Ensemble is: {(a/(a+b))*100}%')


# In[111]:


#Question3_2&3
#Spy

a_u=0
a_d=0
b_u=0
b_d=0
i=755

for e in sp500[sp500['Year']>=2021]['True Label']:
    i+=1
    if (sp500.loc[i,'Ensemble']=='+') and (e =='+'):
        a_u+=1
    elif(sp500.loc[i,'Ensemble']=='-') and (e =='-'):
        a_d+=1
    elif (sp500.loc[i,'Ensemble']=='+') and (e =='-'):
        b_u+=1
    elif (sp500.loc[i,'Ensemble']=='-') and (e =='+'):
        b_d+=1
        
print(f'Spy Ensemble TP value: {a_u}')
print(f'Spy Ensemble TN value: {a_d}')
print(f'Spy Ensemble FP value: {b_u}')
print(f'Spy Ensemble FN value: {b_d}')

print(f'Spy Ensemble Accuracy for predicting "+"  is: {(a_u/(a_u+b_u))*100}%')
print(f'Spy Ensemble Accuracy for predicting "-"  is: {(a_d/(a_d+b_d))*100}%')
print('SPY Ensemble Predicting "+"  has higher accuracy')


# In[ ]:


#Question 3_2
#Spy  Comparing '-' or "+"accuracy

# Ensemble "-" : 47.92%
# W2 "-" : 0
# W3 "-" : 51.94%
# W4 "-" : 50.52%

# Ensemble "+" : 50.25%
# W2 "+" : 50.6%
# W3 "+" : 51.35%
# W4 "+" : 51.62%



#Conclusion: W3 has higher predicting accuracy for "-"
#Conclusion: W4 has higher predicting accuracy for "+"
#Findings: with longer patterns to analyze, we will have higher precision.


# In[72]:


sp500[sp500['Year']==2021].head(30)


# In[ ]:


#Question 3
#w=2 , because the probability of "+" is higher in training data, so it automatically sets to "+"


# In[ ]:


df['ensemble_predict_w3']=' '


# In[ ]:


#Question 3
#ensemble w=3 Ensemble method

i=752
while i>=752 and i<=1254:
    i+=1
    if df.loc[i,'True Label']=='+':
        if df.loc[i+1,'True Label']=='+':
            df.loc[i+3,'ensemble_predict_w3']='+'
        elif df.loc[i+1,'True Label']=='-':
            
            if df.loc[i+2,'True Label']=='-':
                df.loc[i+3,'ensemble_predict_w3']='-'
            
            elif df.loc[i+2,'True Label']=='+':
                df.loc[i+3,'ensemble_predict_w3']='+' 
                
    elif df.loc[i,'True Label']=='-':
        if df.loc[i+1,'True Label']=='+':
            if df.loc[i+1,'True Label']=='+':
                df.loc[i+3,'ensemble_predict_w3']='+'
            elif df.loc[i+1,'True Label']=='+':
                df.loc[i+3,'ensemble_predict_w3']='-'
        elif df.loc[i+1,'True Label']=='-':
            df.loc[i+3,'ensemble_predict_w3']='-'


# In[ ]:


df['ensemble_predict_w4']=''


# In[ ]:


#Question 3
#///SPECIAL NOTES/// 
#///HERE IS A DIFFERENT METHOD
# Since most of my data predict for "+",I found ENSEMBLE BASED ON SPECIFIC NO. work better.


#Ensemble w=4  Special Ensemble method

i=751
while i>=751 and i<=1253:
    i+=1
    if df.loc[i,'True Label']=='+':
        if df.loc[i+1,'True Label']=='+':
            if df.loc[i+2,'True Label']=='+':
                df.loc[i+4,'ensemble_predict_w4']='+'
            else:
                df.loc[i+4,'ensemble_predict_w4']='+'
                
        elif df.loc[i+1,'True Label']=='-':
            
            if df.loc[i+2,'True Label']=='+':
                df.loc[i+4,'ensemble_predict_w4']='+'
            
            elif df.loc[i+2,'True Label']=='-':
                if df.loc[i+3,'True Label']=='+':
                    df.loc[i+4,'ensemble_predict_w4']='+'
                elif df.loc[i+3,'True Label']=='-':
                    df.loc[i+4,'ensemble_predict_w4']='-'  
                    
    elif df.loc[i,'True Label']=='-':
        if df.loc[i+1,'True Label']=='+':
            if df.loc[i+2,'True Label']=='+':
                df.loc[i+4,'ensemble_predict_w4']='+'
            elif df.loc[i+2,'True Label']=='-':
                if df.loc[i+2,'True Label']=='+':
                    df.loc[i+4,'ensemble_predict_w4']='+'
                elif df.loc[i+2,'True Label']=='-':
                    df.loc[i+4,'ensemble_predict_w4']='-'
        elif df.loc[i+1,'True Label']=='-':
            
            if df.loc[i+2,'True Label']=='+':
                df.loc[i+4,'ensemble_predict_w4']='+'
            
            elif df.loc[i+2,'True Label']=='-':
                df.loc[i+4,'ensemble_predict_w4']='-' 


# In[ ]:


df["ensemble_predict_w2"]=" "


# In[ ]:


#Ensemble w2

i=753
while i>=753 and i<=1255:
    i+=1
    if df.loc[i,'True Label']=='+':
        df.loc[i+2,'ensemble_predict_w2']='+'
                
    elif df.loc[i,'True Label']=='-':
        if df.loc[i+1,'True Label']=='+':
            df.loc[i+2,'ensemble_predict_w2']='+'
        elif df.loc[i+1,'True Label']=='-':
            df.loc[i+2,'ensemble_predict_w2']='-'


# In[ ]:


df[df["Year"]==2022].head(30)


# In[ ]:


df['ensemble_predict_w2'].value_counts()
#total "+" for ensemble w2 :1262
#total "-" for ensemble w2 :0


# In[ ]:


df['ensemble_predict_w3'].value_counts()
#total "+" for ensemble w3 :326
#total "-" for ensemble w3 :174


# In[ ]:


df['ensemble_predict_w4'].value_counts()
#total "+" for ensemble w4 :336
#total "-" for ensemble w4 :164


# In[ ]:


df['ensemble']='+'


# In[ ]:


#total "+" for ensemble w2 : 384
#total "-" for ensemble w2 : 117

count_en_rightw2=0
count_en_rightw2_u=0
count_en_rightw2_d=0

i=755
while i>=755 and i <=1257:
    i+=1
    if df.loc[i,'ensemble_predict_w2']==df.loc[i,'True Label']:
        count_en_rightw2+=1
        if df.loc[i,'ensemble_predict_w2']=='+':
            count_en_rightw2_u+=1
        elif df.loc[i,'ensemble_predict_w2']=='-':
            count_en_rightw2_d+=1
    else:
        continue
        
print(f'probability of right predict for ensemble w2 is: {(count_en_rightw2/503)*100}%\n')
# the predict 50.10% is more accurate than before. previously accuracy is only around 25%
print('Compared to the Right predict Probability of "+"= 49.8%')
print(f'probability of predict for right"+" by ensemble w2 is: {(count_en_rightw2_u/384)*100}% has lower accuracy.')

print('\n\nCompared to the Right predict Probability of "-"= 0')
print(f'\nprobability of predict for right"-" by ensemble w2 is: {(count_en_rightw2_d/117)*100}% has lower accuracy.')


# In[ ]:


#total "+" for ensemble w3 :326
#total "-" for ensemble w3 :174

count_en_rightw3=0
count_en_rightw3_u=0
count_en_rightw3_d=0

i=755
while i>=755 and i <=1257:
    i+=1
    if df.loc[i,'ensemble_predict_w3']==df.loc[i,'True Label']:
        count_en_rightw3+=1
        if df.loc[i,'ensemble_predict_w3']=='+':
            count_en_rightw3_u+=1
        elif df.loc[i,'ensemble_predict_w3']=='-':
            count_en_rightw3_d+=1  
    else:
        continue
        
print(f'probability of right predict for ensemble w3 is: {(count_en_rightw3/503)*100}%\n\n')
# the predict 51.89% is more accurate than before. previously accuracy is only around 25%
print('Compared to the Right predict Probability of "+"is: 49.8%')
print(f'probability of predict for right"+" by ensemble w3 is: {(count_en_rightw3_u/326)*100}% has higher accuracy.\n\n')

print('Compared to the Right predict Probability of "-"is: 0')
print(f'probability of predict for right"-" by ensemble w3 is: {(count_en_rightw3_d/174)*100}% has higher accuracy.')


# In[ ]:


#total "+" for ensemble w4 :336
#total "-" for ensemble w4 :164

count_en_rightw4=0
count_en_rightw4_u=0  #represent number of counting "+" right
count_en_rightw4_d=0  #represent number of counting "-" right

i=755
while i>=755 and i <=1257:
    i+=1
    if df.loc[i,'ensemble_predict_w4']==df.loc[i,'True Label']:
        count_en_rightw4+=1
        if df.loc[i,'ensemble_predict_w4']=='+':
            count_en_rightw4_u+=1
        elif df.loc[i,'ensemble_predict_w4']=='-':
            count_en_rightw4_d+=1      
    else:
        continue
        
print(f'probability of total right predict by ensemble w4 is: {(count_en_rightw4/503)*100}%')
# the predict 49.7% is more accurate than before. previously accuracy is only around 25%
print('\nCompared to the Right predict Probability= 49.8%')
print(f'probability of predict for right"+" by ensemble w4 is: {(count_en_rightw4_u/336)*100}% has higher accuracy.')

print('\nCompared to the Right predict Probability= 0')
print(f'probability of predict for right"-" by ensemble w4 is: {(count_en_rightw4_d/164)*100}% has higher accuracy.')


# In[116]:


#Question 4

data = {'W': ['2','3','4','ensemble','2','3','4','ensemble'],
        'Ticker': ['S&P-500','S&P-500','S&P-500','S&P-500','Your Stock','Your Stock','Your Stock','Your Stock'],
        'TP':[253,190,159,203,249,249,249,249],
       'FP':[247,180,149,201,251,251,251,251],
       'TN':[0,67,98,46,0,0,0,0],
       'FN':[0,62,96,50,0,0,0,0],
       'Accuracy':['50.30%','51.09%','51.09%','49.50%','25%','25%','25%','25%'],
       'TPR':['100%','75.40%','62.40%','80.24%','49.8%','49.8%','49.8%','49.8%'],
       'TNR':['0','27.13%','39.68%','18.62%',0,0,0,0],}

summary = pd.DataFrame(data)
print(summary)
#Even though Ensemble doesn't have the highest accuracy, it has higher "+"prediction than any other W.
#W3&4 has higher prediction in total, which possibly means using longer patterns to predict is wiser choice. 


# In[120]:


#Question 5

df['Accumulative Balance']=''


# In[123]:


#Question 5

#put $100 in stock and see what's left in the end
i=756
df.loc[756,'Accumulative Balance']=100
while i>=756 and i<=1257:
    i+=1
    df.loc[i,'Accumulative Balance']=(df.loc[i-1,'Accumulative Balance'])*(1+df.loc[i,'Return'])


# In[128]:


#we can see that at the end, there is 100.13233 dollars compared to the starting fund $100 
# Earn 0.13233 dollars if we do nothing.
df[df['Year']>=2021]


# In[147]:


#slightly adjust table value for aesthetics
df.loc[df['Year'] <= 2020, 'Predict Label_w=4'] = ""


# In[149]:


#slightly adjust table value for aesthetics
df.loc[df['Year'] <= 2020, 'Predict Label_w=3'] = ""


# In[150]:


#slightly adjust table value for aesthetics
df.loc[df['Year'] <= 2020, 'Predict Label_w=2'] = ""


# In[151]:


df.head()


# In[135]:


df_last2years=df[df['Year']>=2021]


# In[161]:


#Question 5
#If you accumulate money $100 for two years, the graph shows below
plt.figure(figsize=(18, 6)) 
plt.plot(df_last2years['Date'], df_last2years['Accumulative Balance'])


# Add labels and title
plt.xlabel('Date')
plt.ylabel('Accumulative Balance')
plt.title('$100 in Starbucks')

# Show the plot
plt.show()


# In[ ]:


#Question 5
#My best W strategy always goes for "+" for the next day, 
#it is the same graph as Accumulated Balance, as previously shown.


# In[ ]:




