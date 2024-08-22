#!/usr/bin/env python
# coding: utf-8

# ## Project Proposal  
# ### by Yidan Sun
# Introduction: Use Kaggle LOL 15 minutes match Diamond data (50000 in total) to train different classifiers(eliminate the less weighted features), analyze features and implement the final strategy that has the best accuracy.

# In[1]:


import numpy as np
import pandas as pd
import math
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import probplot

from sklearn . linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.ensemble import RandomForestClassifier


from sklearn import svm

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


# In[2]:


df=pd.read_csv('MatchTimelinesFirst15.csv')
df_origin_blue=df.drop(columns=['blueDragonKills','redDragonKills','matchId','blue_win','redGold','redMinionsKilled','redJungleMinionsKilled','redAvgLevel','redChampKills','redHeraldKills','redTowersDestroyed'])
df_origin_red=df.drop(columns=['blueDragonKills','redDragonKills','matchId','blue_win','blueGold','blueMinionsKilled','blueJungleMinionsKilled','blueAvgLevel','blueChampKills','blueHeraldKills','blueTowersDestroyed'])


# In[3]:


24062/48651


# In[4]:


df.head()


# In[5]:


df_origin_red
correlation_matrix = df_origin_red.corr()

# Set up the matplotlib figure
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, square=True)
plt.yticks(rotation=0)
# Show the plot
plt.show()


# In[6]:


df_origin_blue
correlation_matrix = df_origin_blue.corr()

# Set up the matplotlib figure
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, square=True)
plt.yticks(rotation=0)
# Show the plot
plt.show()


# In[7]:


#Total No. of Games =48651 
#Blue Team wins: 24589 (round 50%)
df['blue_win'].describe()


# In[8]:


df_blue=df[df['blue_win']==1]
df_red=df[df['blue_win']==0]


# ## KNN=4 (51.90%) 

# In[9]:


X = df[[ 'blueChampKills', 'blueHeraldKills','blueTowersDestroyed']].values
Y = df['blue_win'].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=42)

scaler = StandardScaler (). fit (X_train)
X_train = scaler . transform (X_train)
knn_classifier = KNeighborsClassifier ( n_neighbors =4)
knn_classifier . fit (X_train,Y_train)

prediction = knn_classifier.predict ( X_test)
acc=accuracy_score(Y_test,prediction)
cf=confusion_matrix(Y_test,prediction)

print(acc)
print(cf)


# In[10]:


KNN_TPR=round(12366*100/(12366+4),2)
KNN_TNR=round(260*100/(260+11696),2)
print(f'KNN TPR:{KNN_TPR}%')
print(f'KNN TNR:{KNN_TNR}%')


# In[11]:


df.insert(4,'KNN',0)


# In[12]:


prediction = knn_classifier.predict (X)
df['KNN']=prediction


# ## Logistic Regression(50.85%)

# In[13]:


X = df[['blueGold', 'blueMinionsKilled', 'blueJungleMinionsKilled', 'blueAvgLevel', 'blueChampKills','blueHeraldKills', 'blueTowersDestroyed']].values
Y = df['blue_win'].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=42)

scaler = StandardScaler ()
scaler .fit(X_train)
X_train = scaler .transform (X_train)

log_reg_classifier = LogisticRegression ()
log_reg_classifier . fit (X_train,Y_train)

predicted = log_reg_classifier.predict(X_test)
acc=accuracy_score(Y_test,predicted)
cf=confusion_matrix(Y_test,predicted)

print(acc)
print(cf)


# In[14]:


LOG_TPR=0
LOG_TNR=100
print(f'Logistics TPR:{LOG_TPR}%')
print(f'Logistics TNR:{LOG_TNR}%')


# ## Linear Discriminant Analysis(75.94%)

# In[15]:


X = df[[ 'blueGold', 'blueMinionsKilled', 'blueJungleMinionsKilled', 'blueAvgLevel','blueChampKills', 'blueHeraldKills', 'blueDragonKills', 'blueTowersDestroyed']].values
Y = df['blue_win'].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=42)

# Standardize the data
scaler = StandardScaler()
scaler.fit(X_train)
X_train= scaler.transform(X_train)
X_test= scaler.transform(X_test)
X_test=np.asarray(X_test)

lda_classifier = LDA( )
lda_classifier.fit(X_train, Y_train)
prediction = lda_classifier.predict(X_test)


accuracy = lda_classifier.score (X_test, Y_test)
cf=confusion_matrix(Y_test,prediction)

print(accuracy)
print(cf)


# In[16]:


#TPR and TNR
LDA_TPR= round(9445*100/(2925+9445),2)
LDA_TNR=round(9027*100/(9027+2929),2)
print(f'LinearDiscriminant Analysis TPR:{LDA_TPR}%')
print(f'LinearDiscriminant Analysis TNR:{LDA_TNR}%')


# ## Quadratic Discriminant Analysis(49.15%)

# In[17]:


X = df[['blueGold', 'blueMinionsKilled', 'blueJungleMinionsKilled', 'blueAvgLevel', 'blueChampKills', 'blueHeraldKills', 'blueDragonKills', 'blueTowersDestroyed']].values
Y = df['blue_win'].values


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=42)

scaler = StandardScaler()
scaler.fit(X_train)

X_scaled_train = scaler.transform(X_train)
X_scaled_test = scaler.transform(X_test)

qda_classifier = QDA()
qda_classifier.fit(X_scaled_train, Y_train)

predicted = qda_classifier.predict(X_scaled_test)


accuracy = qda_classifier.score(X_test, Y_test)
cf=confusion_matrix(predicted,Y_test)

print(accuracy)
print(cf)


# In[18]:


#TPR and TNR
QDA_TPR= 0
QDA_TNR=round(11956*100/(11956+12370),2)
print(f'Quadratic Discriminant Analysis TPR:{QDA_TPR}%')
print(f'Quadratic Discriminant Analysis TNR:{QDA_TNR}%')


# ## Naive Bayes(75.90%)

# In[19]:


#BlueJungleMinionsKilled has little importance in improving accuracy

X = df[['blueGold', 'blueMinionsKilled', 'blueAvgLevel', 'blueChampKills', 'blueHeraldKills', 'blueDragonKills','blueTowersDestroyed']].values
Y = df['blue_win'].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=42)
NB_classifier = GaussianNB (). fit (X_train, Y_train)
prediction = NB_classifier .predict (X_test)

accuracy=accuracy_score(Y_test,prediction)
cf=confusion_matrix(Y_test,prediction)
print(accuracy)
print(cf)


# In[20]:


#TPR and TNR
NB_TPR= round(9640*100/(2730+9640),2)
NB_TNR=round(8823*100/(8824+3132),2)
print(f'Naive Bayes TPR:{NB_TPR}%')
print(f'Naive Bayes TNR:{NB_TNR}%')


# ## Decision Tree(67.49%)

# In[21]:


#BlueJungleMinionsKilled, BlueAvgLevel has little importance in improving accuracy

X = df[['blueGold', 'blueMinionsKilled', 'blueChampKills', 'blueHeraldKills', 'blueDragonKills', 'blueTowersDestroyed']].values
Y = df['blue_win'].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=42)

clf = tree . DecisionTreeClassifier( criterion = 'entropy')
clf = clf .fit(X_train,Y_train)
prediction = clf .predict (X_test)

print(accuracy_score(Y_test,prediction))
print(confusion_matrix(Y_test,prediction))


# In[22]:


#TPR and TNR
DC_TPR= round(8363*100/(8363+4007),2)
DC_TNR=round(8070*100/(8070+3886),2)
print(f'Decision Tree TPR:{DC_TPR}%')
print(f'Decision Tree TNR:{DC_TNR}%')


# ## Choosing Linear Discriminant Analysis

# In[23]:


X = df[[ 'blueGold', 'blueMinionsKilled', 'blueJungleMinionsKilled', 'blueAvgLevel','blueChampKills', 'blueHeraldKills', 'blueDragonKills', 'blueTowersDestroyed']].values
Y = df['blue_win'].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=42)

# Standardize the data
scaler = StandardScaler()
scaler.fit(X_train)
X_train= scaler.transform(X_train)
X_test= scaler.transform(X_test)
X_test=np.asarray(X_test)

lda_classifier = LDA( )
lda_classifier.fit(X_train, Y_train)
prediction = lda_classifier.predict(X_test)


accuracy = lda_classifier.score (X_test, Y_test)
cf=confusion_matrix(Y_test,prediction)

print(accuracy)
print(cf)


# In[24]:


#Scale the new instance
new_instance= scaler.transform(X)
predicted= lda_classifier.predict(new_instance)
acc=accuracy_score(Y,predicted)
print(acc)


# In[25]:


df.insert(2, 'LDA_Predict', predicted)


# ## Choosing Naive Bayes

# In[26]:


X = df[['blueGold', 'blueMinionsKilled', 'blueAvgLevel', 'blueChampKills', 'blueHeraldKills', 'blueDragonKills','blueTowersDestroyed']].values
Y = df['blue_win'].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=42)
NB_classifier = GaussianNB (). fit (X_train, Y_train)
prediction = NB_classifier .predict (X_test)

accuracy=accuracy_score(Y_test,prediction)
cf=confusion_matrix(Y_test,prediction)
print(accuracy)
print(cf)


# In[27]:


# TPR and TNR
NB_TPR= 9640/(2730+9640)
NB_TNR=8823/(8824+3132)
print(NB_TPR,NB_TNR)


# In[28]:


predicted= NB_classifier .predict(X)
acc=accuracy_score(Y,predicted)
print(acc)


# In[29]:


df.insert(3, 'NB_Predict', predicted)


# In[30]:


#Try looking at the table where LDA value is different from True Value
df_dif_LDA=df[df['LDA_Predict'] != df['blue_win']]
df_dif_LDA['blueAvgLevel'].describe()


# In[31]:


df_dif_LDA=df_dif_LDA.reset_index(drop=True)


# In[32]:


#Table View
df_dif_LDA


# ## Choosing Random Forest

# In[33]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
clf.fit(X_train, Y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(Y_test, y_pred)
cf=confusion_matrix(Y_test, y_pred)
print("Accuracy:", accuracy)
print(cf)


# In[34]:


TNR_RF=8834/(8834+3122)
TPR_RF=8998/(8998+3372)
print(TNR_RF)
print(TPR_RF)


# ## Some Analysis in data insights

# In[35]:


df.groupby(['blue_win','blueHeraldKills']).size()


# In[36]:


df.groupby(['blue_win','redHeraldKills']).size()


# In[37]:


x = [0, 1, 2, 3, 4]
y1=[2422,8013,9010,5120,24]
y2=[2405,7946,8931,4759,21]

# Set the width of the bars
bar_width = 0.35

# Create a double-column bar plot
plt.bar(np.array(x) - bar_width/2, y1, bar_width, label='Blue Team', color='skyblue')
plt.bar(np.array(x) + bar_width/2, y2, bar_width, label='Red Team', color='green')

# Set plot labels
plt.xlabel('# of Herald Kills')
plt.ylabel('# of Win Games')
plt.title('Herald Kills in Winning Games')
plt.legend()

# Show the plot
plt.show()


# In[38]:


df.groupby(['blue_win','blueTowersDestroyed']).size()


# In[39]:


df.groupby(['blue_win','redTowersDestroyed']).size()


# In[40]:


x = [0,1,2,3, 4,5,6]
y1=[16668,6491,1226,178,24,2,0]
y2=[15218,7081,1488,237,35,2,1]

# Set the width of the bars
bar_width = 0.35

# Create a double-column bar plot
plt.bar(np.array(x) - bar_width/2, y1, bar_width, label='Blue Team', color='skyblue')
plt.bar(np.array(x) + bar_width/2, y2, bar_width, label='Red Team', color='yellow')

# Set plot labels
plt.xlabel('# of Towers Destroyed')
plt.ylabel('# of Win Games')
plt.title('Towers Destroyed in Winning Games')
plt.legend()

# Show the plot
plt.show()


# ## Optimize Blue Team Win(blue_win=1)

# In[41]:


df[df['blue_win']==1].groupby(['blue_win','blueHeraldKills']).size()


# In[42]:


#if blue_Herald<=2, blue has higher probability of win
df_dif_LDA[df_dif_LDA['blue_win']==1].groupby(['blue_win','blueHeraldKills']).size()


# In[43]:


df[df['blue_win']==1].groupby(['blue_win','blueTowersDestroyed']).size()


# In[44]:


#blueTowerDestroyed >=1 blue more possible to win.
df_dif_LDA[df_dif_LDA['blue_win']==1].groupby(['blue_win','blueTowersDestroyed']).size()


# ## Optimize Red Team Win (blue_win=0)

# with 0/1 herald is not enough for red_team to win, but if red herald kills>=2, higher probability to win

# In[45]:


df[df['blue_win']==0].groupby(['blue_win','redHeraldKills']).size()


# majority of red win will be when red herald kills <=1

# In[46]:


df_dif_LDA[df_dif_LDA['blue_win']==0].groupby(['blue_win','redHeraldKills']).size( )


# No obvious influence on win/loss

# In[47]:


df_dif_LDA[df_dif_LDA['blue_win']==0].groupby(['blue_win','redChampKills']).size()


# Important factor:  redTowersDestroyed>=2 , more inclined to lose; redTowerDestroyed=0, inclined to win

# In[48]:


df[(df['redTowersDestroyed']==0)|(df['redHeraldKills']>=2)]['blue_win'].value_counts()


# In[49]:


df[df['blue_win']==0].groupby(['blue_win','redTowersDestroyed']).size()


# In[50]:


df_dif_LDA[df_dif_LDA['blue_win']==0].groupby(['blue_win','redTowersDestroyed']).size()


# In[51]:


df_dif_LDA[df_dif_LDA['blue_win']==1]['redHeraldKills'].describe()


# ## Strategy 0 

# Combining LDA + NB

# In[52]:


df.insert(5,'Strategy 0',0)


# In[53]:


same=0
diff=0
correct_guess=0
i=0
for elem in df['LDA_Predict']:
    if elem==df.loc[i,'NB_Predict']:
        same+=1
        if elem==df.loc[i,'blue_win']:
            correct_guess+=1
    else:
        diff+=1
    i+=1
count=diff+same
print(same/count)
print(correct_guess/count)
print(count)

#The accuracy of combining two strategy fails. 73.43% accuracy
#Strategy: if they both have the same signal in LDA and NB, then take it, compare to the real label-blue_win. 


# In[54]:


import random

i=0
for elem in df['LDA_Predict']:
    if elem==df.loc[i,'NB_Predict']:
        df.loc[i,'Strategy 0']==elem
    else:
        df.loc[i,'Strategy 0']=np.random.choice([df.loc[i,'NB_Predict'],df.loc[i,'LDA_Predict']])
    i+=1


# In[55]:


a=df['blue_win']
b=df['Strategy 0']
acc=accuracy_score(a,b)
cf=confusion_matrix(a,b)
print(acc)
print(cf)


# In[56]:


S0_TPR=round(597*100/(597+23992),2)
S0_TNR=round(23469*100/(23469+593),2)
print(f'Strategy 0 TPR:{S0_TPR}%')
print(f'Strategy 0 TNR:{S0_TNR}%')


# In[57]:


df


# ## Strategy I

# According to the stats data from table(LDA wrongly predicted) to find the pattern and implement

# In[58]:


df.insert(6,'Strategy I',0)


# In[59]:


i=0
for e in df['LDA_Predict']:
    if e==0:     
        if (df.loc[i,'redTowersDestroyed']==0) | (df.loc[i,'redHeraldKills']>=2):
            df.loc[i,'Strategy I']=1
        else:
            df.loc[i,'Strategy I']=0
    elif e==1:
        if (df.loc[i,'blueTowersDestroyed']<1) | (df.loc[i,'blueHeraldKills']>=2):
            df.loc[i,'Strategy I']=0
        else:
            df.loc[i,'Strategy I']=1
    i+=1


# In[60]:


# Calculate Accuracy and Confusion Matrix
a=df['blue_win']
b=df['Strategy I']
acc=accuracy_score(a,b)
cf=confusion_matrix(a,b)
print(acc)
print(cf)


# In[61]:


S1_TPR=round(6340*100/(6340+18249),2)
S1_TNR=round(6727*100/(6727+17335),2)
print(f'Strategy I TPR:{S1_TPR}%')
print(f'Strategy I TNR:{S1_TNR}%')


# In[62]:


df_dif_LDA[df_dif_LDA['blue_win']==0][['blueHeraldKills']].value_counts()


# In[63]:


df_dif_LDA[df_dif_LDA['blue_win']==1][['blueHeraldKills']].value_counts()


# In[64]:


df_dif_LDA[df_dif_LDA['blue_win']==0][['blueTowersDestroyed']].value_counts()


# In[65]:


df_dif_LDA[df_dif_LDA['blue_win']==1][['blueTowersDestroyed']].value_counts()


# ## Strategy II

# According to the stats data from table(LDA wrongly predicted) and other variables [ChampKills][Heraldkills][TowersDestroyed]
# make a strategy to implement

# In[66]:


df.insert(7,'Strategy II',0)


# In[67]:


i=0
for e in df['LDA_Predict']:
    if e==0:     
        if df.loc[i,'blueChampKills']< df.loc[i,'redChampKills']:
            df.loc[i,'Strategy II']=e
        if (df.loc[i,'blueTowersDestroyed']==1) | (df.loc[i,'blueHeraldKills']==1):
            df.loc[i,'Strategy II']=1    
        else:
            df.loc[i,'Strategy II']==e
            
    elif e==1:
        if df.loc[i,'blueChampKills']> df.loc[i,'redChampKills']:
            df.loc[i,'Strategy II']=e
        if (df.loc[i,'blueTowersDestroyed']==0) | (df.loc[i,'blueHeraldKills']==2):
            df.loc[i,'Strategy II']=0
        else:
            df.loc[i,'Strategy II']==e
    i+=1


# In[68]:


# Calculate Accuracy and Confusion Matrix
a=df['blue_win']
b=df['Strategy II']
acc=accuracy_score(a,b)
cf=confusion_matrix(a,b)
print(acc)
print(cf)


# In[69]:


S2_TPR=round(6111*100/(6111+18478),2)
S2_TNR=round(11726*100/(11726+12336),2)
print(f'Strategy II TPR:{S2_TPR}%')
print(f'Strategy II TNR:{S2_TNR}%')


# ## Strategy III

# Based on LDA Predict, and add [ChampKills] [Gold] variables to implement

# In[70]:


df.insert(8,'Strategy III',0)


# In[71]:


i=0
for e in df['LDA_Predict']:
    if e==0:     
        if (df.loc[i,'blueChampKills']> df.loc[i,'redChampKills']) & (df.loc[i,'redGold']<= df.loc[i,'blueGold']):
            df.loc[i,'Strategy III']=1
        else:
            df.loc[i,'Strategy III']=e
            
    elif e==1:
        if (df.loc[i,'blueChampKills']< df.loc[i,'redChampKills']) &(df.loc[i,'redGold']>= df.loc[i,'blueGold']):
            df.loc[i,'Strategy III']=0
        else:
            df.loc[i,'Strategy III']=e
    i+=1


# In[72]:


# Calculate Accuracy and Confusion Matrix
a=df['blue_win']
b=df['Strategy III']
acc=accuracy_score(a,b)
cf=confusion_matrix(a,b)
print(acc)
print(cf)


# In[73]:


S3_TPR=round(19412*100/(19412+5177),2)
S3_TNR=round(18507*100/(18507+5555),2)
print(f'Strategy III TPR:{S3_TPR}%')
print(f'Strategy III TNR:{S3_TNR}%')


# In[74]:


df


# In[75]:


df.insert(9,'Strategy IV',0)


# In[76]:


i=0
for e in df['Strategy III']:
    if df.loc[i,'KNN']==1:
        df.loc[i,'Strategy IV']=1
    else:
        df.loc[i,'Strategy IV']=e
    i+=1


# In[77]:


a=df['blue_win']
b=df['Strategy IV']
acc=accuracy_score(a,b)
cf=confusion_matrix(a,b)
print(acc)
print(cf)


# In[78]:


S4_TPR=round(24579*100/(24579+10),2)
S4_TNR=round(522*100/(522+23540),2)
print(f'Strategy IV TPR:{S4_TPR}%')
print(f'Strategy IV TNR:{S4_TNR}%')


# In[79]:


df_blue= df_blue.reset_index(drop=True)


# ## Get close to the winning team

# In[80]:


#For Blue Team wins, 16673 of 24589 has higher average level.
i=-1
n=0
for e in df_blue['blueAvgLevel']:
    i+=1
    if e> df_blue.loc[i,'redAvgLevel']:
        n+=1
    else:
        continue
print(n/24589)


# In[81]:


#For Blue Team wins, 16673 of 24589 has higher Champion Kills.
i=-1
n=0
for e in df_blue['blueChampKills']:
    i+=1
    if e> df_blue.loc[i,'redChampKills']:
        n+=1
    else:
        continue
print(n/24589)


# In[82]:


#For Blue Team wins, 16673 of 24589 has higher Minions Kills.
i=-1
n=0
for e in df_blue['blueMinionsKilled']:
    i+=1
    if e> df_blue.loc[i,'redMinionsKilled']:
        n+=1
    else:
        continue
print(n/24589)


# In[83]:


#For Blue Team wins, 59% has higher jungle mininions killed

i=-1
n=0
for e in df_blue['blueJungleMinionsKilled']:
    i+=1
    if e> df_blue.loc[i,'redJungleMinionsKilled']:
        n+=1
    else:
        continue
print(n/24589)


# In[84]:


#For Blue Team wins, 19354 of 24589 has higher Herald.

i=-1
n=0
for e in df_blue['blueHeraldKills']:
    i+=1
    if e> df_blue.loc[i,'redHeraldKills']:
        n+=1
    else:
        continue
print(n/24589)


# In[85]:


#For Blue Team wins, 19354 of 24589 has higher gold.

i=-1
n=0
for e in df_blue['blueGold']:
    i+=1
    if e> df_blue.loc[i,'redGold']:
        n+=1
    else:
        continue
print(n/24589)


# In[86]:


#For Blue Team wins, 19354 of 24589 has less towers destroyed.

i=-1
n=0
same=0
dif=0
for e in df_blue['blueTowersDestroyed']:
    i+=1
    if e< df_blue.loc[i,'redTowersDestroyed']:
        n+=1
    elif e==df_blue.loc[i,'redTowersDestroyed']:
        same+=1
    else:
        dif+=1
        continue
print(n,same,dif)
print(n/(n+same+dif))


# In[87]:


x=['blueAvgLevel','blueChampKills','blueMinionsKilled','blueJungleMinionsKilled','blueHeraldKills','blueGold','blueTowersDestroyed']
y=[67.81,74.19,65.91,58.97,60.94,78.71,59.26]


# In[88]:


plt.bar(x, y, color='green')

# Set plot labels
plt.xlabel('Features')
plt.ylabel('Win rates(%)')
plt.title('Measuring Features in Winning Team')
plt.xticks(rotation=45, ha='right')
plt.yscale('log')
plt.show()

