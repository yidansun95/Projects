#!/usr/bin/env python
# coding: utf-8

# ## Final Project

# Name: Yidan Sun
# 
# Project Proposal:Account Sign in/Sign up, User basic info fill-in,  Keep tracking of users dish record, personal info, setting fitness goal, get report of BMI, TDEE,DEFICIT,etc and get diet recommendation.

# In[1]:


import math
import os


# In[95]:


total_dish_record=[]


# In[37]:


user_info=[]


# In[2]:


user_data={}
user_info=[]


# ## User Sign-up/ Sign-in

# In[129]:


print('Welcome to Sunshine Bear Diet APP')
print('-'*87)

#Create Account: Sign-up/ Sign-in
while True:
    try:
        SIGN=int(input('Please Sign in.If you are new user, please enter 0\nFor existing users please enter 1:\n'))
        break
    except:
        print('Please enter 0 for new user , 1 for existing user')
        
#for new-user to sign-up    
if SIGN==0:
    print('='*30+'Create your Profile now'+'='*40)
    try:
        while True:
            user_account=input('Enter your Account(Hint:should contain at least 5 words)\n')
            if user_account in user_data.keys():
                print('User Account Already Exist,Try another one!\n')
            elif user_account not in user_data.keys() and len(user_account)>=5:
                break
            else:
                print('Please enter correct account name\n')
        while True:
            user_password=input('Enter your Password:(Hint:should contain 8~20 spaces with letters and digits combined)\n')
            number=0
            alpha=0
            for e in user_password:
                if e.isdigit():
                    number+=1
                elif e.isalpha():
                    alpha+=1
            if (number>=1) and (alpha>=1) and len(user_password)>=8 and len(user_password)<=20:
                print('\n'+'*'*30+'Account Successfully Created!'+'*'*37)
                print('='*30+'Welcome! Let\'s get started'+'='*40)
                user_data[user_account]=user_password
                break
            print('Please enter the correct password!\n')
            
    except ValueError:
        print('Something goes wrong, please re-enter!')

        
#For existing-user to sign-in      
else:
    print('='*30+'Sign in now'+'='*44)
    while True:
        user_account=input('Enter your Account: ')
        user_password=input('Enter your Password: ')
        if user_account in user_data.keys():
            if user_password==user_data[user_account]:
                print('='*30+'Welcome back,{}~'.format(user_account)+'='*30)
                break
            else:
                print('Either your account or password is incorrect, try again!')
        else:
            print('Either your account or password is incorrect, try again!')


# In[130]:


user_data


# In[131]:


#Tuple of user account + password
account_password = tuple(user_data.items())


# ## For New Users to generate profile

# In[106]:


user_data


# In[133]:


#for new users to begin setting profile
if SIGN==0:
    while True:
        user_account=input('Enter your account: ')
        if user_account in user_data.keys():
            break
        else:
            print('There is no such account existing.')
            
    #Ask for Gender
    while True: 
        try:
            user_gender=input('What is your gender? \na.Female\nb.Male\nc.Transgender\nd.Prefer not to say\n')
            print('-'*80)
            user_gender=user_gender.strip()
            if user_gender=='a':
                user_gender='Female'
                break
            elif user_gender=='b':
                user_gender='Male'
                break
            elif user_gender=='c':
                user_gender='Transgender'
                break
            elif user_gender=='d':
                user_gender='NA'
                break
            else:
                raise ValueError
                
        except ValueError:
            print(':( Oops,only letter permitted: a/b/c/d')

    #Ask for Age
    while True:
        try:
            user_age=int(input('\nWhat is your age?\n'))
            print('-'*80)
            user_age=int(user_age)
            break
        except ValueError:
            print(':( Oops,wrong input,only integer,try again')
        

    #Ask for Height
    while True:
        try:
            user_height=float(input('\nWhat is your height (in cm)?\n'))
            print('-'*80)
            user_height=int(user_height)
            break
        except ValueError:
            print(':( Oops,only numbers permitted')

    #Ask for Weight
    while True:
        try:
            user_weight=float(input('\nWhat is your Weight (in kg)?\n'))
            print('-'*80)
            user_weight=float(user_weight)
            break
        except ValueError:
            print(':( Oops,only numbers permitted')
            
            
    #Ask for exercise frequency
    while True:
        try:
            user_exercise=input('\nHow do you rate your exercise frequency? \na.little\nb.light\nc.moderate\nd.hard\ne.extreme\n')
            print('-'*80)
            user_exercise=user_exercise.strip()
            if user_exercise=='a':
                user_exercise='little'
                break
            elif user_exercise=='b':
                user_exercise='light'
                break
            elif user_exercise=='c':
                user_exercise='moderate'
                break
            elif user_exercise=='d':
                user_exercise='hard'
                break
            elif user_exercise=='e':
                user_exercise='extreme'
                break
            else:
                raise ValueError
        except ValueError:
            print(':( Oops,only letters permitted: a/b/c/d/e')

    #Ask for Health Goal   
    while True:
        try:
            user_goal=input('\nWhat is your goal? \na.Lose Weight \nb.Gain Weight\nc.Increase Muscles \nd.Others\n')
            print('-'*80)
            user_goal=user_goal.strip()
            if user_goal in ['a','b','c','d']:
                break
            else:
                raise ValueError
        except ValueError:
            print(':( Oops,only letters permitted: a/b/c/d')
    
    #BMI measure and printout
    user_BMI=round((user_weight/((user_height/100)**2)),2)        
    print('Your current BMI is: {} and the Ideal BMI is {}'.format(user_BMI,'18.5~24.9'))
    if user_BMI<18.5:
        print('You are below average weight, consider gaining some weights')
    elif user_BMI<=24.9 and user_BMI>=18.5:
        print('You are fit! Keep going!')
    else:
        print('You are above average weight, consider keeping diet now~')
        print('-'*80)
        
    #Ask for Ideal Weight
    while True:
        try:
            user_ideal_weight=float(input('\nWhat is your Ideal Weight(in kg)?\n'))
            print('-'*80)
            break
        except ValueError:
            print(':( Oops,only digits permitted')
            
    usr=[user_account,user_age,user_gender,user_height,user_weight,user_exercise,user_BMI,user_ideal_weight]
    user_info.append(usr)


# In[134]:


#Recorded all users input data
user_info


# ## For Existing Users to record dish

# In[138]:


## Can only record for certain day
num=0
total_calorie=0
dish_record=[]

while True:
    user_account=input('Enter your Account: ')
    if user_account in user_data.keys():
        break
    else:
        print('There is no such account')
        
user_date=input('Enter date you want to record(format as：yyyy-mm-dd): ')
while True:
    print('Lets record dish taken, and calories for each dish!')
    user_dish=input('\nAdd your dish name:  ')
    user_calorie=int(input('\nDish Calories?(kcals)  '))
    try:
        ask=input('do you want to add more dish?\na.Yes\nb.No,I\'m done.\n')
        if ask=='a':
            if num<=50:
                usr=[user_account,user_date,user_dish,user_calorie]
                total_calorie+=user_calorie
                dish_record.append(usr)
            else:
                print('You have achieved limit of adding dish')
                print('You are all set for adding dish belows and check out your calories intaken suggestion for today!:')
                print(dish_record)
                
        elif ask=='b':
            usr=[user_account,user_date,user_dish,user_calorie]
            total_calorie+=user_calorie
            dish_record.append(usr)
            print()
            print(f'***Thank you, {user_account}***')
            print(f'***You are all set,check your calories intaken and suggestion for {user_date}!***')
            print(f'\nDish you have added({total_calorie}kcal):')
            print(dish_record)
            break
        else:
            raise ValueError
    except ValueError:
        print(':( Oops,only letters permitted: a/b/c/d,please re-enter the dish again')


# In[139]:


#Show the dish recorded array
total_dish_record.append(dish_record)
total_dish_record


# In[140]:


flat_data = [item for sublist in total_dish_record for item in sublist]

# Create a DataFrame
columns = ['account', 'date', 'dish', 'calories']
dish_table = pd.DataFrame(flat_data, columns=columns)
dish_table


# In[141]:


## Check what she specific user record
dish_table[dish_table['account']=='carolsyd']


# In[142]:


## Check what she specific user and date dish
dish_table[(dish_table['account']=='carolinesyd') &(dish_table['date']=='20231207')]


# ##### user_info=(user_age,user_gender,user_height,user_weight,user_exercise,user_BMI,user_ideal_weight)

# In[56]:


user_data


# In[143]:


## In database we have all records
table=pd.DataFrame(user_info,columns=['account','age','gender','height','weight','exercise','BMI','ideal weight'])
table


# In[61]:


from my_class import CalorieWeight


# In[120]:


user_info


# In[144]:


# Calorie Weight parameters: Height=168,Weight=55,Age=18,Calorie=2000,Exercise='moderate',Gender='Female'

carolsyd=CalorieWeight(user_info[0][3],user_info[0][4],user_info[0][1],total_calorie,user_info[0][5],user_info[0][2])
print(carolsyd)


# ## Test for Public/Private Method

# In[145]:


#Test1: Inaccessible to private method by this method
a=CalorieWeight()
a.__DEFICIT()


# In[146]:


#Test2: Access private method
a=CalorieWeight()
a._CalorieWeight__DEFICIT()


# In[ ]:




