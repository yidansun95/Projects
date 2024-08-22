#!/usr/bin/env python
# coding: utf-8

# ## CalorieWeight Class

# In[1]:


class CalorieWeight(object):
    '''Calculate user Basal Metabolic Rate an their Weight Deficit'''
    '''Gender input F/M/T/NA'''
    '''Height(cm),Weight(kg),Age(yr),Calorie(kcal)'''
    '''Exercise: little/light/moderate/hard/extreme'''
    
    def __init__(self,Height=168,Weight=55,Age=18,Calorie=2000,Exercise='moderate',Gender='Female'):
        self.height=Height
        self.weight=Weight
        self.age=Age
        self.calorie=Calorie
        self.gender=Gender
        self.exercise=Exercise
    
    def __str__(self):
        bmr_result =self.BMR()
        tdee_result =self.TDEE()
        deficit_result =self.__DEFICIT()
        
        report = f'\nYour Info:\nHeight: {self.height}\nWeight: {self.weight}\nTakenCalorie: {self.calorie}\n\nYour Report:\nBMR(kcal): {bmr_result}\nTDEE(kcal): {tdee_result}\nDeficit(kcal): {deficit_result}'
        return report
        
    #Basal Metabolic Rate
    def BMR(self):
        if self.gender=='Male':
              bmr=round(88.362+(13.397*self.weight)+(4.799*self.height)-(5.677*self.age))
        else:
              bmr=round(447.593+(9.247*self.weight)+(3.098*self.height)-(4.330*self.age))
        
        return(bmr)
    
    
    '''Exercise:little(1.2)/light(1.375)/moderate(1.55)/hard(1.725)/extreme(1.9)'''
    #Total Daily Energy Expenditure    
    def TDEE(self):
        if self.exercise=='little':
            tdee=round(self.BMR()*1.2)
        elif self.exercise=='light':
            tdee=round(self.BMR()*1.375)
        elif self.exercise=='moderate':
            tdee=round(self.BMR()*1.55)
        elif self.exercise=='hard':
            tdee=round(self.BMR()*1.725)
        elif self.exercise=='extreme':
            tdee=round(self.BMR()*1.9)
        
        return(tdee)

    def __DEFICIT(self):
        deficit=round(self.TDEE()-self.calorie)
        if deficit>0:
            print(f'###Warning:{abs(deficit)} more calories are taken###')
        if deficit<0:
            print(f'Good Job! you still can eat {deficit} more calories are taken')
        if deficit==0:
            print('Well done,no energy deficit.')
        
        return(deficit)


# ## User Info Class

# In[1]:


class UserInfo(object):
    def __init__(self,Account='',Password='',id=0):
        self.account=Account
        self.password=Password
        self.id=id
    
    def __str__(self):
        return('Account: '+str(self.account)+'\nPassword: '+str(self.password)+'\nID: '+str(self.id))
    
    def gender(self,gender):
        return("Your Gender is: "+gender)
    
    def age(self,age):
        return(f'Your Age is: {age}')
    
    


# In[ ]:




