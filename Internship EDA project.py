#!/usr/bin/env python
# coding: utf-8

# ### Analysis on AEMo data

# ### Introduction

# ### Description of the Dataset

# The dataset was released by Aspiring Minds from the Aspiring Mind Employment Outcome 2015 (AMEO). The study is primarily limited  only to students with engineering disciplines. The dataset contains the employment outcomes of engineering graduates as dependent variables (Salary, Job Titles, and Job Locations) along with the standardized scores from three different areas – cognitive skills, technical skills and personality skills. The dataset also contains demographic features. The dataset  contains  around  40 independent variables and 4000 data points. The independent variables are both continuous and categorical in nature. The dataset contains a unique identifier for each candidate.

# ### Objective

# The target variable is salary.
# . Finding the relationship between the target variable and the other independent variables.
# . Finding the outliers.
# . Interpreting insights based on Univariate and Bivariate analysis

# ### Importing the data

# In[1]:


import pandas as pd
df = pd.read_csv(r"C:\Users\jaswanth talla\Downloads\data.xlsx - Sheet1.csv")
df1 = df.copy()


# In[2]:


pd.set_option('display.max_columns',100)


# In[3]:


df1


# # Head, Shape and Description

# In[4]:


df1.head()


# In[5]:


df1.shape


# In[6]:


df1.info()


# In[7]:


df1.describe()


# In[8]:


df1.isnull().sum()


# In[9]:


df1.duplicated().sum()


# In[10]:


df1.columns


# In[11]:


df1.nunique()


# In[12]:


df1 = df1.drop(columns = ['Unnamed: 0', 'ID', 'CollegeID', 'CollegeCityID'])
df1.head()


# As the unnamed: 0 is not useful and the columns ID, CollegeID, CollegeCityID are the ID's of the individuals we cannot find the relationship between the target variable salary and ID's so we are droping them.

# # Datatypes Conversion

# Converting DOL(Date of Leaving) & DOJ(Date of Joining) as datetime

# DOL - Date of Leaving.

# As the servey is conducted in 2015, the respondents who have filled the column DOL as present as assumed to leave the organization in 2015 only. So we replace present as 2015-12-31(end of 2015) as DOL.

# In[13]:


df1['DOL'].replace('present','2015-12-31', inplace = True)


# In[14]:


df1['DOL'] = pd.to_datetime(df1['DOL'])
df1['DOJ'] = pd.to_datetime(df1['DOJ'])


# In[15]:


df1.head()


# Converting the categorical_columns as category

# In[16]:


categorical_columns = ['Designation' ,'JobCity', 'Gender', '10board', '12board', 'CollegeTier', 'Degree' ,'Specialization', 'CollegeCityTier', 'CollegeState']
df1[categorical_columns] = df1[categorical_columns].astype('category')


# In[17]:


df1.info()


#  Checking DOL & DOJ
# 

# In[18]:


dates = df1[(df1['DOL'] < df1['DOJ'])].shape[0]
print(f'Dol is earlier than DOJ for{dates} observations.')
print(df1.shape)


# The DOJ should be greater than DOL so these 40 observations should be considered as outliers, drop them.

# In[19]:


df1 = df1.drop(df1[~(df1['DOL'] > df1['DOJ'])]. index)


# In[20]:


df1.shape


# Checking the results are in percentages and not in CGPA

# In[21]:


print((df1['10percentage'] <=10).sum())
print((df1['12percentage'] <=10).sum())
print((df1['collegeGPA'] <=10).sum())


# In[22]:


df1.loc[df1['collegeGPA'] <= 10, 'collegeGPA'].index


# In[23]:


df1.loc[df1['collegeGPA'] <= 10, 'collegeGPA'] = (df1.loc[df1['collegeGPA'] <= 10, 'collegeGPA']/10)*100


# In[24]:


df1


# In[25]:


print((df1==0).sum()[(df1==-0).sum() >0])


# In[26]:


(df1==-1).sum()[(df1==-1).sum()>0]/len(df1)*100


# In[27]:


df1 = df1.drop(columns = ['MechanicalEngg', 'ElectricalEngg', 'TelecomEngg', 'CivilEngg'])


# In[28]:


df1


# In[29]:


import numpy as np
df1['10board'] = df1['10board'].replace({'0':np.nan})
df1['12board'] = df1['12board'].replace({'0':np.nan})
df1['GraduationYear'] = df1['GraduationYear'].replace({0:np.nan})
df1['JobCity'] = df1['JobCity'].replace({'-1':np.nan})
df1['Domain'] = df1['Domain'].replace({-1:np.nan})
df1['ElectronicsAndSemicon'] = df1['ElectronicsAndSemicon'].replace({-1:0})
df1['ComputerProgramming'] = df1['ComputerProgramming'].replace({-1:np.nan})
df1['ComputerScience'] = df1['ComputerScience'].replace({-1:np.nan})


# In[30]:


df1['10board'].fillna(df1['10board'].mode()[0], inplace=True)
df1['12board'].fillna(df1['12board'].mode()[0], inplace=True)
df1['GraduationYear'].fillna(df1['GraduationYear'].mode()[0], inplace=True)
df1['JobCity'].fillna(df1['JobCity'].mode()[0], inplace=True)


# In[31]:


df1


# In[32]:


df1['Domain'].fillna(df1['Domain'].mode()[0], inplace=True)
df1['ComputerProgramming'].fillna(df1['ComputerProgramming'].mode()[0], inplace=True)


# In[33]:


df1


# In[34]:


df1['Gender'].replace({'f': 'Female', 'm':'Male'}, inplace = True)


# # Feature Engineering

# Creating a column name 'AGE'

# In[35]:


df1['DOB'] = pd.to_datetime(df1['DOB'])
df1['Age'] = 2015 - df1['DOB'].dt.year
df1.head()


# The dataset is released on 2015 so we calculated the age accordingly.

# calculating the working years

# In[36]:


working_days = (df1['DOL'] - df1['DOJ']).dt.days
days = 365
working_years = working_days / days
df1['working_years'] = working_years


# In[37]:


df1['working_years'] = df1['working_years'].round(2)


# In[38]:


df1


# In[39]:


len(df1[(df1['GraduationYear'] > df1['DOJ']. dt.year)]. index)


# In[40]:


df1 = df1.drop(df1[(df1['GraduationYear'] > df1['DOJ']. dt.year)]. index)


# In[41]:


df1


# Calculate CDF

# In[42]:


def cdf(data):
    x = np.sort(data)
    y = np.arange(1, len(x)+1)/len(x)
    return x, y


# # Univariate Analysis

# In[43]:


import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
get_ipython().run_line_magic('matplotlib', 'inline')


# Numerical Features

# # Summary plots  
# for working_years, salary, 10th percentage, 12th percentage, CollegeGPA, English, Logical, Quant, Computer Programming, Electronics & Semiconductors, Age. 

# In[44]:


# summary plot for working_years

plt.figure(figsize = (5,4))
df1['working_years'].describe()[1:].plot(alpha = 0.8,
                                        marker = 'D', markersize = 8)
plt.title('Summary Statistics for Tenure')
plt.xlabel('Statistical Measures')
plt.tight_layout()
plt.show()


# The maximum respondents have experience of 6 years.

# In[45]:


# summary plot for salary

plt.figure(figsize = (5,4))
df1['Salary'].describe()[1:].plot(alpha = 0.8,
                                        marker = 'D', markersize = 8)
plt.title('Summary Statistics for Salary')
plt.xlabel('Statistical Measures')
plt.tight_layout()
plt.show()


# Their is variations in salary.

# In[46]:


# summary plot for 10th percentage

plt.figure(figsize = (5,4))
df1['10percentage'].describe()[1:].plot(alpha = 0.8,
                                        marker = 'D', markersize = 8)
plt.title('Summary Statistics for 10percentage')
plt.xlabel('Statistical Measures')
plt.tight_layout()
plt.show()


# Most of the respondents have achieved 80%.

# In[47]:


# summary plot for 12th percentage

plt.figure(figsize = (5,4))
df1['12percentage'].describe()[1:].plot(alpha = 0.8,
                                        marker = 'D', markersize = 8)
plt.title('Summary Statistics for 12percentage')
plt.xlabel('Statistical Measures')
plt.tight_layout()
plt.show()


# Many respondents have achieved 75% score.

# In[48]:


# summary plot for CollegeGPA

plt.figure(figsize = (5,4))
df1['collegeGPA'].describe()[1:].plot(alpha = 0.8,
                                        marker = 'D', markersize = 8)
plt.title('Summary Statistics for collegeGPA')
plt.xlabel('Statistical Measures')
plt.tight_layout()
plt.show()


# More than 50% of the respondents have achieved 75% score.

# In[49]:


# summary plot for English

plt.figure(figsize = (5,4))
df1['English'].describe()[1:].plot(alpha = 0.8,
                                        marker = 'D', markersize = 8)
plt.title('Summary Statistics for English')
plt.xlabel('Statistical Measures')
plt.tight_layout()
plt.show()


# 50% of the respondents have achieved scores more than 500.

# In[50]:


# summary plot for Logical

plt.figure(figsize = (5,4))
df1['Logical'].describe()[1:].plot(alpha = 0.8,
                                        marker = 'D', markersize = 8)
plt.title('Summary Statistics for Logical')
plt.xlabel('Statistical Measures')
plt.tight_layout()
plt.show()


# 50% of the respondents have achieved more than or equal to the score of 500.

# In[51]:


# summary plot for Quant

plt.figure(figsize = (5,4))
df1['Quant'].describe()[1:].plot(alpha = 0.8,
                                        marker = 'D', markersize = 8)
plt.title('Summary Statistics for Quant')
plt.xlabel('Statistical Measures')
plt.tight_layout()
plt.show()


# Most of the respondents have achieved the score of 500 and more.

# In[52]:


# summary plot for Computer Programming

plt.figure(figsize = (5,4))
df1['ComputerProgramming'].describe()[1:].plot(alpha = 0.8,
                                        marker = 'D', markersize = 8)
plt.title('Summary Statistics for ComputerProgramming')
plt.xlabel('Statistical Measures')
plt.tight_layout()
plt.show()


# 75% of the respondents have scored approximately 500 and more.

# In[53]:


# summary plot for Electronics And Semicon

plt.figure(figsize = (5,4))
df1['ElectronicsAndSemicon'].describe()[1:].plot(alpha = 0.8,
                                        marker = 'D', markersize = 8)
plt.title('Summary Statistics for ElectronicsAndSemicon')
plt.xlabel('Statistical Measures')
plt.tight_layout()
plt.show()


# 75% respondents have scored less than 250.

# In[54]:


# summary plot for Age

plt.figure(figsize = (5,4))
df1['Age'].describe()[1:].plot(alpha = 0.8,
                                        marker = 'D', markersize = 8)
plt.title('Summary Statistics for Age')
plt.xlabel('Statistical Measures')
plt.tight_layout()
plt.show()


# most of the respondents come under the age group of 25-35.

# # Histogram 
# for working_years, salary, 10th percentage, 12th percentage, CollegeGPA, English, Logical, Quant, Computer Programming, Electronics & Semiconductors, Age. 

# In[55]:


# Histogram for working_years

plt.figure(figsize = (6,4))
plt.hist(df1['working_years'],
        ec = 'k',
        bins = np.arange(0, df1['working_years'].max()+0.5, 0.5),
        color = 'slateblue',
        alpha = 0.7,
        label = f"Skewness : {round(df1['working_years'].skew(),2)}",
        density = True)
plt.xticks(ticks = np.arange(0, df1['working_years'].max()+0.5, 0.5))
plt.xlabel('Experience')
plt.ylabel('Density')
plt.axvline(df1['working_years'].mean(), label = f"Mean: {round(df1['working_years'].mean(),2)}",
            linestyle = '--',
           color = 'green', linewidth = 2)
plt.axvline(df1['working_years'].median(), label = f"Median: {round(df1['working_years'].median(),2)}",
            linestyle = ':',
           color = 'k', linewidth = 2)
plt.axvline(df1['working_years'].mode()[0], label = f"Mode: {round(df1['working_years'].mode()[0],2)}"
            , linestyle = '-.',
           color = 'red', linewidth = 2)
sns.kdeplot(df1['working_years'])
plt.legend()
plt.show()


# The data is right skewed(positive).The mean is 1.75. The mean, median and mode lies closely to each other.The peak is at 1.0 so the maximum respondents have experience of 1 year.

# In[56]:


#Salary

bins = np.arange(0, df1['Salary'].max()+250000, 250000)
plt.figure(figsize = (10,6))
plt.hist(df1['Salary'], ec = 'k',
        bins = bins,
        label = f"Skewness : {round(df1['Salary'].skew(),2)}",
        alpha = 0.7,
        density = True)
plt.xticks(bins)
plt.xlabel('Salary', size = 15)
plt.ylabel('Density', size = 15)

plt.axvline(df1['Salary'].mean(), label = f"Mean: {round(df1['Salary'].mean(),2)}"
            , linestyle = '-.',
           color = 'red', linewidth = 2)
plt.axvline(df1['Salary'].median(), label = f"Median: {round(df1['Salary'].median(),2)}"
            , linestyle = '-.',
           color = 'green', linewidth = 2)
plt.axvline(df1['Salary'].mode()[0], label = f"Mode: {round(df1['Salary'].mode()[0],2)}"
            , linestyle = '-.',
           color = 'k', linewidth = 2)
sns.kdeplot(df1['Salary'])
plt.legend()
plt.show()


# With a skewness score of around 6, the data shows strong positive skewness, which deviates from a normal distribution. The mean, median, and mode—three measures of central tendency—are about equal.

# In[57]:


#10percentage

bins = np.arange(df1['10percentage'].min(), df1['10percentage'].max()+df1['10percentage'].std(),
                 df1['10percentage'].std()/3)
plt.figure(figsize = (15,6))
plt.hist(df1['10percentage'], ec = 'k',
        bins = bins,
        label = f"Skewness : {round(df1['10percentage'].skew(),2)}",
        alpha = 0.7,
        density = True)
plt.xticks(bins)
plt.xlabel('10th Percentage', size = 15)
plt.ylabel('Density', size = 15)

plt.axvline(df1['10percentage'].mean(), label = f"Mean: {round(df1['10percentage'].mean(),2)}"
            , linestyle = '-.',
           color = 'red', linewidth = 2)
plt.axvline(df1['10percentage'].median(), label = f"Median: {round(df1['10percentage'].median(),2)}"
            , linestyle = '-.',
           color = 'green', linewidth = 2)
plt.axvline(df1['10percentage'].mode()[0], label = f"Mode: {round(df1['10percentage'].mode()[0],2)}"
            , linestyle = '-.',
           color = 'k', linewidth = 2)
sns.kdeplot(df1['10percentage'])
plt.legend()
plt.show()


# Few pupils have low percentages, as seen by the histogram, with most falling between 75% and 90%. The average score is roughly 77%, with the peak frequency occurring at 78%.

# In[58]:


# 12percentage

bins = np.arange(df1['12percentage'].min(), df1['12percentage'].max()+df1['12percentage'].std(),
                 df1['12percentage'].std()/3)
plt.figure(figsize = (15,8))
plt.hist(df1['12percentage'], ec = 'k',
        bins = bins,
        label = f"Skewness : {round(df1['12percentage'].skew(),2)}",
        alpha = 0.7,
        density = True)
plt.xticks(bins)
plt.xlabel('12th Percentage', size = 15)
plt.ylabel('Density', size = 15)

plt.axvline(df1['12percentage'].mean(), label = f"Mean: {round(df1['12percentage'].mean(),2)}"
            , linestyle = '-.',
           color = 'red', linewidth = 2)
plt.axvline(df1['12percentage'].median(), label = f"Median: {round(df1['12percentage'].median(),2)}"
            , linestyle = '-.',
           color = 'green', linewidth = 2)
plt.axvline(df1['12percentage'].mode()[0], label = f"Mode: {round(df1['12percentage'].mode()[0],2)}"
            , linestyle = '-.',
           color = 'k', linewidth = 2)
sns.kdeplot(df1['12percentage'])
plt.legend()
plt.show()


# The histogram illustrates a scarcity of students with low percentages, with the majority scoring between 69% and 84%. The peak frequency occurs at 70%, and the average score is around 74%.

# In[59]:


# collegeGPA

bins = np.arange(df1['collegeGPA'].min(), df1['collegeGPA'].max()+df1['collegeGPA'].std(),
                 df1['collegeGPA'].std()/2)
plt.figure(figsize = (15,8))
plt.hist(df1['collegeGPA'], ec = 'k',
        bins = bins,
        label = f"Skewness : {round(df1['collegeGPA'].skew(),2)}",
        alpha = 0.7,
        density = True)
plt.xticks(bins)
plt.xlabel('College GPA', size = 15)
plt.ylabel('Density', size = 15)

plt.axvline(df1['collegeGPA'].mean(), label = f"Mean: {round(df1['collegeGPA'].mean(),2)}"
            , linestyle = '-.',
           color = 'red', linewidth = 2)
plt.axvline(df1['collegeGPA'].median(), label = f"Median: {round(df1['collegeGPA'].median(),2)}"
            , linestyle = '-.',
           color = 'green', linewidth = 2)
plt.axvline(df1['collegeGPA'].mode()[0], label = f"Mode: {round(df1['collegeGPA'].mode()[0],2)}"
            , linestyle = '-.',
           color = 'k', linewidth = 2)
sns.kdeplot(df1['collegeGPA'])
plt.legend()
plt.show()


# Most of the students got GPAs in the range of 63% to 78%. The average GPA was 74%, while the largest frequency of pupils received scores of 70%.

# In[60]:


# English

bins = np.arange(df1['English'].min(), df1['English'].max()+df1['English'].std(),
                 df1['English'].std()/2)
plt.figure(figsize = (15,8))
plt.hist(df1['English'], ec = 'k',
        bins = bins,
        label = f"Skewness : {round(df1['English'].skew(),2)}",
        alpha = 0.7,
        density = True)
plt.xticks(bins)
plt.xlabel('English Scores', size = 15)
plt.ylabel('Density', size = 15)

plt.axvline(df1['English'].mean(), label = f"Mean: {round(df1['English'].mean(),2)}"
            , linestyle = '-.',
           color = 'red', linewidth = 2)
plt.axvline(df1['English'].median(), label = f"Median: {round(df1['English'].median(),2)}"
            , linestyle = '-.',
           color = 'green', linewidth = 2)
plt.axvline(df1['English'].mode()[0], label = f"Mode: {round(df1['English'].mode()[0],2)}"
            , linestyle = '-.',
           color = 'k', linewidth = 2)
sns.kdeplot(df1['English'])
plt.legend()
plt.show()


# The bulk of the scores fell within the range of 389 to 545. The peak occurred at 475, with an average score of 502.

# In[61]:


# Logical

bins = np.arange(df1['Logical'].min(), df1['Logical'].max()+df1['Logical'].std(),
                 df1['Logical'].std()/2)
plt.figure(figsize = (15,8))
plt.hist(df1['Logical'], ec = 'k',
        bins = bins,
        label = f"Skewness : {round(df1['Logical'].skew(),2)}",
        alpha = 0.7,
        density = True)
plt.xticks(bins)
plt.xlabel('Logical Scores', size = 15)
plt.ylabel('Density', size = 15)

plt.axvline(df1['Logical'].mean(), label = f"Mean: {round(df1['Logical'].mean(),2)}"
            , linestyle = '-.',
           color = 'red', linewidth = 2)
plt.axvline(df1['Logical'].median(), label = f"Median: {round(df1['Logical'].median(),2)}"
            , linestyle = '-.',
           color = 'green', linewidth = 2)
plt.axvline(df1['Logical'].mode()[0], label = f"Mode: {round(df1['Logical'].mode()[0],2)}"
            , linestyle = '-.',
           color = 'k', linewidth = 2)
sns.kdeplot(df1['Logical'])
plt.legend()
plt.show()


# Most scores fell within the range of 454 to 584, peaking at 495, with an average of 502.

# In[62]:


# Quant

bins = np.arange(df1['Quant'].min(), df1['Quant'].max()+df1['Quant'].std(),
                 df1['Quant'].std()/2)
plt.figure(figsize = (15,8))
plt.hist(df1['Quant'], ec = 'k',
        bins = bins,
        label = f"Skewness : {round(df1['Quant'].skew(),2)}",
        alpha = 0.7,
        density = True)
plt.xticks(bins)
plt.xlabel('Quant Scores', size = 15)
plt.ylabel('Density', size = 15)

plt.axvline(df1['Quant'].mean(), label = f"Mean: {round(df1['Quant'].mean(),2)}"
            , linestyle = '-.',
           color = 'red', linewidth = 2)
plt.axvline(df1['Quant'].median(), label = f"Median: {round(df1['Quant'].median(),2)}"
            , linestyle = '-.',
           color = 'green', linewidth = 2)
plt.axvline(df1['Quant'].mode()[0], label = f"Mode: {round(df1['Quant'].mode()[0],2)}"
            , linestyle = '-.',
           color = 'k', linewidth = 2)
sns.kdeplot(df1['Logical'])
plt.legend()
plt.show()


# The majority of the results fell between 425 and 608. The highest number of pupils, who averaged 513, had a score of 605.

# In[63]:


# computer programming

bins = np.arange(df1['ComputerProgramming'].min(), df1['ComputerProgramming'].max()+df1['ComputerProgramming'].std(),
                 df1['ComputerProgramming'].std()/2)
plt.figure(figsize = (15,6))
plt.hist(df1['ComputerProgramming'], ec = 'k',
        bins = bins,
        label = f"Skewness : {round(df1['ComputerProgramming'].skew(),2)}",
        alpha = 0.7,
        density = True)
plt.xticks(bins)
plt.xlabel('Computer Programming Scores', size = 15)
plt.ylabel('Density', size = 15)

plt.axvline(df1['ComputerProgramming'].mean(), label = f"Mean: {round(df1['ComputerProgramming'].mean(),2)}"
            , linestyle = '-.',
           color = 'red', linewidth = 2)
plt.axvline(df1['ComputerProgramming'].median(), label = f"Median: {round(df1['ComputerProgramming'].median(),2)}"
            , linestyle = '-.',
           color = 'green', linewidth = 2)
plt.axvline(df1['ComputerProgramming'].mode()[0], label = f"Mode: {round(df1['ComputerProgramming'].mode()[0],2)}"
            , linestyle = '-.',
           color = 'k', linewidth = 2)
sns.kdeplot(df1['ComputerProgramming'])
plt.legend()
plt.show()


# Most of the scores fell between 416 and 459. With an average score of 452, the pinnacle happened at 455.

# In[64]:


# Electronics  & semiconductors

bins = np.arange(df1['ElectronicsAndSemicon'].min(), df1['ElectronicsAndSemicon'].max()+df1['ElectronicsAndSemicon'].std(),
                 df1['ElectronicsAndSemicon'].std()/2)
plt.figure(figsize = (15,6))
plt.hist(df1['ElectronicsAndSemicon'], ec = 'k',
        bins = bins,
        label = f"Skewness : {round(df1['ElectronicsAndSemicon'].skew(),2)}",
        alpha = 0.7,
        density = True)
plt.xticks(bins)
plt.xlabel('Electronics & Semiconductors Scores', size = 15)
plt.ylabel('Density', size = 15)

plt.axvline(df1['ElectronicsAndSemicon'].mean(), label = f"Mean: {round(df1['ElectronicsAndSemicon'].mean(),2)}"
            , linestyle = '-.',
           color = 'red', linewidth = 2)
plt.axvline(df1['ElectronicsAndSemicon'].median(), label = f"Median: {round(df1['ElectronicsAndSemicon'].median(),2)}"
            , linestyle = '-.',
           color = 'green', linewidth = 2)
plt.axvline(df1['ElectronicsAndSemicon'].mode()[0], label = f"Mode: {round(df1['ElectronicsAndSemicon'].mode()[0],2)}"
            , linestyle = '-.',
           color = 'k', linewidth = 2)
sns.kdeplot(df1['ElectronicsAndSemicon'])
plt.legend()
plt.show()


# Most scores fell between 0 and 79. The highest number of students scored 0, with an average score of 96.

# In[65]:


#Age

bins = np.arange(df1['Age'].min(), df1['Age'].max()+df1['Age'].std(),
                 df1['Age'].std()/2)
plt.figure(figsize = (15,8))
plt.hist(df1['Age'], ec = 'k',
        bins = bins,
        label = f"Skewness : {round(df1['Age'].skew(),2)}",
        alpha = 0.7,
        density = True)
plt.xticks(bins)
plt.xlabel('Age', size = 15)
plt.ylabel('Density', size = 15)

plt.axvline(df1['Age'].mean(), label = f"Mean: {round(df1['Age'].mean(),2)}"
            , linestyle = '-.',
           color = 'red', linewidth = 2)
plt.axvline(df1['Age'].median(), label = f"Median: {round(df1['Age'].median(),2)}"
            , linestyle = '-.',
           color = 'green', linewidth = 2)
plt.axvline(df1['Age'].mode()[0], label = f"Mode: {round(df1['Age'].mode()[0],2)}"
            , linestyle = '-.',
           color = 'k', linewidth = 2)
sns.kdeplot(df1['Age'])
plt.legend()
plt.show()


# The age range of the majority of students was 22 to 25. The ages of the mean, median, and mode are roughly 25.

# # Box Plot

# In[66]:


# working years

plt.figure(figsize=(5, 4))
sns.boxplot(df1['working_years'])
plt.xlabel('working_years')
plt.tight_layout()
plt.show()


# Few values, or outliers, have a long tenure.

# In[67]:


# Salary
plt.figure(figsize=(5,4))
sns.boxplot(df1['Salary'])
plt.xlabel('Salary')
plt.tight_layout()
plt.show()


# The box plot shows that there is a noticeable concentration of data points with high salaries.

# In[68]:


# 10percentage
plt.figure(figsize=(5,4))
sns.boxplot(df1['10percentage'])
plt.xlabel('10percentage')
plt.tight_layout()
plt.show()


# The box plot makes it clear that there are a few extreme outliers present.

# In[69]:


# 12percentage
plt.figure(figsize=(5,4))
sns.boxplot(df1['12percentage'])
plt.xlabel('12percentage')
plt.tight_layout()
plt.show()


# There is just one data point with an exceptionally low score in the box plot.

# In[70]:


# collegeGPA
plt.figure(figsize=(5,4))
sns.boxplot(df1['collegeGPA'])
plt.xlabel('collegeGPA')
plt.tight_layout()
plt.show()


# The box plot indicates that the dataset has both low and high extreme values.

# In[71]:


# English
plt.figure(figsize=(5,4))
sns.boxplot(df1['English'])
plt.xlabel('English')
plt.tight_layout()
plt.show()


# The distribution representation makes clear both the lower and higher extreme values.

# In[72]:


# Logical
plt.figure(figsize=(5,4))
sns.boxplot(df1['Logical'])
plt.xlabel('Logical')
plt.tight_layout()
plt.show()


# lower extreme values are present, with only one noteworthy high extreme value.

# In[73]:


# Quant
plt.figure(figsize=(5,4))
sns.boxplot(df1['Quant'])
plt.xlabel('Quant')
plt.tight_layout()
plt.show()


# Both low and high extreme values are present, as the box plot demonstrates.

# In[74]:


# ComputerProgramming
plt.figure(figsize=(5,4))
sns.boxplot(df1['ComputerProgramming'])
plt.xlabel('ComputerProgramming')
plt.tight_layout()
plt.show()


# Both high and low extreme values are present in large numbers, as the box plot shows.

# In[75]:


# ElectronicsAndSemicon
plt.figure(figsize=(5,4))
sns.boxplot(df1['ElectronicsAndSemicon'])
plt.xlabel('ElectronicsAndSemicon')
plt.tight_layout()
plt.show()


# The dataset's median is the same as the lowest score.

# In[76]:


# Age

plt.figure(figsize=(5,4))
sns.boxplot(df1['Age'])
plt.xlabel('Age')
plt.tight_layout()
plt.show()


# In comparison to other data points, the box plot shows that four students have very high ages and one student has a very low age.

# # CDF

# In[77]:


# working_years

# CDF

plt.figure(figsize=(5, 4))
x_working_years, y_working_years = cdf(df1['working_years'])
x_sample_working_years, y_sample_working_years = cdf(np.random.normal(df1['working_years'].mean(), df1['working_years'].std(), size = len(df1['working_years'])))
plt.plot(x_working_years, y_working_years, linestyle = 'None',
        marker = '.', color = 'orange',
         alpha = 0.7, label = 'working_years')
plt.plot(x_sample_working_years, y_sample_working_years, linestyle = 'None',
        marker ='.', color = 'red',
        alpha = 0.7, label = 'Normal Distribution')
plt.xlabel('working_years')
plt.ylabel('CDF')
plt.legend()
plt.tight_layout()
plt.show()


# We can conclude that tenure is not normally distributed because the data is not normally distributed.

# In[78]:


#Salary

plt.figure(figsize=(5,4))
x_salary, y_salary = cdf(df1['Salary'])
x_sample_salary, y_sample_salary = cdf(np.random.normal(df1['Salary'].mean(), df1['Salary'].std(), size = len(df1['Salary'])))
plt.plot(x_salary, y_salary, linestyle = 'None',
        marker = '.', color = 'orange',
         alpha = 0.7, label = 'Tenure')
plt.plot(x_sample_salary, y_sample_salary, linestyle = 'None',
        marker ='.', color = 'red',
        alpha = 0.7, label = 'Normal Distribution')
plt.xlabel('Salary')
plt.ylabel('CDF')
plt.legend()
plt.tight_layout()
plt.show()


# The data exhibits a significant degree of skewness, as indicated by the cumulative distribution function (PDF), which deviates significantly from the expected distribution.

# In[79]:


#10percentage

plt.figure(figsize=(5,4))
x_10, y_10 = cdf(df1['10percentage'])
x_sample_10 , y_sample_10 = cdf(np.random.normal(df1['10percentage'].mean(), df1['10percentage'].std(), size = len(df1['10percentage'])))
plt.plot(x_10, y_10, linestyle = 'None',
        marker = '.', color = 'orange',
         alpha = 0.7, label = '10th %')
plt.plot(x_sample_10, y_sample_10, linestyle = 'None',
        marker ='.', color = 'red',
        alpha = 0.7, label = 'Normal Distribution')
plt.xlabel('10th Percentage')
plt.ylabel('CDF')
plt.legend()
plt.tight_layout()
plt.show()


# The data deviates from a normal distribution pattern and shows some skewness.

# In[80]:


#12percentage

plt.figure(figsize=(5,4))
x_12, y_12 = cdf(df1['12percentage'])
x_sample_12 , y_sample_12 = cdf(np.random.normal(df1['12percentage'].mean(), df1['12percentage'].std(), size = len(df1['12percentage'])))
plt.plot(x_12, y_12, linestyle = 'None',
        marker = '.', color = 'orange',
         alpha = 0.7, label = '12th %')
plt.plot(x_sample_10, y_sample_10, linestyle = 'None',
        marker ='.', color = 'red',
        alpha = 0.7, label = 'Normal Distribution')
plt.xlabel('12th Percentage')
plt.ylabel('CDF')
plt.legend()
plt.tight_layout()
plt.show()


# There is no normal distribution pattern in the data.

# In[81]:


#collegeGPA

plt.figure(figsize=(5,4))
x_gpa, y_gpa = cdf(df1['collegeGPA'])
x_sample_gpa , y_sample_gpa = cdf(np.random.normal(df1['collegeGPA'].mean(), df1['collegeGPA'].std(), size = len(df1['12percentage'])))
plt.plot(x_gpa, y_gpa, linestyle = 'None',
        marker = '.', color = 'orange',
         alpha = 0.7, label = 'GPA')
plt.plot(x_sample_gpa, y_sample_gpa, linestyle = 'None',
        marker ='.', color = 'red',
        alpha = 0.7, label = 'Normal Distribution')
plt.xlabel('College GPA')
plt.ylabel('CDF')
plt.legend()
plt.tight_layout()
plt.show()


# It is determined that the data is suitably regularly distributed.

# In[82]:


# English

plt.figure(figsize=(5,4))
x_eng, y_eng = cdf(df1['English'])
x_sample_eng , y_sample_eng = cdf(np.random.normal(df1['English'].mean(), df1['English'].std(), size = len(df1['English'])))
plt.plot(x_eng, y_eng, linestyle = 'None',
        marker = '.', color = 'orange',
         alpha = 0.7, label = 'English ')
plt.plot(x_sample_eng, y_sample_eng, linestyle = 'None',
        marker ='.', color = 'red',
        alpha = 0.7, label = 'Normal Distribution')
plt.xlabel('English Scores')
plt.ylabel('CDF')
plt.legend()
plt.tight_layout()
plt.show()


# The data follows a reasonably normal distribution pattern.

# In[83]:


#Logical

plt.figure(figsize=(5,4))
x_log, y_log = cdf(df1['Logical'])
x_sample_log , y_sample_log = cdf(np.random.normal(df1['Logical'].mean(), df1['Logical'].std(), size = len(df1['Logical'])))
plt.plot(x_log, y_log, linestyle = 'None',
        marker = '.', color = 'orange',
         alpha = 0.7, label = 'Logical ')
plt.plot(x_sample_log, y_sample_log, linestyle = 'None',
        marker ='.', color = 'red',
        alpha = 0.7, label = 'Normal Distribution')
plt.xlabel('Logical Scores')
plt.ylabel('CDF')
plt.legend()
plt.tight_layout()
plt.show()


# Data closely approximates a normal distribution pattern.

# In[84]:


#Quant

plt.figure(figsize=(5,4))
x_q, y_q = cdf(df1['Quant'])
x_sample_q , y_sample_q = cdf(np.random.normal(df1['Quant'].mean(), df1['Quant'].std(), size = len(df1['Quant'])))
plt.plot(x_q, y_q, linestyle = 'None',
        marker = '.', color = 'orange',
         alpha = 0.7, label = 'Quant ')
plt.plot(x_sample_q, y_sample_q, linestyle = 'None',
        marker ='.', color = 'red',
        alpha = 0.7, label = 'Normal Distribution')
plt.xlabel('Quant Scores')
plt.ylabel('CDF')
plt.legend()
plt.tight_layout()
plt.show()


# The data is nearly regularly distributed enough.

# In[85]:


#computerprogramming

plt.figure(figsize=(5,4))
x_cp, y_cp = cdf(df1['ComputerProgramming'])
x_sample_cp , y_sample_cp = cdf(np.random.normal(df1['ComputerProgramming'].mean(), df1['ComputerProgramming'].std(), size =                      len(df1['ComputerProgramming'])))
plt.plot(x_cp, y_cp, linestyle = 'None',
        marker = '.', color = 'orange',
         alpha = 0.7, label = 'Computer Programming ')
plt.plot(x_sample_cp, y_sample_cp, linestyle = 'None',
        marker ='.', color = 'red',
        alpha = 0.7, label = 'Normal Distribution')
plt.xlabel('Computer Programming Scores')
plt.ylabel('CDF')
plt.legend()
plt.tight_layout()
plt.show()


# There is no normal distribution pattern in the data.

# In[86]:


#Electronics & Semiconductors Scores

plt.figure(figsize=(5,4))
x_cp, y_cp = cdf(df1['ElectronicsAndSemicon'])
x_sample_cp , y_sample_cp = cdf(np.random.normal(df1['ElectronicsAndSemicon'].mean(), df1['ElectronicsAndSemicon'].std(), size =                      len(df1['ElectronicsAndSemicon'])))
plt.plot(x_cp, y_cp, linestyle = 'None',
        marker = '.', color = 'orange',
         alpha = 0.7, label = 'Electronics & Semiconductors')
plt.plot(x_sample_cp, y_sample_cp, linestyle = 'None',
        marker ='.', color = 'red',
        alpha = 0.7, label = 'Normal Distribution')
plt.xlabel('Electronics & Semiconductors Scores')
plt.ylabel('CDF')
plt.legend()
plt.tight_layout()
plt.show()


# There is no normal distribution pattern in the data.

# In[87]:


# Age

plt.figure(figsize=(5,4))
x_cp, y_cp = cdf(df1['Age'])
x_sample_cp , y_sample_cp = cdf(np.random.normal(df1['Age'].mean(), df1['Age'].std(), size =                      len(df1['Age'])))
plt.plot(x_cp, y_cp, linestyle = 'None',
        marker = '.', color = 'orange',
         alpha = 0.7, label = 'Age')
plt.plot(x_sample_cp, y_sample_cp, linestyle = 'None',
        marker ='.', color = 'red',
        alpha = 0.7, label = 'Normal Distribution')
plt.xlabel('Age')
plt.ylabel('CDF')
plt.legend()
plt.tight_layout()
plt.show()


# There is no trend of normal distribution in the age data.

# In[88]:


df1.dtypes


# # Categorical Features

# **Designation**

# In[89]:


top_10_groups = df1.groupby('Designation').size().sort_values(ascending=False).head(10)

plt.figure(figsize=(8, 6))
plt.bar(range(len(top_10_groups)), list(top_10_groups))
plt.xticks(range(len(top_10_groups)), list(top_10_groups.index), rotation=45)
plt.title('Top 10 Designation Frequencies')
plt.xlabel('Designations')
plt.ylabel('# Occurrences')
plt.tight_layout()
plt.show()


# The most popular designation is "software engineer," which is followed by "system engineer" and "software developer."

# **JobCity**

# In[90]:


top_10_groups = df1.groupby('JobCity').size().sort_values(ascending=False).head(10)

plt.figure(figsize=(8, 6))
plt.bar(range(len(top_10_groups)), list(top_10_groups))
plt.xticks(range(len(top_10_groups)), list(top_10_groups.index))
plt.title('Top 10 Job City Frequencies')
plt.xlabel('# Occurrences')
plt.ylabel('Job Cities')
plt.tight_layout()
plt.show()


# Bangalore is the best city for job placements, followed by Hyderabad, Pune, Noida, and Bangalore. Kolkata and Mumbai are the least favorable.

# **Gender**

# In[91]:


plt.figure(figsize=(3,3))
plt.pie(df1['Gender'].value_counts().tolist(),
        labels=df1['Gender'].value_counts().index,
        autopct='%1.1f%%',
        radius=1.5,
        wedgeprops={'edgecolor': 'k'},
        textprops={'fontsize': 10, 'fontweight': 'bold'},
        shadow=True,
        startangle=90,
        pctdistance=0.85)
plt.pie(df1['Gender'].value_counts().tolist(),
        colors=['white'],
        wedgeprops={'edgecolor': 'white'},
        radius=1)
plt.title('Gender %', pad=40, size=20)
plt.tight_layout()
plt.show()


# The male population in the sample is significantly bigger than the female population, indicating a lack of gender balance.

# **10board & 12board**

# In[92]:


fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

top_10_boards_10 = df1['10board'].str.upper().value_counts().nlargest(10)
top_10_boards_10.sort_values(ascending=True).plot(kind='barh', ax=ax[0], ec='k', title='Top 10 10th Boards')
ax[0].set_ylabel('Board', size=15)

top_10_boards_12 = df1['12board'].str.upper().value_counts().nlargest(10)
top_10_boards_12.sort_values(ascending=True).plot(kind='barh', ax=ax[1], ec='k', title='Top 10 12th Boards')
ax[1].set_ylabel('Board', size=15)
ax[1].set_xlabel('Count', size=15)

plt.tight_layout()
plt.show()


# The most popular school board for both 10th and 12th grades is CBSE.
# 

# **collegeTier**

# In[93]:


plt.figure(figsize=(3,3))
plt.pie(df1['CollegeTier'].value_counts().tolist(), labels = df1['CollegeTier'].value_counts().index,
       autopct = '%1.1f%%',
       radius = 1.75,
       wedgeprops = {'edgecolor':'k'},
       textprops = {'fontsize':9,'fontweight':'bold'},
       shadow = True,
       startangle = 90,
       pctdistance = 0.85)
plt.pie(df1['CollegeTier'].value_counts().tolist(), colors = ['white'],
        wedgeprops = {'edgecolor':'white'},
       radius = 1)
plt.title('College Tier %',pad = 40, size = 12)
plt.margins(0.02)
plt.tight_layout()
plt.show()


# Only 92.5 percent of the college is in Tier 1, which includes almost all of it.

# **Degree**

# In[94]:


df1['Degree'].value_counts().sort_values(ascending=True).plot(
    kind='barh',
    title='Degree',
    figsize=(5, 3),
    ec='k',
    alpha=0.7
)
plt.ylabel('Degree')
plt.xlabel('Count')
plt.xscale('log')
plt.tight_layout()
plt.show()


# There are relatively few students pursuing an M.Sc. in technology, and the majority of students have completed their B.Tech degrees.

# **collegeCityTire**

# In[95]:


plt.figure(figsize=(3,3))
plt.pie(df1['CollegeCityTier'].value_counts().tolist(), labels = df1['CollegeCityTier'].value_counts().index,
       autopct = '%1.1f%%',
       radius = 1.5,
       wedgeprops = {'edgecolor':'k'},
       textprops = {'fontsize':8,'fontweight':'bold'},
       shadow = True,
       startangle = 90,
       pctdistance = 0.84)
plt.pie(df1['CollegeCityTier'].value_counts().tolist(), colors = ['white'],
        wedgeprops = {'edgecolor':'white'},
       radius = 1)
plt.title('College Tier %',pad = 30, size = 12)
plt.margins(0.02)
plt.tight_layout()
plt.show()


# The majority of colleges are located in Tier 0 cities.

# **Graduationyear**

# In[96]:


df1['GraduationYear'].value_counts().sort_values(ascending=True).plot(
    kind='bar',
    title='Graduation Year',
    figsize=(6, 3),
    ec='k',
    alpha=0.7
)
plt.ylabel('Year')
plt.xlabel('Count')
plt.tight_layout()
plt.show()


# 2013 saw the highest number of students graduate, with 2014 and 2012 coming next.

# # Bivariate Analysis

# **Numerial**

# Salary & 10percentage

# In[97]:




plt.figure(figsize=(8, 6))
plt.scatter(df1['Salary'], df1['10percentage'])
plt.xlabel('Salary')
plt.ylabel('10percentage')
plt.title('Scatter Plot of Salary vs 10percentage')
plt.show()


# There isn't any relationship between salary and 10th grade results.

# **12percentage & Salary**

# In[98]:


plt.figure(figsize=(8, 6))
plt.scatter(df1['Salary'], df1['12percentage'])
plt.xlabel('Salary')
plt.ylabel('12percentage')
plt.title('Scatter Plot of Salary vs 12percentage')
plt.show()


# There isn't any relationship between salary and 12th grade results.

# **Salary & GraduationYear**

# In[99]:


sns.pairplot(df1, vars=['GraduationYear', 'Salary'])
plt.show()


# The experience comes when you graduate earlier.

# **Salary & collegeGPA**

# In[100]:


plt.figure(figsize=(8, 6))
plt.scatter(df1['Salary'], df1['collegeGPA'])
plt.xlabel('Salary')
plt.ylabel('CollegeGPA')
plt.title('Scatter Plot of Salary vs CollegeGPA')
plt.show()


# There isn't any relationship between salary and collegeGPA results.

# **Salary & Age**

# In[101]:


plt.figure(figsize=(8, 6))
plt.scatter(df1['Salary'], df1['Age'])
plt.xlabel('Salary')
plt.ylabel('Age')
plt.title('Scatter Plot of Salary vs Age')
plt.show()


# Salary and age are not related to each other.

# **Salary & working_years**

# In[102]:


plt.figure(figsize=(8, 6))
plt.scatter(df1['Salary'], df1['working_years'])
plt.xlabel('Salary')
plt.ylabel('working_years')
plt.title('Scatter Plot of Salary vs working_years')
plt.show()


# The salary increases by working years.

# **Salary with English, Quants, Logical**

# In[103]:


#English

plt.figure(figsize=(8, 6))
plt.scatter(df1['Salary'], df1['English'])
plt.xlabel('Salary')
plt.ylabel('English')
plt.title('Scatter Plot of Salary vs English')
plt.show()

#Logical

plt.figure(figsize=(8, 6))
plt.scatter(df1['Salary'], df1['Logical'])
plt.xlabel('Salary')
plt.ylabel('Logical')
plt.title('Scatter Plot of Salary vs Logical')
plt.show()

#Quant

plt.figure(figsize=(8, 6))
plt.scatter(df1['Salary'], df1['Quant'])
plt.xlabel('Salary')
plt.ylabel('Quant')
plt.title('Scatter Plot of Salary vs Quant')
plt.show()


# The scatter plots above provide sufficient proof that none of the scores above had an impact on salary.

# # Relation between categorical and numerical

# **Slary & Designation**

# In[104]:


avg_salary_top_10 = df1.groupby('Designation')['Salary'].mean().nlargest(10)

plt.figure(figsize=(10, 6))
avg_salary_top_10.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Average Salary for Top 10 Designations')
plt.xlabel('Designation')
plt.ylabel('Average Salary')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# The sales acoount manager and junior manager ears more

# **Salary & Gender**

# In[105]:


avg_salary_gender = df1.groupby('Gender')['Salary'].mean()


plt.figure(figsize=(8, 6))
avg_salary_gender.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Average Salary by Gender')
plt.xlabel('Gender')
plt.ylabel('Average Salary')
plt.xticks(rotation=0)  # Rotation of x-axis labels, if needed
plt.tight_layout()
plt.show()


# The male employess earn more.

# In[106]:


avg_salary_top_10 = df1.groupby('JobCity')['Salary'].mean().nlargest(10)

plt.figure(figsize=(10, 6))
avg_salary_top_10.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Average Salary for Top 10 JobCitys')
plt.xlabel('JobCity')
plt.ylabel('Average Salary')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# The respondents from kalmar,sweden earn more.

# **Salary & 10board**

# In[107]:


avg_salary_top_10 = df1.groupby('10board')['Salary'].mean().nlargest(10)

plt.figure(figsize=(10, 6))
avg_salary_top_10.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Average Salary for Top 10 10boards')
plt.xlabel('10board')
plt.ylabel('Average Salary')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# The students from Up board earn more salary.

# **Salary & 12board**

# In[108]:


avg_salary_top_10 = df1.groupby('12board')['Salary'].mean().nlargest(10)

plt.figure(figsize=(10, 6))
avg_salary_top_10.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Average Salary for Top 10 12boards')
plt.xlabel('12board')
plt.ylabel('Average Salary')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# The students from Up board earn more salary.

# **Collegetier & Salary**

# In[109]:


avg_salary_collegeTire = df1.groupby('CollegeTier')['Salary'].mean()


plt.figure(figsize=(8, 6))
avg_salary_collegeTire.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Average Salary by collegeTire')
plt.xlabel('collegeTire')
plt.ylabel('Average Salary')
plt.xticks(rotation=0)  # Rotation of x-axis labels, if needed
plt.tight_layout()
plt.show()


# collegetier 1 earns more.

# **Salary & Degree**

# In[110]:


avg_salary_collegeTire = df1.groupby('Degree')['Salary'].mean()


plt.figure(figsize=(8, 6))
avg_salary_collegeTire.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Average Salary by collegeTire')
plt.xlabel('collegeTire')
plt.ylabel('Average Salary')
plt.xticks(rotation=0)  # Rotation of x-axis labels, if needed
plt.tight_layout()
plt.show()


# Respondents from M.sc(Tech) earns more.

# **Salary & Specialization**

# In[111]:


avg_salary_top_10 = df1.groupby('Specialization')['Salary'].mean().nlargest(10)

plt.figure(figsize=(10, 6))
avg_salary_top_10.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Average Salary for Top 10 Specialization')
plt.xlabel('Specialization')
plt.ylabel('Average Salary')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Polymer technology specialization earns more.

# **Salary & CollegeCityTier**

# In[112]:


avg_salary_collegeTire = df1.groupby('CollegeCityTier')['Salary'].mean()


plt.figure(figsize=(8, 6))
avg_salary_collegeTire.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Average Salary by collegeTire')
plt.xlabel('collegeTire')
plt.ylabel('Average Salary')
plt.xticks(rotation=0)  # Rotation of x-axis labels, if needed
plt.tight_layout()
plt.show()


# collegecitytier of 1 earns more.

# **Salary & CollegeState**

# In[113]:


avg_salary_top_10 = df1.groupby('CollegeState')['Salary'].mean().nlargest(10)

plt.figure(figsize=(10, 6))
avg_salary_top_10.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Average Salary for Top 10 CollegeState')
plt.xlabel('CollegeState')
plt.ylabel('Average Salary')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# The respondents from Jharkhand earn more.

# # Relation between categorical and Categorical
# 

# In[114]:


df1.dtypes


# CollegeTier & Gender

# In[115]:


grouped = df1.groupby(['CollegeTier', 'Gender']).size().unstack()

grouped.plot(kind='bar', stacked=True)
plt.title('Stacked Bar Plot for Degree and Gender')
plt.xlabel('CollegeTier')
plt.ylabel('Count')
plt.xticks(rotation=45)  
plt.legend(title='Gender')
plt.show()


# Most of the respondents are from collegetier 2 and and more than 70% are male.

# CollegeCityTire & Gender

# In[116]:


grouped = df1.groupby(['CollegeCityTier', 'Gender']).size().unstack()

grouped.plot(kind='bar', stacked=True)
plt.title('Stacked Bar Plot for Degree and Gender')
plt.xlabel('CollegeCityTier')
plt.ylabel('Count')
plt.xticks(rotation=45)  
plt.legend(title='Gender')
plt.show()


# Most of the respondents are under 0 collegecitytier and more than 70% are male.

# Gender & College State

# In[117]:


top_5_college_states = df1['CollegeState'].value_counts().head(5).index

df_top_5_states = df1[df1['CollegeState'].isin(top_5_college_states)]

grouped = df_top_5_states.groupby(['CollegeState', 'Gender']).size().unstack()

grouped.plot(kind='barh', stacked=True)
plt.title('Stacked Bar Plot for Top 5 College States and Gender')
plt.xlabel('Top 5 College States')
plt.ylabel('Count')
plt.xticks(rotation=45)  
plt.legend(title='Gender')
plt.show()


# Uttar Pradesh is Top 1 College State in them most of them are male.

# # Research Questions
# 

# 1.Times of India article dated Jan 18, 2019 states that “After doing your Computer Science Engineering if you take up jobs as a Programming Analyst, Software Engineer, Hardware Engineer and Associate Engineer you can earn up to 2.5-3 lakhs as a fresh graduate.” 

# # Solution with Visualization

# In[118]:


designations = df['Designation'].value_counts().sort_index()
pd.set_option('display.max_rows', None)

print(designations)


# In[119]:


df['Designation'] = df['Designation'].replace([
    'programmer analyst trainee', 'programmer analyst'
], 'programmer analyst'
)

df['Designation'] = df['Designation'].replace([
    'software eng', 'software engg', 'software engineer', 'software engineere', 'software enginner'
], 'software engineer'
)


# In[120]:


df3 = df[(df["Designation"].isin(["programmer analyst", "software engineer", "hardware engineer", "associate engineer"])) &
                (df["Specialization"].isin(["computer science & engineering", "computer engineering"]))]


# In[121]:


fig, ax = plt.subplots(figsize=(10, 4))
sns.barplot(x='Salary', y='Designation',
            data=df3,
            capsize=0.1,
            ax=ax)
ax.axvline(df3['Salary'].mean(), color='k',
           linestyle=':',
           linewidth=2, label='Overall\nAvg. Salary')
ax.set_title('Avg Salary for Each Designation after pursuing Computer Science Engineering')
ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1))
ax.set_xlabel('Salary')
ax.set_ylabel('Designation')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

plt.tight_layout()
plt.show()


# Its true that software Engineer & Programmer Analyst high salary compared to anyother designations.

# 2. Is there a relationship between gender and specialization? (i.e. Does the preference of Specialisation depend on the Gender?)

# In[122]:


from scipy.stats import chi2
from scipy.stats import chi2_contingency


# In[123]:


x = np.linspace(0, 100, 100)
y = chi2.pdf(x, df = 4)
plt.plot(x, y)


# In[124]:


obsr = pd.crosstab(df1.Specialization,df1.Gender)
obsr


# The above column shows the male and female ratio acoording to their specialization.

# # Conclusion

# **Data Understanding:**

# The dataset, which focuses on the goal variable Salary, includes the job outcomes of engineering graduates.
# It also contains standardised scores in three different domains: personality, technical, and cognitive skills.

# **Data manipulation:**

# The dataset is initially observed to have 40 columns and 4000 rows.
# There are a lot of duplicate values in the dataset, which makes data processing necessary.
# First, we eliminate any unnecessary rows and columns.
# Next, we determine whether any missing values (NaN) exist.
# We start with data cleaning and then move on to visualization.

# **Data Visualization**

# **Univariate Analysis:**

# Plots such as Histograms, Box Plots, Summary Plots, and Cumulative Distribution Functions (PDF) are all included in univariate analysis.
# Distributions of probability and frequency are depicted in these graphics.
# 

# **Bivariate Analysis:**

# Pie charts, pivot tables, crosstabs, scatterplots, and barplots are examples of bivariate analysis tools.
# This approach facilitates percentage comparisons among many variables.
# It also helps with outlier identification, as shown by Boxplots.
# For example, by emphasizing the cities with greater employee counts, Countplots help discover outliers within categorical variables, like Job City.

# # Making A Research question

# Is their a relation between salary and Degree

# As we seen earlier the M.sc(Tech)respondents earns more
