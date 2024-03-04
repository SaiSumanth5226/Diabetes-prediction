#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis

# In[6]:


#Installation of required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


#importing daytaset
dataset = pd.read_csv("diabetes.csv")
dataset


# In[8]:


dataset.info()


# In[9]:


dataset.isnull().sum()


# In[13]:


dataset.describe


# # Step 2 : Data Visualization

# In[17]:


#plot of corelation between independent variables
plt.figure(figsize=(8,8))
sns.heatmap(dataset.corr(), annot = True, fmt = ".3f", cmap=("YlGnBu"))
plt.title ("Correlation Heatmap")


# In[23]:


#Exploring pregnancy and target variables
plt.figure(figsize =(10,8))
#Ploting density function graph of the Pregnancies and target variables
kde = sns.kdeplot(dataset["Pregnancies"][dataset["Outcome"]==1], color ="Red", shade = True)
kde = sns.kdeplot(dataset["Pregnancies"][dataset["Outcome"]==0], color ="Blue", shade = True)
kde.set_xlabel("Pregnancies")
kde.set_ylabel("Density")
kde.legend(["Positive", "Negative"])


# In[24]:


#Exploring Glucose and Target variables
plt.figure(figsize =(10,8))
sns.violinplot(data=dataset, x="Outcome",y="Glucose", split=True,linewidth=2, Inner="quart")


# In[25]:


#Exploring Glucose and target variables
plt.figure(figsize =(10,8))
#Ploting density function graph of the Glucose and target variables
kde = sns.kdeplot(dataset["Glucose"][dataset["Outcome"]==1], color ="Red", shade = True)
kde = sns.kdeplot(dataset["Glucose"][dataset["Outcome"]==0], color ="Blue", shade = True)
kde.set_xlabel("Glucose")
kde.set_ylabel("Density")
kde.legend(["Positive", "Negative"])


# In[29]:


#Replace 0 values with the mean/median off the respective feature
#Glucose
dataset["Glucose"]=dataset["Glucose"].replace(0, dataset["Glucose"].median())
#BloodPressure
dataset["BloodPressure"]=dataset["BloodPressure"].replace(0, dataset["BloodPressure"].median())
#BMI
dataset["BMI"]=dataset["BMI"].replace(0, dataset["BMI"].mean())
#BMI
dataset["SkinThickness"]=dataset["SkinThickness"].replace(0, dataset["SkinThickness"].mean())
#BMI
dataset["Insulin"]=dataset["Insulin"].replace(0, dataset["Insulin"].mean())



# # Step 4 : Data Preparation

# In[30]:


dataset


# In[33]:


#Splitting the dependent and independent variable
x = dataset.drop(["Outcome"],axis=1)
y = dataset["Outcome"]


# In[34]:


x


# In[35]:


y


# In[36]:


#Split train and test dataset
#!pip install scikit-learn
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=42)
# In[40]:


x_train


# In[41]:


x_test


# In[42]:


y_train


# In[43]:


y_test


# # Step 5 : Build ML Model

# In[47]:


#KNN
from sklearn.neighbors import KNeighborsClassifier


# In[52]:


training_accuracy = [ ]
test_accuracy = [ ]
for n_neighbors in range (1,11):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(x_train,y_train)
    
    #check accuracy score
    training_accuracy.append(knn.score(x_train,y_train))
    test_accuracy.append(knn.score(x_test,y_test))
    


# In[53]:


plt.plot(range(1,11),test_accuracy,label="test_accuracy")
plt.plot(range(1,11),test_accuracy,label="test_accuracy")
plt.ylabel("accuracy")
plt.xlabel("n_neighbors")
plt.legend()


# In[56]:


knn= KNeighborsClassifier(n_neighbors=9)
knn.fit(x_train,y_train)
print(knn.score(x_train,y_train),":Training accuracy")
print(knn.score(x_test,y_test),":Test accuracy")


# In[60]:


from sklearn.tree import DecisionTreeClassifier
dt= DecisionTreeClassifier(random_state =0)
dt.fit(x_train,y_train)
print(dt.score(x_train,y_train),":Training accuracy")
print(dt.score(x_test,y_test),":Test accuracy")


# In[61]:


dt1= DecisionTreeClassifier(random_state =0,max_depth=3)
dt1.fit(x_train,y_train)
print(dt1.score(x_train,y_train),":Training accuracy")
print(dt1.score(x_test,y_test),":Test accuracy")


# In[64]:


from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier (random_state=42)
mlp.fit(x_train,y_train)
print(mlp.score(x_train,y_train),":Training accuracy")
print(mlp.score(x_test,y_test),":Test accuracy")


# In[66]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train_scaled=sc.fit_transform(x_train)
x_test_scaled=sc.fit_transform(x_test)


# In[67]:


mlp1 = MLPClassifier (random_state=0)
mlp1.fit(x_train_scaled,y_train)
print(mlp1.score(x_train_scaled,y_train),":Training accuracy")
print(mlp1.score(x_test_scaled,y_test),":Test accuracy")


# In[ ]:




