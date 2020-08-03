#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from pandas import Series, DataFrame


# In[2]:


titanic_df = pd.read_csv('train.csv')


# In[3]:


titanic_df.head()


# In[4]:


titanic_df.info()


# In[5]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')


# In[54]:


sns.countplot('Sex',data=titanic_df)


# In[55]:


sns.countplot('Sex',data=titanic_df,hue='Pclass')


# In[56]:


sns.countplot('Pclass',data=titanic_df,hue='Sex')


# In[23]:


def male_female_child(passenger):
    age, sex = passenger
    
    if age < 16:
        return 'child'
    else:
        return sex


# In[24]:


titanic_df['person'] = titanic_df[['Age','Sex']].apply(male_female_child,axis=1)


# In[25]:


titanic_df[0:10]


# In[57]:


sns.countplot('Pclass',data=titanic_df,hue='person')


# In[32]:


titanic_df['Age'].hist(bins=70)


# In[34]:


titanic_df['Age'].mean()


# In[36]:


titanic_df['person'].value_counts()


# In[38]:


fig = sns.FacetGrid(titanic_df,hue='Sex',aspect=4)

fig.map(sns.kdeplot,'Age',shade=True)

oldest = titanic_df['Age'].max()

fig.set(xlim=(0,oldest))

fig.add_legend()


# In[39]:


fig = sns.FacetGrid(titanic_df,hue='person',aspect=4)

fig.map(sns.kdeplot,'Age',shade=True)

oldest = titanic_df['Age'].max()

fig.set(xlim=(0,oldest))

fig.add_legend()


# In[40]:


fig = sns.FacetGrid(titanic_df,hue='Pclass',aspect=4)

fig.map(sns.kdeplot,'Age',shade=True)

oldest = titanic_df['Age'].max()

fig.set(xlim=(0,oldest))

fig.add_legend()


# In[41]:


titanic_df.head()


# In[42]:


deck = titanic_df['Cabin'].dropna()


# In[43]:


deck.head()


# In[59]:


levels = []

for level in deck:
    levels.append(level[0])

cabin_df = DataFrame(levels)

cabin_df.columns = ['Cabin']

sns.countplot('Cabin',data=cabin_df,palette = 'winter_d')


# In[64]:


cabin_df = cabin_df[cabin_df.Cabin != 'T']

sns.countplot('Cabin',data=cabin_df,palette = 'summer')


# In[65]:


titanic_df.head()


# In[74]:


sns.countplot('Embarked',data=titanic_df,hue='Pclass')


# In[75]:


#Who was alone and who was with family?


# In[76]:


titanic_df.head()


# In[77]:


titanic_df['Alone'] = titanic_df.SibSp + titanic_df.Parch


# In[78]:


titanic_df['Alone']


# In[80]:


titanic_df['Alone'].loc[titanic_df['Alone'] > 0] = 'With family'

titanic_df['Alone'].loc[titanic_df['Alone'] == 0] = 'Alone'


# In[81]:


titanic_df.head()


# In[82]:


sns.countplot('Alone',data=titanic_df, palette='Blues')


# In[84]:


titanic_df['Survivor'] = titanic_df.Survived.map({0:'no',1:'yes'})

sns.countplot('Survivor',data=titanic_df, palette='Set1')


# In[86]:


sns.factorplot('Pclass','Survived',data=titanic_df)


# In[88]:


sns.factorplot('Pclass','Survived',data=titanic_df, hue='person')


# In[91]:


sns.lmplot('Age','Survived',data=titanic_df)


# In[94]:


sns.lmplot('Age','Survived',data=titanic_df, hue='Pclass', palette='winter')


# In[95]:


generations = [10,20,40,60,80]

sns.lmplot('Age','Survived',data=titanic_df, hue='Pclass', palette='winter', x_bins= generations)


# In[97]:


sns.lmplot('Age','Survived',data=titanic_df, hue='Sex', palette='winter', x_bins=generations)


# In[ ]:




