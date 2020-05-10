#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[98]:


df = pd.read_csv("C:\\Users\\Satish Iyer\\Downloads\\EastWestAirlines.csv")
df.head()


# In[99]:


### Scatter Plot
plt.scatter(df.Balance,df.Bonus_miles)


# In[100]:


# Normalization function 
def norm_func(i):
    x = (i-i.min())	/	(i.max()	-	i.min())
    return (x)


# In[101]:


# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(df.iloc[:,1:])
df_norm


# In[103]:


km = KMeans(n_clusters=3)
km


# In[117]:


y_predicted = km.fit_predict(df_norm[['Balance','Bonus_miles']])
y_predicted


# In[128]:


km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df_norm[['Balance','Bonus_miles']])
y_predicted


# In[134]:


df1=df_norm[df_norm.clusters==0]
df2=df_norm[df_norm.clusters==1]
df3=df_norm[df_norm.clusters==2]

plt.scatter(df1.Balance,df1.Bonus_miles,color='green')
plt.scatter(df2.Balance,df2.Bonus_miles,color='red')
plt.scatter(df3.Balance,df3.Bonus_miles,color='blue')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label="centroid")


# In[121]:


km.cluster_centers_


# In[138]:


k_rng = range(1,10)
sse = []
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(df_norm[['Balance','Bonus_miles']])
    sse.append(km.inertia_)


# In[139]:


sse


# In[140]:


plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng,sse)


# In[ ]:




