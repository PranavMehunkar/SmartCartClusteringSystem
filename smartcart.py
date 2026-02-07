#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv("smartcart_customers.csv")


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.isnull().sum()


# # Data Preprocessing

# ## 1. Handle Missing Values

# In[6]:


df["Income"]=df["Income"].fillna(df["Income"].median())


# In[7]:


df.head()


# ## Feature engineering

# In[8]:


# Age
df["Age"]=2026-df["Year_Birth"]


# In[9]:


# Customer Joining Date

df["Dt_Customer"]=pd.to_datetime(df["Dt_Customer"],dayfirst=True)

reference_date=df["Dt_Customer"].max()

df["Customer_Tenure_Days"]=(reference_date-df["Dt_Customer"]).dt.days


# In[10]:


# Spending

df["Total_Spending"]=df["MntWines"]+df["MntFruits"]+df["MntMeatProducts"]+df["MntFishProducts"]+df["MntSweetProducts"]+df["MntGoldProds"]


# In[11]:


# Children
df["Total_Children"]=df["Kidhome"]+df["Teenhome"]


# In[14]:


# Education

df["Education"].value_counts()

df["Education"]=df["Education"].replace({
    "Basic":"Undergraduate","2n Cycle":"Undergraduate",
    "Graduation":"Graduate",
    "Master":"Postgraduate","PhD":"Postgraduate"
})


# In[18]:


# Marital Status

df["Living_With"]=df["Marital_Status"].replace({
    "Married":"Partner","Together":"Partner",
    "Single":"Alone","Divorced":"Alone",
    "Widow":"Alone","Absurd":"Alone","YOLO":"Alone"
})


# ## Drop Columns

# In[20]:


df.head()


# In[23]:


cols=["ID","Year_Birth","Marital_Status","Kidhome","Teenhome","Dt_Customer"]
spending_cols=["MntWines","MntFruits","MntMeatProducts","MntFishProducts","MntSweetProducts","MntGoldProds"]

cols_to_drop=cols+spending_cols

df_cleaned=df.drop(columns=cols_to_drop)


# In[24]:


df_cleaned.shape


# In[26]:


df_cleaned.head()


# # Outliers

# In[27]:


cols=["Income","Recency","Response","Age","Total_Spending","Total_Children"]

# relative plots of some features- pair plots
sns.pairplot(df_cleaned[cols])


# In[28]:


# Remove outliers

print("data size with outliers:",len(df_cleaned))

df_cleaned=df_cleaned[(df_cleaned["Age"]<90)]
df_cleaned=df_cleaned[(df_cleaned["Income"]<600_000)]

print("data size withput outliers:",len(df_cleaned))


# # Heatmap

# In[29]:


corr=df_cleaned.corr(numeric_only=True)


# In[32]:


plt.figure(figsize=(8,6))

sns.heatmap(
    corr,
    annot=True,
    annot_kws={"size":6},
    cmap="coolwarm"
)


# In[33]:


df_cleaned.shape


# In[34]:


df_cleaned.head()


# # Encoding

# In[35]:


from sklearn.preprocessing import OneHotEncoder


# In[37]:


ohe=OneHotEncoder()

cat_cols=["Education","Living_With"]

enc_cols=ohe.fit_transform(df_cleaned[cat_cols])


# In[40]:


enc_df=pd.DataFrame(enc_cols.toarray(),columns=ohe.get_feature_names_out(cat_cols),index=df_cleaned.index)


# In[42]:


df_encoded=pd.concat([df_cleaned.drop(columns=cat_cols),enc_df],axis=1)


# In[43]:


df_encoded.shape


# In[44]:


df_encoded.head()


# # Scaling

# In[45]:


from sklearn.preprocessing import StandardScaler


# In[46]:


X=df_encoded


# In[47]:


scaler=StandardScaler()

X_scaled=scaler.fit_transform(X)


# # Visualize

# In[49]:


X_scaled.shape


# In[50]:


# 2D
from sklearn.decomposition import PCA


# In[54]:


pca=PCA(n_components=3)

X_pca=pca.fit_transform(X_scaled)


# In[55]:


pca.explained_variance_ratio_


# In[59]:


# plot
fig=plt.figure(figsize=(8,6))

ax=fig.add_subplot(111,projection="3d")

ax.scatter(X_pca[:,0],X_pca[:,1],X_pca[:,2])

ax.set_xlabel("PCA1")
ax.set_ylabel("PCA2")
ax.set_zlabel("PCA3")
ax.set_title("3d projection")


# # Analyze K value
# ## 1. Elbow Method

# In[61]:


import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# In[62]:


from sklearn.cluster import KMeans
from kneed import KneeLocator

wcss=[]
for k in range(1,11):
    kmeans=KMeans(n_clusters=k,random_state=42)
    kmeans.fit_predict(X_pca)
    wcss.append(kmeans.inertia_)


# In[63]:


knee=KneeLocator(range(1,11),wcss,curve="convex",direction="decreasing")
optimal_k=knee.elbow


# In[66]:


print("best k =",optimal_k)


# In[68]:


# plot

plt.plot(range(1,11),wcss,marker='o')
plt.xlabel("K")
plt.ylabel("WCSS")


# ## 2. Silhouette Score

# In[71]:


from sklearn.metrics import silhouette_score

scores=[]

for k in range(2,11):
    kmeans=KMeans(n_clusters=k,random_state=42)
    labels=kmeans.fit_predict(X_pca)
    score=silhouette_score(X_pca,labels)
    scores.append(score)

# plot
plt.plot(range(2,11),scores,marker='o')
plt.xlabel("K")
plt.ylabel("Silhouette score")


# In[79]:


# combined plot

k_range=range(2,11)

fig,ax1=plt.subplots(figsize=(8,6))

ax1.plot(k_range,wcss[:len(k_range)],marker="o",color="blue")
ax1.set_xlabel("K")
ax1.set_ylabel("WCSS")

ax2=ax1.twinx()
ax2.plot(k_range,scores[:len(k_range)],marker="x",color="red",linestyle="--")
ax2.set_ylabel("SS")


# # Clustering

# In[80]:


# K_means

kmeans=KMeans(n_clusters=4,random_state=42)
labels_kmeans=kmeans.fit_predict(X_pca)


# In[81]:


fig=plt.figure(figsize=(8,6))

ax=fig.add_subplot(111,projection="3d")

ax.scatter(X_pca[:,0],X_pca[:,1],X_pca[:,2],c=labels_kmeans)


# In[82]:


# Agglomerative Clustering
from sklearn.cluster import AgglomerativeClustering


# In[83]:


agg_clf=AgglomerativeClustering(n_clusters=4,linkage="ward")
labels_agg=agg_clf.fit_predict(X_pca)


# In[84]:


fig=plt.figure(figsize=(8,6))
ax=fig.add_subplot(111,projection="3d")
ax.scatter(X_pca[:,0],X_pca[:,1],X_pca[:,2],c=labels_agg)


# # Characterization of Clusters

# In[98]:


X["cluster"]=labels_agg


# In[99]:


X.head()


# In[100]:


pal=["red","blue","yellow","green"]

sns.countplot(x=X["cluster"],palette=pal,hue=X["cluster"])


# In[101]:


# Income & Spending patterns

sns.scatterplot(x=X["Total_Spending"],y=X["Income"],hue=X["cluster"],palette=pal)


# In[102]:


# Cluster Summary

cluster_summary=X.groupby("cluster").mean()
print(cluster_summary)


# In[ ]:




