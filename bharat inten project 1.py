#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib 
matplotlib.rcParams["figure.figsize"] = (20,10)
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score


# In[2]:


df = pd.read_csv("C:/Users/bhava/Downloads/banglore.csv")
df.head()


# In[26]:


df.shape


# In[27]:


df.columns


# In[28]:


df.info()


# In[29]:


df.describe()


# In[30]:


df.isnull()


# In[31]:


df.isnull().sum()


# In[32]:


df.groupby("area_type")["area_type"].agg("count")


# In[33]:


df.info()


# In[34]:


df.head()


# In[35]:


df = df.drop(["area_type", "society","balcony", "availability"], axis = "columns")


# In[36]:


df.shape


# In[37]:


df = df.dropna()


# In[38]:


df.isnull().sum()


# In[39]:


df.shape


# In[40]:


df["size"].unique()


# In[41]:


df['BHK'] = df["size"].apply(lambda x: int(x.split(" ")[0]))


# In[42]:


df.head()


# In[43]:


df.total_sqft.unique()


# In[44]:


def is_float(x):
    try:
        float(x)
    except:
        return False
    return True


# In[45]:


df[~df["total_sqft"].apply(is_float)].head(10)


# In[46]:


def convert_sqft_to_number(x):
    tokens = x.split("-")
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None


# In[47]:


df = df.copy()
df["total_sqft"] = df["total_sqft"].apply(convert_sqft_to_number)
df.head(10)


# In[48]:


df = df.copy()
df["price_per_sqft"] = df["price"]*100000/df["total_sqft"]
df.head()


# In[49]:


df.location = df.location.apply(lambda x: x.strip())
location_stats = df['location'].value_counts(ascending=False)
location_stats


# In[50]:


len(location_stats[location_stats<=10])


# In[51]:


location_stats_less_than_10 = location_stats[location_stats<=10]
location_stats_less_than_10


# In[52]:


df.location = df.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
len(df.location.unique())


# In[53]:


df.head()


# In[54]:


df[df.total_sqft/df.BHK<300].head()


# In[55]:


df = df[~(df.total_sqft/df.BHK<300)]
df.shape


# In[60]:


df.describe()


# In[61]:


def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
df = remove_pps_outliers(df)
df.shape


# In[62]:


df.head()


# In[65]:


df.head(20)


# In[66]:


def plot_scatter_chart(df,location):
    bhk2 = df[(df.location==location) & (df.BHK==2)]
    bhk3 = df[(df.location==location) & (df.BHK==3)]
    matplotlib.rcParams['figure.figsize'] = (8,6)
    plt.scatter(bhk2.total_sqft,bhk2.price,color='blue',label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price,marker='+', color='green',label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price (Lakh Indian Rupees)")
    plt.title(location)
    plt.legend()
    
plot_scatter_chart(df,"1st Phase JP Nagar")


# In[67]:


plt.hist(df.price_per_sqft,rwidth=0.8)
plt.xlabel("Price Per Square Feet")
plt.ylabel("Count")


# In[68]:


plt.hist(df.bath,rwidth=0.8)
plt.xlabel("Number of bathrooms")
plt.ylabel("Count")


# In[69]:


df[df.bath>10]


# In[70]:


df[df.bath>df.BHK+2]


# In[71]:


df.head()


# In[72]:


df.shape


# In[73]:


dummies = pd.get_dummies(df.location)
dummies.head(20)


# In[74]:


dummies.head(20)


# In[75]:


df = pd.concat([df,dummies.drop('other',axis='columns')],axis='columns')
df.head()


# In[76]:


df = df.drop('location',axis='columns')
df.head()


# In[77]:


X = df.drop(['price'],axis='columns')
X.head()


# In[78]:


X = df.drop(['size'],axis='columns')
X.head()


# In[79]:


y = df.price
y.head()


# In[80]:


X = X.drop(['price_per_sqft'],axis='columns')
X.head()


# In[81]:


X = X.drop(['price'],axis='columns')
X.head()


# In[82]:


X.shape


# In[83]:


y.shape


# In[84]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)


# In[85]:


lr_clf = LinearRegression()
lr_clf.fit(X_train,y_train)
lr_clf.score(X_test,y_test)


# In[86]:


cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
cross_val_score(LinearRegression(), X, y, cv=cv)


# In[87]:


def find_best_model_using_gridsearchcv(X,y):
    algos = {
        'linear_regression' : {
            'model': LinearRegression(),
            'params': {
                'normalize': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1,2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion' : ['mse','friedman_mse'],
                'splitter': ['best','random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X,y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores,columns=['model','best_score','best_params'])


# In[88]:


find_best_model_using_gridsearchcv(X,y)


# In[89]:


def predict_price(location,sqft,bath,bhk):    
    loc_index = np.where(X.columns==location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return lr_clf.predict([x])[0]


# In[90]:


predict_price('1st Phase JP Nagar',1000, 2, 2)


# In[95]:


df.head()


# In[97]:


predict_price('Banashankari Stage V',2000, 3, 3)


# In[96]:


predict_price('2nd Stage Nagarbhavi',5000, 2, 2)


# In[94]:


predict_price('Indira Nagar',1500, 3, 3)

