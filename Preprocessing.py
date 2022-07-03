#!/usr/bin/env python
# coding: utf-8

# ---
# 
# ***Now that we have discussed what is data preprocessing, let us dive into some code to complete preprocessing for getting our datasets ready for model building***
# 
# 
# 
# ---
# 
# 
# First we will pre-process the categorical features and then the numerical features on all the 3 available tables.
# 
# ---
# 
# ## `PREPROCESSING: CATEGORICAL FEATURES`
# 
# - Find out and impute, if we have missing values in the categorical features.
# - Remove the features which do not add much information
# - Choose an Encoding scheme to convert categorical feature into numeric.
# 
# 
# 
# ---

# In[10]:


# importing required libraries

import pandas as pd
import numpy as np
import category_encoders as ce

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ---
# ### `DATASET 1: Weekly Sales Data` contains the following features
# 
# - **WEEK_END_DATE** - week date
# - **STORE_NUM** - store number
# - **UPC** - (Universal Product Code) product specific identifier
# - **BASE_PRICE** - base price of item
# - **DISPLAY** - product was a part of in-store promotional display
# - **FEATURE** - product was in in-store circular
# - **UNITS** - units sold (target)
# 
# ---

# In[ ]:


# read the train data
data = pd.read_csv('dataset/train.csv')


# In[ ]:


data.head()


# ---
# 
# ###  `WEEKLY SALES DATA`  has the following categorical features
# 
#     - STORE_NUM
#     - UPC
#     - FEATURE
#     - DISPLAY
#     
# ---

# In[11]:


# check for the null values in the categorical features
data[['STORE_NUM', 'UPC', 'FEATURE', 'DISPLAY']].isna().sum()


# ***No Null Values***
# 
# ---
#  -  `STORE_NUM` - No changes required as it is a key and will be used to merge tables later.
#  -  `UPC      ` - No changes required as it is a key and will be used to merge tables later.
#  -  `FEATURE  ` - No Preprocessing Required
#  -  `DISPLAY  ` - No Preprocessing Required
#  ---

# ---
# 
# ### `DATASET 2: PRODUCT DATA` contains the details about the products
# 
# - **UPC** - (Universal Product Code) product specific identifier
# - **DESCRIPTION**	- product description
# - **MANUFACTURER** - product	manufacturer
# - **CATEGORY** - category of product
# - **SUB_CATEGORY** - sub-category of product
# - **PRODUCT_SIZE** - package size or quantity of product
# 
# ---

# In[12]:


# read the product data
product_data = pd.read_csv('dataset/product_data.csv')


# In[13]:


product_data.head()


# ---
# 
# ### `PRODUCT DATA`  has the following categorical features
# 
#     - UPC
#     - DESCRIPTION
#     - MANUFACTURER
#     - CATEGORY
#     - SUB_CATEGORY
#     - PRODUCT_SIZE
#     
# ---

# In[14]:


# shape of the data
product_data.shape


# In[15]:


# check for the null values in the categorical features
product_data[['UPC', 'DESCRIPTION', 'MANUFACTURER', 'CATEGORY', 'SUB_CATEGORY', 'PRODUCT_SIZE']].isna().sum()


# ***No Null Values***

# In[16]:


# number of unique description
product_data.DESCRIPTION.nunique()


# In[17]:


# number of unique manufacturer
product_data.MANUFACTURER.nunique()


# In[18]:


# number of unique categories
product_data.CATEGORY.nunique()


# In[19]:


# number of unique sub categories
product_data.SUB_CATEGORY.nunique()


# In[20]:


# number of unique product sizes
product_data.PRODUCT_SIZE.nunique()


# ---
#  - `DESCRIPTION` - In the description, we have category, subcategory and size of the product and these are already present in the other features as well. So, We will drop this feature as it will not add much value to the model.
#  - `MANUFACTURER`, `CATEGORY`, `SUB_CATEGORY`- As, there is no order in the given categories, so we will One Hot Encode this features.
#  - `PRODUCT_SIZE` - The product size units are different for different categories of products. So, here for each category we will do the binning based on different sizes.
# ---

# In[21]:


# drop the DESCRIPTION FEATURE
product_data = product_data.drop(columns= ['DESCRIPTION'])


# In[22]:


product_data


# In[23]:


# remove the units from the product size
# we will keep only the values
product_data['PRODUCT_SIZE'] = product_data.PRODUCT_SIZE.apply(lambda x: x.split()[0])


# In[24]:


# change data type of product size from string to float
product_data.PRODUCT_SIZE = product_data.PRODUCT_SIZE.astype(float)


# In[25]:


# Let's see the unique product size values for each category
product_data.groupby(['CATEGORY'])['PRODUCT_SIZE'].unique()


# In[ ]:


# define 3 bins for category type = "COLD CEREAL"
product_data.loc[product_data.CATEGORY == 'COLD CEREAL', 'PRODUCT_SIZE'] = pd.cut(product_data.PRODUCT_SIZE,
                                                                                 bins=[10,13,16,21],
                                                                                 labels=[1,2,3])


# In[ ]:


# define 2 bins for category type = "ORAL HYGIENE PRODUCTS"
product_data.loc[product_data.CATEGORY == 'ORAL HYGIENE PRODUCTS', 'PRODUCT_SIZE'] = pd.cut(product_data.PRODUCT_SIZE,
                                                                                            bins=[0,501,1001],
                                                                                            labels=[1,2])


# In[ ]:


# define 3 bins for category type = "FROZEN PIZZA"
product_data.loc[product_data.CATEGORY == 'FROZEN PIZZA', 'PRODUCT_SIZE'] = pd.cut(product_data.PRODUCT_SIZE,
                                                                                   bins=[20,25,30,35],
                                                                                   labels=[1,2,3])


# In[ ]:


# define 2 bins for category type = "BAG SNACKS"
product_data.loc[product_data.CATEGORY == 'BAG SNACKS', 'PRODUCT_SIZE'] = pd.cut(product_data.PRODUCT_SIZE,
                                                                                 bins=[9,14,20],
                                                                                 labels=[1,2])


# In[ ]:


# value counts of PRODUCT SIZE
product_data.PRODUCT_SIZE.value_counts()


# In[ ]:


# One Hot Encode the features
OHE_p = ce.OneHotEncoder(cols= ['MANUFACTURER', 'CATEGORY', 'SUB_CATEGORY'])


# In[ ]:


# transform the data
product_data = OHE_p.fit_transform(product_data)


# In[ ]:


# updated data
product_data.head()


# In[ ]:


# shape of the updated data
product_data.shape


# In[ ]:


# columns of the updated data
product_data.columns


# ---
# ### `DATASET 3: STORE DATA`
# 
# - **STORE_ID** - store number
# - **STORE_NAME** - Name of store
# - **ADDRESS_CITY_NAME** - city
# - **ADDRESS_STATE_PROV_CODE** - state
# - **MSA_CODE** - (Metropolitan Statistical Area) Based on geographic region and population density
# - **SEG_VALUE_NAME** - Store Segment Name
# - **PARKING_SPACE_QTY** - number of parking spaces in the store parking lot
# - **SALES_AREA_SIZE_NUM** - square footage of store
# - **AVG_WEEKLY_BASKETS** - average weekly baskets sold in the store
# 
# ---

# In[ ]:


# read the store data
store_data = pd.read_csv('dataset/store_data.csv')


# In[ ]:


store_data.head()


# ---
# 
# ### `STORE DATA`  has the following categorical features
# 
#     - STORE_ID
#     - STORE_NAME
#     - ADDRESS_CITY_NAME
#     - ADDRESS_STATE_PROV_CODE
#     - MSA_CODE
#     - SEG_VALUE_NAME
#     
# ---    

# In[ ]:


# shape of the store data
store_data.shape


# In[1]:


# check for the null values

store_data[['STORE_ID', 'STORE_NAME', 'ADDRESS_CITY_NAME', 'ADDRESS_STATE_PROV_CODE', 'MSA_CODE', 'SEG_VALUE_NAME']].isna().sum()


# In[ ]:


# number of unique store names
store_data.STORE_NAME.nunique()


# In[ ]:


# number of unique city names
store_data.ADDRESS_CITY_NAME.nunique()


# In[ ]:


# number of unique state provision code
store_data.ADDRESS_STATE_PROV_CODE.nunique()


# In[ ]:


# number of unique msa code
store_data.MSA_CODE.nunique()


# In[ ]:


# number of unique segment value names
store_data.SEG_VALUE_NAME.nunique()


# ---
# 
#    - `STORE_ID` - No changes required as it is a key and will be used to merge files later.
#    - `STORE_NAME` - Since, Out of 76 different stores we have 72 unique store names. Store name contains some location information of the store which we have in the form of address city name and state.
#    - `ADDRESS_CITY_NAME` - Since, Out of 76 different stores we have 51 unique address city names, So we will drop this feature due to high cardinality
#    - `ADDRESS_STATE_PROV_CODE`, `MSA_CODE` - As, there is no order in the given categories, So, we will One Hot Encode this variable.
#    - `SEG_VALUE_NAME` - Stores segments are divided into 3 categories: upscale, mainstream and value. Upscale stores are just what they sound like; they are normally located in high income neighborhoods and offer more high-end product. Mainstream is middle of the road, mostly located in middle class areas, offering a mix of upscale and value product. Value stores cater more to low income customers, so there will be more focus on low prices than anything else.
#    
#    So we will map `VALUE AS 1`, `MAINSTREAM AS 2` and `UPSCALE AS 3`.

# In[2]:


# drop store name and address
store_data = store_data.drop(columns=['STORE_NAME', 'ADDRESS_CITY_NAME'])


# In[3]:


# OneHotEncode the rest of the categorical features
OHE = ce.OneHotEncoder(cols=['ADDRESS_STATE_PROV_CODE', 'MSA_CODE'])

store_data.SEG_VALUE_NAME = store_data.SEG_VALUE_NAME.map({'VALUE': 1, 'MAINSTREAM' : 2, 'UPSCALE': 3})


# In[4]:


# transform the data
store_data = OHE.fit_transform(store_data)


# In[5]:


# updated data
store_data.head()


# In[6]:


# shape of the updated data
store_data.shape


# In[404]:


# columns of the updated data
store_data.columns


# In[405]:


store_data.loc[0]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ---
# ---
# 
# ## `PREPROCESSING: NUMERICAL FEATURES`
# 
# - Check and impute the missing values in the numerical features.
# - Check for the outliers and treat them.
# 
# ---

# ---
# 
# ### `DATASET 1: WEEKLY SALES DATA`
# 
# ---

# In[2]:


data.head()


# ---
# 
# ### `WEEKLY SALES DATA`  has the following numerical features
# 
#     - BASE_PRICE
#     - UNITS (Target)
#     
# ---    

#  - `BASE_PRICE` - Missing Value Imputation
# ---

# In[3]:


# check the null values for the numerical features
data[[ 'BASE_PRICE', 'UNITS']].isna().sum()


# ***Imputing the missing values in the Base Price***
# 
# ---

# In[4]:


# create a new dataframe which will have "average base price" for the combination of STORE_NUM and UPC
# we will use this to impute the missing values 
avg_price = data.groupby(['STORE_NUM', 'UPC'])['BASE_PRICE'].mean().reset_index()


# In[5]:


avg_price


# In[6]:


# null values in BASE PRICE
data.loc[data.BASE_PRICE.isna() == True]


# In[7]:


# define function to fill missing base price values
def fill_base_price(x) :
    return avg_price.BASE_PRICE[(avg_price.STORE_NUM == x['STORE_NUM']) & (avg_price.UPC == x['UPC'])].values[0]


# In[8]:


data.BASE_PRICE[data.BASE_PRICE.isna() == True] = data[data.BASE_PRICE.isna() == True].apply(fill_base_price, axis=1)


# In[9]:


# scatter plot for UNITS variable
# sort the target variable and scatter plot to see if it has some outliers or not.  

get_ipython().run_line_magic('matplotlib', 'notebook')
plt.figure(figsize=(8,6))
plt.scatter(x = range(data.shape[0]), y = np.sort(data['UNITS'].values))
plt.xlabel('Index', fontsize=12)
plt.ylabel('Units Sold', fontsize=12)
plt.show()


# In[ ]:


# number of data points where units are more than 750
data['UNITS'][data.UNITS > 750].shape[0]


# ---
# 
# ***We can see that, there are a some points above where UNITS are more than 750 and there number is only 21. So, we can remove them as there number is only 21 and will not affect the data and these will act as a noise to our model.***
# 
# ---

# In[415]:


data.shape


# In[416]:


# remove the valures where UNITS are more than 750
data = data[~(data.UNITS > 750)]


# In[417]:


data[data.UNITS > 750].shape[0]


# ---
# 
# ### `DATASET 2: PRODUCT DATA`
# 
# ---

# In[418]:


# view the product data
product_data.head()


# ---
# 
# ### `PRODUCT DATA`  has the following numerical feature
# 
#     - This dataset has no numerical feature.
#     
# ---    

# ---
# 
# ### `DATASET 3: STORE DATA`
# 
# ---

# In[419]:


# view the data
store_data.head()


# ---
# 
# ### `STORE DATA`  has the following numerical features
# 
#     - PARKING_SPACE_QTY
#     - SALES_AREA_SIZE_NUM
#     - AVG_WEEKLY_BASKETS
#     
# ---    

# In[420]:


# shape of the data
store_data.shape


# In[421]:


# check for the null values
store_data[['PARKING_SPACE_QTY', 'SALES_AREA_SIZE_NUM', 'AVG_WEEKLY_BASKETS']].isna().sum()


# ---
# - `PARKING_SPACE_QTY` - Check its correlation with the `SALES_AREA_SIZE_NUM`
# 
# ---

# In[422]:


# check correlation
store_data[['PARKING_SPACE_QTY','SALES_AREA_SIZE_NUM']].corr()


# ***Note:*** Since the correlation of the **PARKING_SPACE_QTY** with **SALES_AREA_SIZE_NUM** is high so we can drop this column as it will not add much value to the model.
# 
# ---

# In[423]:


# drop the column
store_data = store_data.drop(columns=['PARKING_SPACE_QTY'])


# ---
# ### `SAVE THE UPDATED FILES`
# 
# ---

# In[424]:


data.to_csv('updated_train_data.csv',index=False)
product_data.to_csv('updated_product_data.csv',index=False)
store_data.to_csv('updated_store_data.csv',index=False)

