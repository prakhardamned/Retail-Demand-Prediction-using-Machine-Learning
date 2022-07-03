#!/usr/bin/env python
# coding: utf-8

# # Notebook Structure

# 1. Problem Statement and Data Description
# 2. Loading Datasets and Libraries
# 3. Understanding and Validating the Data
# 4. Data Exploration - Train, Product, Store 

# # 1. Problem Statement and Data Description

# **Problem Statement:** 
# Prevent overstocking and understocking of Items by forecasting demand 
# of items for the next week, based on historical data.
# 
# 
# <img src = 'retail-shopping-business-wallpaper-preview.jpg' width = 500 height = 500>

# **Data Description:**
# 
# Train Data- 
# - **WEEK_END_DATE** - week ending date
# - **STORE_NUM** - store number
# - **UPC** - (Universal Product Code) product specific identifier
# - **BASE_PRICE** - base price of item
# - **DISPLAY** - product was a part of in-store promotional display
# - **FEATURE** - product was in in-store circular
# - **UNITS** - units sold (target)
# 
# Product Data-
# - **UPC** - (Universal Product Code) product specific identifier
# - **DESCRIPTION**	- product description
# - **MANUFACTURER** - product	manufacturer
# - **CATEGORY** - category of product
# - **SUB_CATEGORY** - sub-category of product
# - **PRODUCT_SIZE** - package size or quantity of product
# 
# Store Data-
# - **STORE_ID** - store number
# - **STORE_NAME** - Name of store
# - **ADDRESS_CITY_NAME** - city
# - **ADDRESS_STATE_PROV_CODE** - state
# - **MSA_CODE** - (Metropolitan Statistical Area) Based on geographic region and population density
# - **SEG_VALUE_NAME** - Store Segment Name
# - **PARKING_SPACE_QTY** - number of parking spaces in the store parking lot
# - **SALES_AREA_SIZE_NUM** - square footage of store
# - **AVG_WEEKLY_BASKETS** - average weekly baskets sold in the store

# # 2. Loading Required Libraries and Datasets

# In[1]:

import os 
import seaborn as sns
import pandas as pd
import numpy as np
import random

os.getcwd()
os.chdir("C:\\Use\\Retail Demand Prediction using Machine Learning")

sns.set_context('notebook',font_scale=1.5)

import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings("ignore")


# We are provided with three tables containing the required information:
# 
# - **product_data**: Consists of details about the product
# - **store_data**: Consists of details of various stores associated with the retailer  
# - **train**: Contains transaction data of products

# In[2]:


# reading the data files
train = pd.read_csv('train.csv')
product_data = pd.read_csv('product_data.csv')
store_data = pd.read_csv('store_data.csv')


# In[3]:


# checking the size of the dataframes
train.shape, product_data.shape, store_data.shape


# # 3. Understanding and Validating Data

# ### Train Data

# In[4]:


# printing first 5 rows of the train file
train.head(6)


# In[5]:


# checking datatypes of columns in train file 
train.dtypes

train.nunique(), train.dtypes 


# - WEEK_END_DATE has the data type object, but its a datetime variable 
# - The store number and product codes are read as int, but these are categorical variables.

# #### Datetime variable 
# 
# - The data is captured for what duration?
# - What are the start and end dates?
# - Is there any missing data points?
# 
# #### Numerical Variables
# 
# - Check the distribution of numerical variables
# - Are there any extreme values?
# - Are there any missing values in the variables?
# 
# #### Categorical Variables
# 
# - Check the unique values for categorical variables
# - Are there any missing values in the variables?
# - Is there any variable with high cardinality/ sparsity?
# 

# ##### WEEK_END_DATE
# 

# In[6]:


# convert into the date time format
train['WEEK_END_DATE'] = pd.to_datetime(train['WEEK_END_DATE'])


# In[7]:


train['WEEK_END_DATE'].isnull().sum()


# In[8]:


train['WEEK_END_DATE'].min(), train['WEEK_END_DATE'].max()


# - The data collected is from January 2009 to September 2011.
# 
# #### Are any dates missing from this period?

# In[8]:


(train['WEEK_END_DATE'].max() - train['WEEK_END_DATE'].min())/7
#Timedelta('141 days 00:00:00')

# In[12]:


train['WEEK_END_DATE'].nunique()


# - The training data is for 142 weeks, based on the number of unique *weekend dates* in the train file. 
# - No dates are missing from this period.
# 
# #### Are all dates at a gap of a week?

# In[13]:


train['WEEK_END_DATE'].dt.weekday_name.value_counts()


# ##### STORE_NUM  and UPC

# In[14]:


train[['STORE_NUM', 'UPC']].isnull().sum()

train.isnull().sum()
# In[15]:


train['STORE_NUM'].nunique()


# In[16]:


(train['STORE_NUM'].value_counts()).sort_values()


# - We have 76 unique stores.
# - Every store has minimum of 1676 transactions.

# #### Does each store hold atleast one entry per week?

# We have 76 unique stores and 142 weeks of data for the sales. If each store is selling occupies atleast one row in the data, the minimum number of unique rows should be 142*76

# In[15]:


142*76


# In[16]:


train[['WEEK_END_DATE','STORE_NUM']].drop_duplicates().shape


# - Implies that each store is atleast selling 1 product each week

# In[17]:


train['UPC'].nunique()


# In[18]:


(train['UPC'].value_counts()).sort_values()


# 
# #### Is every product sold atleast once, for all 142 weeks?

# In[19]:


142*30


# In[20]:


train[['WEEK_END_DATE','UPC']].drop_duplicates().shape


# - We have 30 unique products in the training data
# - There are 76 different stores associated with the retailer
# - Both the variables do not have any missing values

# #### Is each store selling each product throughout the given period?

# Assuming we have information for the sale of every product that is present in the product table (30), against each store associated (76), and for every week (142); we should have 142*76*30 data rows. 

# In[21]:


142*76*30


# In[22]:


train.shape


# In[23]:


232286/323760 


# - We can conclude that all stores are not selling all products each week
# - Of all the possible combinations, about 72% of the data is present

# #### For a store selling a particular product, do we have more than one entry?

# 
# Each product sold by any store should hold only one row, i.e. a particular store, say 'store A' selling a product 'prod P' should contribute a single row for every week. Let us check that.

# In[24]:


train.shape


# In[25]:


train[['WEEK_END_DATE','STORE_NUM','UPC']].drop_duplicates().shape


# In[26]:


train.groupby(['WEEK_END_DATE','STORE_NUM'])['UPC'].count().mean()


# - The shape does not change after using drop duplicates
# - Implies that there are unique combinations for week, store and UPC
# - On an average, each week we are selling 22 products

# #### Is a store selling a product throughout the period or is there a break?

# In[27]:


(train.groupby(['STORE_NUM', 'UPC'])['UNITS'].count()).sort_values()


# - Not all stores sell a product throughout the week
# - The minimum number is 137/142

# We now have a basic understanding of the number of products and stores we are dealing with in this data.

# ##### BASE_PRICE

# In[17]:


train['BASE_PRICE'].isnull().sum()


# In[29]:


train['BASE_PRICE'].describe()


# In[18]:


# distribution of Base Price variable
plt.figure(figsize=(8,6))
sns.distplot((train['BASE_PRICE'].values), bins=20, kde=True)
plt.xlabel('Price Distribution', fontsize=12)
plt.show()


# - No extreme values in the base price variable
# - Range for base price is 1 dollar to 8 dollars

# ##### FEATURE and DISPLAY

# In[19]:


train[['FEATURE','DISPLAY']].isnull().sum()


# In[20]:


train[['FEATURE','DISPLAY']].dtypes


# In[21]:


train[['FEATURE','DISPLAY']].nunique()


# In[22]:


train['FEATURE'].value_counts(normalize=True)


# In[35]:


train['FEATURE'].value_counts(normalize=True).plot('bar')


# - Approximately 10 percent of product are featured

# In[23]:


train['DISPLAY'].value_counts(normalize=True)


# In[78]:


train['DISPLAY'].value_counts(normalize=True).plot(kind ='bar')


# - About 13% of products are on display

# In[79]:


pd.crosstab(train['FEATURE'], train['DISPLAY']).apply(lambda r: r/len(train), axis=1)


# ##### UNITS

# In[28]:


train['UNITS'].isnull().sum()


# In[29]:


# basic statistical details of UNITS variable
train['UNITS'].describe()


# - The Range of values is very high
# - Minimum number of units sold is 0 and maximum is 1800
# - A huge difference between the 75th percentile and the max value indicates presence of outliers

# #### How many rows in the data have 0 units sold?
# #### Is there only one row with such high sales of 1800? 

# In[30]:


train[train['UNITS'] == 0]


# - Only one entry with 0 items sold
# - Indicates the given store does not sell the following item
# - It's simply a Data Anomaly and will not be useful in model training

# In[31]:


# keeping rows with UNITS sold not equal to zero
train = train[train['UNITS'] != 0]


# In[43]:


# scatter plot for UNITS variable
plt.figure(figsize=(8,6))
plt.scatter(x = range(train.shape[0]), y = np.sort(train['UNITS'].values))
plt.xlabel('Index', fontsize=12)
plt.ylabel('Units Sold', fontsize=12)
plt.show()


# - Most of the values are less than 250 
# - There are a few outliers (with 1 outlier way outside the range)

# In[32]:


train[train['UNITS'] > 1000]


# To reduce the effect of outliers and for better visualization, here is a log transform of the variable
# 

# In[33]:


# distribution of UNITS variable
plt.figure(figsize=(8,6))
sns.distplot((train['UNITS'].values), bins=25, kde=True)
plt.xlabel('Units Sold', fontsize=12)
plt.show()


# In[46]:


# log transformed UNITS column
plt.figure(figsize=(8,6))
sns.distplot(np.log(train['UNITS'].values), bins=25, kde=True)
plt.xlabel('Log Units Sold', fontsize=12)
plt.show()


# - After log transformation, the distribution looks closer to a normal distribution

# ### Understanding Product Data

# In[34]:


# first five rows of product data
product_data.head()


# In[48]:


product_data.dtypes


# #### Categorical Variables
# 
# - Check the unique values for categorical variables
# - Are there any missing values in the variables?
# - Is there any variable with high cardinality/ sparsity?

# ##### UPC 

# In[36]:


product_data['UPC'].nunique()


# - The number is consistent through the train and product data.
# ##### Are all the product codes exactly the same?

# In[37]:


len(set(product_data.UPC).intersection(set(train.UPC)))


# ##### CATEGORY

# In[38]:


# number and list of unique categories in the product data
product_data['CATEGORY'].nunique(), product_data['CATEGORY'].unique()


# In[39]:


product_data['CATEGORY'].isnull().sum()


# In[40]:


product_data['CATEGORY'].value_counts()


# - We have four product categories - 
#   *  BAG SNACKS
#   *  ORAL HYGIENE PRODUCTS 
#   *  COLD CEREAL 
#   *  FROZEN PIZZA
# 
# - There are 9 products with the category 'Cold Cereal'
# - Similarly, 8 products labeled 'Bag snacks', 7 with category 'Frozen Pizza' and 6 'Oral Hygiene' Products

# #### Is there any subdivision among the product categories?

# #####  SUB_CATEGORY 

# In[41]:


product_data['SUB_CATEGORY'].isnull().sum()


# In[55]:


product_data['SUB_CATEGORY'].nunique()


# In[42]:


# displaying subcategories against each category
product_data[['CATEGORY','SUB_CATEGORY']].drop_duplicates().sort_values(by = 'CATEGORY')


# The sub-categories give additional detail about the product.
# 
# - Cereal has 3 sub categories, differentiating on the age group 
# - Oral hygiene products have 2 sub categories, antiseptic and rinse/spray
# - Bag Snacks & Frozen Pizza have just 1 sub category, no further division
# 
# 
# #### Does the sub category has anything to do with the size of the product?

# #####  	PRODUCT_SIZE

# In[43]:


# unique category, sub-category and product size combinations
product_data[['CATEGORY','SUB_CATEGORY','PRODUCT_SIZE']].drop_duplicates().sort_values(by = 'CATEGORY')


# The cold cereal for kids is available in two different sizes.
# Also, the cold cereal for all family has the same size as the cold cereal for kids.
# Hence subcategory is not an indicator of size.

# **To summarize**
# - Bag Snacks has 1 sub category and 3 product size available
# - Oral Hygiene product has 2 sub categories and 2 size options
# - Frozen Pizza has only 1 sub category and 6 different package size
# - cold ceral has 3 sub categoeies, and 6 options in size

# ##### DESCRIPTION 

# In[44]:


product_data['DESCRIPTION'].isnull().sum()


# In[45]:


# number and list of unique descriptions in the prodcut data
product_data['DESCRIPTION'].nunique(), product_data['DESCRIPTION'].unique()


# - We have 29 descriptions in the dataset, for 30 products.
# - Almost all products have a unique description. 

# In[46]:


(product_data['DESCRIPTION'].value_counts())


# In[47]:


product_data.loc[product_data['DESCRIPTION']=='GM CHEERIOS']


# In[48]:


product_data.loc[product_data['UPC'] == 1600027527]


# - More granular description for the product
# - Includes the type of product and manufacturer

# ##### MANUFACTURER

# 
# #### How many Manufacturers/ suppliers are we associated with?
# 
# #### Are same products created by multiple manufacturers?
# 

# In[63]:


product_data['MANUFACTURER'].isnull().sum()


# In[64]:


product_data['MANUFACTURER'].nunique()


# In[50]:


# displaying the list of manufacturers against the 4 categories
temp = product_data[['CATEGORY','MANUFACTURER']].drop_duplicates()
pd.crosstab([temp['CATEGORY']], temp['MANUFACTURER'])


# - We have 4 unique categories of Products
# - Each category has three manufacturers
# - Every category has a manufacturer 'private label' (and 2 other manufacturers)

# In[ ]:





# In[ ]:





# ### Understanding Store Data

# In[66]:


store_data.head()


# In[67]:


store_data.dtypes


# #### Numerical Variables
# 
# - Check the distribution of numerical variables
# - Are there any extreme values?
# - Are there any missing values in the variables?
# 
# #### Categorical Variables
# 
# - Check the unique values for categorical variables
# - Are there any missing values in the variables?
# - Is there any variable with high cardinality/ sparsity?

# ##### STORE_ID

# In[68]:


store_data['STORE_ID'].nunique()


# In[69]:


len(set(store_data.STORE_ID).intersection(set(train.STORE_NUM)))


# ##### STORE_NAME

# In[70]:


store_data['STORE_NAME'].isnull().sum()


# In[71]:


store_data['STORE_NAME'].nunique()


# - The number of unique store IDs is more than number of unique store names
# - There might be stores with same name, located in different city

# #### Which store name is being repeated?
# 
# #### Why do some stores have same name and different ID?

# In[72]:


# number of store names repeating
store_data['STORE_NAME'].value_counts()


# In[51]:


store_data.loc[store_data['STORE_NAME'] == 'HOUSTON']


# In[74]:


store_data.loc[store_data['STORE_NAME'] == 'MIDDLETOWN']


# The store names that are repeated, are actually different stores which either have a different city or different segment (upscale, mainstream, value) or  location. Hence they are given a different IDs. 

# ##### ADDRESS_CITY_NAME  and ADDRESS_STATE_PROV_CODE

# In[75]:


store_data[['ADDRESS_STATE_PROV_CODE', 'ADDRESS_CITY_NAME']].isnull().sum()


# #### How many cities and states are the stores located in?

# In[76]:


store_data[['ADDRESS_STATE_PROV_CODE', 'ADDRESS_CITY_NAME']].nunique()


# <img src = 'texas-to-ohio-map-map-of-arizona-and-california-cities-california-map-major-cities-of-texas-to-ohio-map.jpg' width = 700 height = 700>

# Let's find out the number of stores in each of the state 

# In[52]:


store_data.groupby(['ADDRESS_STATE_PROV_CODE'])['STORE_ID'].count()


# - Each store has a unique store ID 
# - Most stores are from Ohio and Texas ~93%
# - Few from Kentucky and Indiana ~7%

# In[78]:


store_data.groupby(['ADDRESS_STATE_PROV_CODE'])['ADDRESS_CITY_NAME'].nunique()


# In[79]:


store_data['ADDRESS_CITY_NAME'].value_counts()


# ##### MSA_CODE

# In[80]:


store_data['MSA_CODE'].isnull().sum()


# In[53]:


store_data['MSA_CODE'].nunique(), store_data['MSA_CODE'].unique()


# In[82]:


store_data['MSA_CODE'].value_counts()


# In[54]:


(store_data.groupby(['MSA_CODE', 'ADDRESS_STATE_PROV_CODE'])['STORE_ID'].count())


# - These codes are assigned based on the geographical location and population density. 
# - 17140 is present in all three except Texas (which has a different geographical region)

# ##### PARKING_SPACE_QTY  and SALES_AREA_SIZE_NUM 

# In[84]:


store_data[['PARKING_SPACE_QTY', 'SALES_AREA_SIZE_NUM']].isnull().sum()


# - Of 76 stores, parking area of 51 is missing

# In[85]:


plt.figure(figsize=(8,6))
sns.distplot(store_data['PARKING_SPACE_QTY'], bins=25, kde=False)
plt.xlabel('Parking Area Size', fontsize=12)
plt.show()


# - About 15 stores have parking area between 250 - 500 units

# In[86]:


plt.figure(figsize=(8,6))
sns.distplot(store_data['SALES_AREA_SIZE_NUM'], bins=30, kde=True)
plt.xlabel('Sales Area Size (Sq Feet)', fontsize=12)
plt.show()


# - Most stores have the area between 30-70 K
# - Only a small number of stores have area less than 30k or greater than 90k

# #### How is Average store size varying for different states?

# In[87]:


(store_data.groupby(['ADDRESS_STATE_PROV_CODE'])['SALES_AREA_SIZE_NUM'].mean()).sort_values(ascending=False)


# In[88]:


state_oh = store_data.loc[store_data['ADDRESS_STATE_PROV_CODE'] == 'OH']
state_tx = store_data.loc[store_data['ADDRESS_STATE_PROV_CODE'] == 'TX']

sns.distplot(state_oh['SALES_AREA_SIZE_NUM'], hist=False,color= 'dodgerblue', label= 'OHIO')
sns.distplot(state_tx['SALES_AREA_SIZE_NUM'], hist=False,  color= 'orange', label= 'TEXAS')


# - Indiana has only one store and the area size is 58,563 sq feet. 
# - Ohio and Texas have average around 52k and 50k. 
# - Ohio has stores distributed at all sizes.
# - Texas mainly has stores between sales area 30k to 60k 

# ##### AVG_WEEKLY_BASKETS

# In[89]:


store_data['AVG_WEEKLY_BASKETS'].isnull().sum()


# In[90]:


store_data['AVG_WEEKLY_BASKETS'].describe()


# In[91]:


plt.figure(figsize=(8,6))
sns.distplot(store_data['AVG_WEEKLY_BASKETS'], bins=30, kde=True)
plt.xlabel('Average Baskets sold per week', fontsize=12)
plt.show()


# #### What are the average weekly baskets sold for the states? 

# In[92]:


(store_data.groupby(['ADDRESS_STATE_PROV_CODE'])['AVG_WEEKLY_BASKETS'].mean()).sort_values(ascending=False)


# ##### SEG_VALUE_NAME

# In[93]:


store_data['SEG_VALUE_NAME'].isnull().sum()


# There are certain segments assigned to store, based on the brand and quality of products sold at the store.
# 
# - **Upscale stores** : Located in high income neighborhoods and offer more high-end product
# - **Mainstream stores** : Located in middle class areas, offering a mix of upscale and value product
# - **Value stores** : Focus on low prices products targeting low income customers

# Let us look at the distribution of stores in each of these segments

# In[94]:


store_data['SEG_VALUE_NAME'].value_counts()


# #### Does the segment has any relation with the store area?
# 
# #### Is there a difference in the average sales for each segment?

# In[95]:


(store_data.groupby(['SEG_VALUE_NAME'])['SALES_AREA_SIZE_NUM'].mean()).sort_values(ascending=False)


# In[96]:


(store_data.groupby(['SEG_VALUE_NAME'])['AVG_WEEKLY_BASKETS'].mean()).sort_values(ascending=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # 4. Data Exploration - Train, Product, Store

# ## Validating the Hypothesis

# During the Hypothesis Generation, we listed down the following hypothesis. 

# **Product Data**
# - Product type/ Category : Different Product Categories can have significantly varying trends/patterns
# - Product Size : Larger products should be more in demand 
# - Price of Product: Same category products with lower price would have more sales
# - Company/ Manufacturer: Well known brands/manufacturers will have higher sales

# **Train Data**
# 
# - Offer Applicable: Featured Products with attractive offers will have higher sales
# - Product Promotion: Sales will be more for products with in-store promotion

# **Store Data**
# - Store Location: Stores in a particular state/city will have a similar trend
# - Size of Store: Stores with larger area would have more sales
# - Average Wait time: If average baskets sold is higher, wait time would be low. Implies higher sale of units. 

# ### Merging the Store and Product Datasets

# In[55]:


store_product_data = train.merge(product_data, how = 'left', on='UPC')

store_product_data = store_product_data.merge(store_data, how = 'left', left_on = 'STORE_NUM', right_on = 'STORE_ID')


# In[56]:


store_product_data.shape


# In[99]:


store_product_data.columns


# ### Trend or Seasonal Pattern in Product Sales
# 
# - Product type/Category: Different Product Categories can have significantly varying trends/patterns
# 

# ### Units sold per week

# In[57]:


#sum of units sold per week
weekly_demand = store_product_data.groupby(['WEEK_END_DATE'])['UNITS'].sum()

plt.figure(figsize=(30,10))
sns.lineplot(x = weekly_demand.index, y = weekly_demand)


# - Displays the total number of units sold by the retailer (including all products and from all stores)
# - The highest number is close to 80,000 and lowest is close to 20,000 units
# - There is no evident pattern or trend in the plot
# - The spikes can be seen in either direction and at no constant interval

# ### Units sold per week - at product level
# 
# Now, we will look at category wise sales or demand patterns to see if there is any similarity within each category

# In[ ]:


#Here we are creating a function to plot weekly sales. We start of by creating a dictionary d which stores upc and weekly sales.
#We use the key of the dictionary, i.e, the product and add manufacturer and description to the the title and finally use seaborn to plot it.


# In[58]:


# function to plot weekly sales of products
def product_plots(product_list):
    
    # dictionary storing UPC and weekly sales
    d = {product: store_product_data[store_product_data['UPC'] == product].groupby(['WEEK_END_DATE'])['UNITS'].sum() for product in product_list}
    fig, axs = plt.subplots(len(product_list), 1, figsize = (20, 20), dpi=300)
    j = 0
    
    for product in d.keys():
        # adding manufacturer and descritption in title
        manu = product_data[product_data['UPC'] == product]['MANUFACTURER'].values[0]
        desc = product_data[product_data['UPC'] == product]['DESCRIPTION'].values[0]            
        # creating the plot
        sns.lineplot(x = d[product].index, y = d[product],ax = axs[j]).set_title(str(manu)+str(" ")+str(desc), y=0.75, fontsize = 16)
        j = j+1
    plt.tight_layout()


# In[59]:


# creating list of products based on category
pretzels = list(product_data[product_data['CATEGORY'] == 'BAG SNACKS']['UPC'])
frozen_pizza = list(product_data[product_data['CATEGORY'] == 'FROZEN PIZZA']['UPC'])
oral_hygiene = list(product_data[product_data['CATEGORY'] == 'ORAL HYGIENE PRODUCTS']['UPC'])
cold_cereal = list(product_data[product_data['CATEGORY'] == 'COLD CEREAL']['UPC'])


# In[103]:


product_plots(pretzels)


# In[104]:


product_plots(frozen_pizza)


# - No increasing/decreasing trend for the sale of products over time
# - No seasonal patterns seen on individual product sale
# - Products by same manufaturer have similar patterns (spikes and drops).
# 

# ### Units sold per week - at store level
# Now, let us look store level demand patterns to see if there are any patterns here.

# In[105]:


# Randomly selecting 5 store ID
stores_plot = random.sample(list(store_data['STORE_ID']), 5)


# In[106]:


#creating dictionary with store number as keys
# for each store, calculate sum of units sold per week
d = {store: train[train['STORE_NUM'] == store].groupby(['WEEK_END_DATE'])['UNITS'].sum() for store in stores_plot}


# In[107]:


plt.figure(figsize=(30,10))

fig, axs = plt.subplots(5, 1, figsize = (15, 15), dpi=300)
j = 0
for store in d.keys():
    sns.lineplot(x = d[store].index, y = d[store],ax = axs[j])
    j = j+1


# For the randomly selected store numbers, we can see that there is no pattern in the plot. The same was repeated for a number of stores and the data showed no increasing or decreasing trend or seasonality. 

# #### Are the sudden increase in sales due to product/in-store promotion?

# ### Featured or Displayed Product have higher sale
# 
# - Offer Applicable: Featured Products with attractive offers will have higher sales
# - Product Promotion: Sales will be more for products with in-store promotion

# In[108]:


def featured_plots(product_list):
    #dictionary storing UPC and 'Featured' variable
    d_f = {product: 1000*train[train['UPC'] == product].groupby(['WEEK_END_DATE'])['FEATURE'].mean() for product in product_list}
    #dictionary storing UPC and Product Sales
    d = {product: train[train['UPC'] == product].groupby(['WEEK_END_DATE'])['UNITS'].sum() for product in product_list}
    
    
    fig, axs = plt.subplots(len(product_list), 1, figsize = (20, 20), dpi=300)
    j = 0
    for product in d.keys():
        # Manufacturer name and Descritption in title
        manu = product_data[product_data['UPC'] == product]['MANUFACTURER'].values[0]
        desc = product_data[product_data['UPC'] == product]['DESCRIPTION'].values[0]
        
        # plotting featured and sales values
        sns.lineplot(x = d_f[product].index, y = d_f[product],ax = axs[j]).set_title(str(manu)+str(" ")+str(desc), y=0.75, fontsize = 16)
        sns.lineplot(x = d[product].index, y = d[product],ax = axs[j]).set_title(str(manu)+str(" ")+str(desc), y=0.75, fontsize = 16)
        j = j+1


# In[109]:


product_list_f = list(product_data[product_data['CATEGORY'] == 'BAG SNACKS']['UPC'])


# In[110]:


featured_plots(product_list_f)


# - When the products are featured, the sales increase.
# 
# #### Does the in-store display also have a similar effect?

# In[61]:


def display_plots(product_list):
    d_d = {product: 1000*train[train['UPC'] == product].groupby(['WEEK_END_DATE'])['DISPLAY'].mean() for product in product_list}
    d = {product: train[train['UPC'] == product].groupby(['WEEK_END_DATE'])['UNITS'].sum() for product in product_list}
    fig, axs = plt.subplots(len(product_list), 1, figsize = (20, 20), dpi=300)
    j = 0
    for product in d.keys():
        manu = product_data[product_data['UPC'] == product]['MANUFACTURER'].values[0]
        desc = product_data[product_data['UPC'] == product]['DESCRIPTION'].values[0]
        sns.lineplot(x = d[product].index, y = d[product],ax = axs[j]).set_title(str(manu)+str(" ")+str(desc), y=0.75, fontsize = 16)
        sns.lineplot(x = d_d[product].index, y = d_d[product],ax = axs[j]).set_title(str(manu)+str(" ")+str(desc), y=0.75, fontsize = 16)
        j = j+1


# In[112]:


display_plots(product_list_f)


# - It is evident that product sales are greatly affected by the display.
# - For products on display, the sales are higher.

# ### Product sales higher for lower priced items
# 
# - Price of Product: Same category products with lower price would have more sales
# 

# In[62]:


product_size_coldcereal = store_product_data.loc[store_product_data['CATEGORY']=='COLD CEREAL']
product_size_bagsnacks  = store_product_data.loc[store_product_data['CATEGORY']=='BAG SNACKS']
product_size_frozenpizza = store_product_data.loc[store_product_data['CATEGORY']=='FROZEN PIZZA']
product_size_oralhyiegne = store_product_data.loc[store_product_data['CATEGORY']=='ORAL HYGIENE PRODUCTS']


# In[63]:


# scatter plot for base price and sales
plt.figure(figsize=(8,6))
plt.scatter(x = (product_size_bagsnacks['BASE_PRICE']), y = (product_size_bagsnacks['UNITS']))
plt.xlabel('BASE_PRICE', fontsize=12)
plt.ylabel('UNITS', fontsize=12)
plt.show()


# In[64]:


# scatter plot for base price and sales
plt.figure(figsize=(8,6))
plt.scatter(x = (product_size_oralhyiegne['BASE_PRICE']), y = (product_size_oralhyiegne['UNITS']))
plt.xlabel('BASE_PRICE', fontsize=12)
plt.ylabel('UNITS', fontsize=12)
plt.show()


# In[65]:


# scatter plot for base price and sales
plt.figure(figsize=(8,6))
plt.scatter(x = (product_size_frozenpizza['BASE_PRICE']), y = (product_size_frozenpizza['UNITS']))
plt.xlabel('BASE_PRICE', fontsize=12)
plt.ylabel('UNITS', fontsize=12)
plt.show()


# In[67]:


# scatter plot for base price and sales
plt.figure(figsize=(8,6))
plt.scatter(x = (product_size_coldcereal['BASE_PRICE']), y = (product_size_coldcereal['UNITS']))
plt.xlabel('BASE_PRICE', fontsize=12)
plt.ylabel('UNITS', fontsize=12)
plt.show()


# - For bag snacks and oral hygiene category, items with lower price show a higher sale. 
# - Frozen pizza items have higher sale for higher price items.
# 
# Check the pattern for cold cereal at your end.

# ## Product size versus Product Sales
# 
# - Product Size : Larger products should be more in demand 
# 

# In[119]:


pd.crosstab(product_size_coldcereal['CATEGORY'], product_size_coldcereal['PRODUCT_SIZE'])


# In[68]:


pd.crosstab(product_size_bagsnacks['CATEGORY'], product_size_bagsnacks['PRODUCT_SIZE'])


# #### Is the product sale higher for a particular brand or manufacturer?

# ## Product sales for different manufacturers
# 
# - Company/ Manufacturer: Well known brands/manufacturers will have higher sales
# 

# In[69]:


pretzels = list(product_data[product_data['CATEGORY'] == 'BAG SNACKS']['UPC'])
frozen_pizza = list(product_data[product_data['CATEGORY'] == 'FROZEN PIZZA']['UPC'])
oral_hygiene = list(product_data[product_data['CATEGORY'] == 'ORAL HYGIENE PRODUCTS']['UPC'])
cold_cereal = list(product_data[product_data['CATEGORY'] == 'COLD CEREAL']['UPC'])


# In[122]:


plt.figure(figsize=(20,6))
ax = sns.boxplot(x="UPC", y="BASE_PRICE", data=train[train['UPC'].isin(pretzels)])
product_data[product_data['UPC'].isin(pretzels)]


# - All Private Label snacks have lower price. 
# - The Snyder S bag snacks with a smaller size has a lower price.

# In[70]:


plt.figure(figsize=(20,6))
ax = sns.boxplot(x="MANUFACTURER", y="UNITS", data=store_product_data[store_product_data['UPC'].isin(pretzels)])


# In[71]:


plt.figure(figsize=(20,6))
ax = sns.boxplot(x="MANUFACTURER", y="UNITS", data=store_product_data[store_product_data['UPC'].isin(cold_cereal)])


# In[72]:


plt.figure(figsize=(20,6))
ax = sns.boxplot(x="MANUFACTURER", y="UNITS", data=store_product_data[store_product_data['UPC'].isin(oral_hygiene)])


# In[73]:


plt.figure(figsize=(20,6))
ax = sns.boxplot(x="MANUFACTURER", y="UNITS", data=store_product_data[store_product_data['UPC'].isin(frozen_pizza)])


# #### Is there a significant difference in the product sales for different regions?
# 
# #### How are the sales different for stores in different cities?

# ## Unit Sales for Stores in Different States
# 
# - Store Location: Stores in a particular state/city will have a similar trend

# In[74]:


grouped_weekly_sales = store_product_data.groupby(['WEEK_END_DATE','STORE_NUM'])['UNITS'].sum().reset_index()

grouped_weekly_sales = grouped_weekly_sales.merge(store_data, how = 'left', left_on = 'STORE_NUM', right_on = 'STORE_ID')

grouped_weekly_sales = grouped_weekly_sales.sort_values(by = 'ADDRESS_STATE_PROV_CODE')


# In[75]:


state = (store_data[['ADDRESS_STATE_PROV_CODE','STORE_ID']].sort_values(by ='ADDRESS_STATE_PROV_CODE'))['STORE_ID']


# In[126]:


plt.figure(figsize=(50,15))

ax=sns.boxplot(x="STORE_NUM",y="UNITS",data=grouped_weekly_sales, hue ='ADDRESS_STATE_PROV_CODE', order =state)
plt.xticks(rotation=45)


# - The most frequent colors we see are green and orange - Ohio and Texas
# - Mostly the number of units is higher for Ohio (considering individual stores)

# ## Store Size and unit sales
# 
# - Size of Store: Stores with larger area would have more sales

# In[76]:


store_agg_data = train.groupby(['STORE_NUM'])['UNITS'].sum().reset_index()
merged_store_data = store_data.merge(store_agg_data, how = 'left', left_on = 'STORE_ID', right_on = 'STORE_NUM')


# In[128]:


state_oh = merged_store_data.loc[merged_store_data['ADDRESS_STATE_PROV_CODE'] == 'OH']
state_tx = merged_store_data.loc[merged_store_data['ADDRESS_STATE_PROV_CODE'] == 'TX']

sns.distplot(state_oh['SALES_AREA_SIZE_NUM'], hist=False,color= 'dodgerblue', label= 'OHIO')
sns.distplot(state_tx['SALES_AREA_SIZE_NUM'], hist=False,  color= 'orange', label= 'TEXAS')


# In[77]:


sns.scatterplot(x = (state_oh['SALES_AREA_SIZE_NUM']), y = (state_oh['UNITS']))
sns.scatterplot(x = (state_tx['SALES_AREA_SIZE_NUM']), y = (state_tx['UNITS']))


# In[ ]:


Assignment
1. Does the "Display" have more impact on demand than "Feature" or is it the other way around. 

Yes Display has more impact on demand than features, we can see in plot made for product_list_f.


# In[ ]:


2. Does the number of unique manifactures for a category & sub category 
(Bag snack, Frozen pizza etc.) impact the demand.


# In[ ]:


store_data.groupby(['ADDRESS_STATE_PROV_CODE'])['STORE_ID'].count()

