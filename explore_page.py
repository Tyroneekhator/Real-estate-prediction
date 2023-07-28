import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from sklearn import tree
import seaborn as sns
import streamlit as st
from sklearn.model_selection import GridSearchCV,\
RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
matplotlib.rcParams["figure.figsize"] = (10,10)

df = pd.read_csv(r"C:\Users\ekhat\PycharmProjects\pythonProject7777\nigeria_houses_data.csv")
df.head()
#%%
df.drop(['toilets'], axis=1, inplace=True)


# removing outliers
def outliers(df, ft):
    q1 = df[ft].quantile(0.25)
    q3 = df[ft].quantile(0.75)
    IQR = q3 - q1
    min = q1 - 1.5 * IQR
    max = q3 + 1.5 * IQR

    ls = df.index[(df[ft] < min) | (df[ft] > max)]
    return ls
#%%
index_list=[]
for outlier in ['bedrooms','bathrooms','parking_space','price']:
    index_list.extend(outliers(df,outlier))
def remove(df,ls):
    ls=sorted(set(ls))
    df=df.drop(ls)
    return df
df2 = remove(df,index_list)
df2.rename(columns = {'title':'house_type'}, inplace = True)
df2['total_rooms'] = df2['bedrooms'] + df2['bathrooms']
df2['price_per_room'] = df2['price']/df2['total_rooms']
#function can remove extreme cases based on mean and standard deviation(how it is spread)
def remove_pps_outliers(df): #dataframe
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('state'): #group by location
        m = np.mean(subdf.price_per_room)
        st = np.std(subdf.price_per_room)
        reduced_df = subdf[(subdf.price_per_room>(m-st)) & (subdf.price_per_room<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index = True)
    return df_out

df3 = remove_pps_outliers(df2)
df3.shape
few_state_records = df3["state"].value_counts() # remove few records
#%%
few_town_records =df3['town'].value_counts()
df4 = df3[~df3['state'].isin(few_state_records[few_state_records < 10].index) & ~df3['town'].isin(few_town_records[few_town_records < 10].index)]

obj = (df4.dtypes == 'object')
object_cols = list(obj[obj].index)
print("Categorical variables:", len(object_cols))

int_ = (df4.dtypes == 'int')
num_cols = list(int_[int_].index)
print("Integer variables:", len(num_cols))

fl = (df4.dtypes == 'float')
fl_cols = list(fl[fl].index)
print("Float variables:", len(fl_cols))

# corrs = df4.corr()
# corrs # no coreelation for parking space

#%%
df4.drop(['parking_space'], axis=1, inplace=True)

def show_explore_page():


    st.title("EDA on house prices in Nigeria")


#################################################################
    fig, ax = plt.subplots()
    (df4.groupby("town")['price'].mean() / 1e6).sort_values(ascending=False).head(20).plot(kind="bar",xlabel='Towns',ylabel="Prices [₦1M]",title="Most Expensive Real Estate Towns in Nigeria")
    st.write(fig)
    st.write('---')
#################################################################
    # States with the highest mean prices
    fig, ax = plt.subplots()
    (df4.groupby("state")["price"].mean() / 1e6).sort_values(ascending=False).plot(kind="bar",xlabel="States",ylabel="Prices [₦1M]",title="Average House Price by State")
    st.write(fig)
    st.write('---')

################################################################
    fig, ax = plt.subplots()
    plt.legend(loc=2)
    (df4.groupby("house_type")["price"].mean() / 1e6).sort_values(ascending=False).plot(kind="barh", xlabel="Prices [₦1M]",ylabel='House Type',title="Average Price by Type of House")

    st.write(fig)
    st.write('---')





#####################################################################
    fig, ax = plt.subplots()
    plt.xlabel("Price Per Room [₦1M]")
    plt.ylabel("Price [₦1M]")
    sns.scatterplot(data=df4, hue='house_type', x=df4['price_per_room'] / 1e6, y=df4['price'] / 1e6)
    plt.legend(loc=2)
    st.write(fig)
###########################################################################



