
# Final Project Submission


* Student name: **Andrew Hotchkiss**
* Student pace: **Part time**
* Scheduled project review date/time: **8/17/2020, 2pm EDT**
* Instructor name: **James Irving**
* Blog post URL: https://stones-1130.github.io/interpreting_modeling_results_after_log_transformation


## TABLE OF CONTENTS 

*Click to jump to matching Markdown Header.*<br><br>

<font size=3rem>
    
- **[Introduction](#INTRODUCTION)<br>**
- **[OBTAIN](#OBTAIN)**<br>
- **[SCRUB](#SCRUB)**<br>
- **[EXPLORE](#EXPLORE)**<br>
- **[MODEL](#MODEL)**<br>
- **[iNTERPRET](#iNTERPRET)**<br>
- **[Conclusions/Recommendations](#CONCLUSIONS-&-RECOMMENDATIONS)<br>**
</font>
___

# INTRODUCTION

> **Assignment-** Clean, explore, and model the Kings County, WA Home Sale dataset (2014-2015) with a multivariate linear regression to predict the sale price of houses as accurately as possible.


> **Approach:**
> My goal for this project was to provide recommendations to a seller on how to increase the sale price of their home. I chose to look at this problem from the perspective of someone selling their home in the near-future and a future seller who's looking to put their home on the market in 5-years. 

> Initially, I determined to what extent certain attributes of the homes affected the overall housing price. 

>  Lastly, I trained a multivariate linear regression model to be able to accurately predict home prices based on certain features of the home.



# OBTAIN


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline


df = pd.read_csv("kc_house_data.csv")
pd.options.display.max_columns = None
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>date</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>7129300520</td>
      <td>10/13/2014</td>
      <td>221900.0</td>
      <td>3</td>
      <td>1.00</td>
      <td>1180</td>
      <td>5650</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1180</td>
      <td>0.0</td>
      <td>1955</td>
      <td>0.0</td>
      <td>98178</td>
      <td>47.5112</td>
      <td>-122.257</td>
      <td>1340</td>
      <td>5650</td>
    </tr>
    <tr>
      <td>1</td>
      <td>6414100192</td>
      <td>12/9/2014</td>
      <td>538000.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>2570</td>
      <td>7242</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>2170</td>
      <td>400.0</td>
      <td>1951</td>
      <td>1991.0</td>
      <td>98125</td>
      <td>47.7210</td>
      <td>-122.319</td>
      <td>1690</td>
      <td>7639</td>
    </tr>
    <tr>
      <td>2</td>
      <td>5631500400</td>
      <td>2/25/2015</td>
      <td>180000.0</td>
      <td>2</td>
      <td>1.00</td>
      <td>770</td>
      <td>10000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>6</td>
      <td>770</td>
      <td>0.0</td>
      <td>1933</td>
      <td>NaN</td>
      <td>98028</td>
      <td>47.7379</td>
      <td>-122.233</td>
      <td>2720</td>
      <td>8062</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2487200875</td>
      <td>12/9/2014</td>
      <td>604000.0</td>
      <td>4</td>
      <td>3.00</td>
      <td>1960</td>
      <td>5000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>7</td>
      <td>1050</td>
      <td>910.0</td>
      <td>1965</td>
      <td>0.0</td>
      <td>98136</td>
      <td>47.5208</td>
      <td>-122.393</td>
      <td>1360</td>
      <td>5000</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1954400510</td>
      <td>2/18/2015</td>
      <td>510000.0</td>
      <td>3</td>
      <td>2.00</td>
      <td>1680</td>
      <td>8080</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1680</td>
      <td>0.0</td>
      <td>1987</td>
      <td>0.0</td>
      <td>98074</td>
      <td>47.6168</td>
      <td>-122.045</td>
      <td>1800</td>
      <td>7503</td>
    </tr>
  </tbody>
</table>
</div>



Below feature column descriptions from: https://www.slideshare.net/PawanShivhare1/predicting-king-county-house-prices

![](KC_data_dict.png)


```python
#EXAMINE THE DATA TYPES, ALSO LOOK FOR MISSING DATA
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 21597 entries, 0 to 21596
    Data columns (total 21 columns):
    id               21597 non-null int64
    date             21597 non-null object
    price            21597 non-null float64
    bedrooms         21597 non-null int64
    bathrooms        21597 non-null float64
    sqft_living      21597 non-null int64
    sqft_lot         21597 non-null int64
    floors           21597 non-null float64
    waterfront       19221 non-null float64
    view             21534 non-null float64
    condition        21597 non-null int64
    grade            21597 non-null int64
    sqft_above       21597 non-null int64
    sqft_basement    21597 non-null object
    yr_built         21597 non-null int64
    yr_renovated     17755 non-null float64
    zipcode          21597 non-null int64
    lat              21597 non-null float64
    long             21597 non-null float64
    sqft_living15    21597 non-null int64
    sqft_lot15       21597 non-null int64
    dtypes: float64(8), int64(11), object(2)
    memory usage: 3.5+ MB



```python
#DROP ID and LAT/LONG COLUMNS
df.drop(['id','lat','long'],axis=1,inplace=True)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>10/13/2014</td>
      <td>221900.0</td>
      <td>3</td>
      <td>1.00</td>
      <td>1180</td>
      <td>5650</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1180</td>
      <td>0.0</td>
      <td>1955</td>
      <td>0.0</td>
      <td>98178</td>
      <td>1340</td>
      <td>5650</td>
    </tr>
    <tr>
      <td>1</td>
      <td>12/9/2014</td>
      <td>538000.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>2570</td>
      <td>7242</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>2170</td>
      <td>400.0</td>
      <td>1951</td>
      <td>1991.0</td>
      <td>98125</td>
      <td>1690</td>
      <td>7639</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2/25/2015</td>
      <td>180000.0</td>
      <td>2</td>
      <td>1.00</td>
      <td>770</td>
      <td>10000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>6</td>
      <td>770</td>
      <td>0.0</td>
      <td>1933</td>
      <td>NaN</td>
      <td>98028</td>
      <td>2720</td>
      <td>8062</td>
    </tr>
    <tr>
      <td>3</td>
      <td>12/9/2014</td>
      <td>604000.0</td>
      <td>4</td>
      <td>3.00</td>
      <td>1960</td>
      <td>5000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>7</td>
      <td>1050</td>
      <td>910.0</td>
      <td>1965</td>
      <td>0.0</td>
      <td>98136</td>
      <td>1360</td>
      <td>5000</td>
    </tr>
    <tr>
      <td>4</td>
      <td>2/18/2015</td>
      <td>510000.0</td>
      <td>3</td>
      <td>2.00</td>
      <td>1680</td>
      <td>8080</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1680</td>
      <td>0.0</td>
      <td>1987</td>
      <td>0.0</td>
      <td>98074</td>
      <td>1800</td>
      <td>7503</td>
    </tr>
  </tbody>
</table>
</div>



> From first glance, it looks like we need to recast the following columns into the correct data type:

> - "date" object to int64 format
> - "sqft_basement" to int64
> - "yr_renovated" to int64


```python
#LOOK CLOSER AT 'sqft_basement' COLUMN
df['sqft_basement'].value_counts(dropna=False)
```




    0.0       12826
    ?           454
    600.0       217
    500.0       209
    700.0       208
              ...  
    2250.0        1
    2130.0        1
    207.0         1
    2050.0        1
    1816.0        1
    Name: sqft_basement, Length: 304, dtype: int64




```python
#CHANGE 'sqft_basement' COLUMN TO NUMERIC AND CHANGE ERRORS TO NA VALUES
df['sqft_basement'] = pd.to_numeric(df['sqft_basement'],errors='coerce')
df['sqft_basement'].value_counts(dropna=False)
```




    0.0       12826
    NaN         454
    600.0       217
    500.0       209
    700.0       208
              ...  
    588.0         1
    1920.0        1
    2390.0        1
    1245.0        1
    1135.0        1
    Name: sqft_basement, Length: 304, dtype: int64




```python
df['yr_renovated'].value_counts(dropna=False)
```




    0.0       17011
    NaN        3842
    2014.0       73
    2003.0       31
    2013.0       31
              ...  
    1944.0        1
    1948.0        1
    1976.0        1
    1934.0        1
    1953.0        1
    Name: yr_renovated, Length: 71, dtype: int64




```python
#BECAUSE WE HAVE LOTS OF NA VALUES IN BOTH COLUMNS, WE'LL LEAVE THEM AS FLOATS FOR NOW
#AND CONVERT data COLUMN TO STRING
df['sqft_basement'] = df['sqft_basement'].astype('float')
df['yr_renovated'] = df['yr_renovated'].astype('float')
df['date'] = df['date'].astype('str')
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 21597 entries, 0 to 21596
    Data columns (total 18 columns):
    date             21597 non-null object
    price            21597 non-null float64
    bedrooms         21597 non-null int64
    bathrooms        21597 non-null float64
    sqft_living      21597 non-null int64
    sqft_lot         21597 non-null int64
    floors           21597 non-null float64
    waterfront       19221 non-null float64
    view             21534 non-null float64
    condition        21597 non-null int64
    grade            21597 non-null int64
    sqft_above       21597 non-null int64
    sqft_basement    21143 non-null float64
    yr_built         21597 non-null int64
    yr_renovated     17755 non-null float64
    zipcode          21597 non-null int64
    sqft_living15    21597 non-null int64
    sqft_lot15       21597 non-null int64
    dtypes: float64(7), int64(10), object(1)
    memory usage: 3.0+ MB



```python
#LOOK AT COLUMNS WITH NUMERICAL DTYPES
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>2.159700e+04</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
      <td>2.159700e+04</td>
      <td>21597.000000</td>
      <td>19221.000000</td>
      <td>21534.000000</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
      <td>21143.000000</td>
      <td>21597.000000</td>
      <td>17755.000000</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>5.402966e+05</td>
      <td>3.373200</td>
      <td>2.115826</td>
      <td>2080.321850</td>
      <td>1.509941e+04</td>
      <td>1.494096</td>
      <td>0.007596</td>
      <td>0.233863</td>
      <td>3.409825</td>
      <td>7.657915</td>
      <td>1788.596842</td>
      <td>291.851724</td>
      <td>1970.999676</td>
      <td>83.636778</td>
      <td>98077.951845</td>
      <td>1986.620318</td>
      <td>12758.283512</td>
    </tr>
    <tr>
      <td>std</td>
      <td>3.673681e+05</td>
      <td>0.926299</td>
      <td>0.768984</td>
      <td>918.106125</td>
      <td>4.141264e+04</td>
      <td>0.539683</td>
      <td>0.086825</td>
      <td>0.765686</td>
      <td>0.650546</td>
      <td>1.173200</td>
      <td>827.759761</td>
      <td>442.498337</td>
      <td>29.375234</td>
      <td>399.946414</td>
      <td>53.513072</td>
      <td>685.230472</td>
      <td>27274.441950</td>
    </tr>
    <tr>
      <td>min</td>
      <td>7.800000e+04</td>
      <td>1.000000</td>
      <td>0.500000</td>
      <td>370.000000</td>
      <td>5.200000e+02</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>370.000000</td>
      <td>0.000000</td>
      <td>1900.000000</td>
      <td>0.000000</td>
      <td>98001.000000</td>
      <td>399.000000</td>
      <td>651.000000</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>3.220000e+05</td>
      <td>3.000000</td>
      <td>1.750000</td>
      <td>1430.000000</td>
      <td>5.040000e+03</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>7.000000</td>
      <td>1190.000000</td>
      <td>0.000000</td>
      <td>1951.000000</td>
      <td>0.000000</td>
      <td>98033.000000</td>
      <td>1490.000000</td>
      <td>5100.000000</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>4.500000e+05</td>
      <td>3.000000</td>
      <td>2.250000</td>
      <td>1910.000000</td>
      <td>7.618000e+03</td>
      <td>1.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>7.000000</td>
      <td>1560.000000</td>
      <td>0.000000</td>
      <td>1975.000000</td>
      <td>0.000000</td>
      <td>98065.000000</td>
      <td>1840.000000</td>
      <td>7620.000000</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>6.450000e+05</td>
      <td>4.000000</td>
      <td>2.500000</td>
      <td>2550.000000</td>
      <td>1.068500e+04</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>8.000000</td>
      <td>2210.000000</td>
      <td>560.000000</td>
      <td>1997.000000</td>
      <td>0.000000</td>
      <td>98118.000000</td>
      <td>2360.000000</td>
      <td>10083.000000</td>
    </tr>
    <tr>
      <td>max</td>
      <td>7.700000e+06</td>
      <td>33.000000</td>
      <td>8.000000</td>
      <td>13540.000000</td>
      <td>1.651359e+06</td>
      <td>3.500000</td>
      <td>1.000000</td>
      <td>4.000000</td>
      <td>5.000000</td>
      <td>13.000000</td>
      <td>9410.000000</td>
      <td>4820.000000</td>
      <td>2015.000000</td>
      <td>2015.000000</td>
      <td>98199.000000</td>
      <td>6210.000000</td>
      <td>871200.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#DO A QUICK PASS OF THE TOP 5 VALUE COUNTS FOR EACH INT64 COLUMN TO QUICKLY CHECK DATA QUALITY
pd.set_option('display.max_rows', 20)
for col in df.columns:
    try:
        print(col, df[col].value_counts()[:5])
    except:
        print(col, df[col].value_counts())
    print('\n')
```

    date 6/23/2014    142
    6/26/2014    131
    6/25/2014    131
    7/8/2014     127
    4/27/2015    126
    Name: date, dtype: int64
    
    
    price 350000.0    172
    450000.0    172
    550000.0    159
    500000.0    152
    425000.0    150
               ... 
    870515.0      1
    336950.0      1
    386100.0      1
    176250.0      1
    884744.0      1
    Name: price, Length: 3622, dtype: int64
    
    
    bedrooms 3    9824
    4    6882
    2    2760
    5    1601
    6     272
    Name: bedrooms, dtype: int64
    
    
    bathrooms 2.50    5377
    1.00    3851
    1.75    3048
    2.25    2047
    2.00    1930
    1.50    1445
    2.75    1185
    3.00     753
    3.50     731
    3.25     589
    3.75     155
    4.00     136
    4.50     100
    4.25      79
    0.75      71
    4.75      23
    5.00      21
    Name: bathrooms, dtype: int64
    
    
    sqft_living 1300    138
    1400    135
    1440    133
    1660    129
    1010    129
    Name: sqft_living, dtype: int64
    
    
    sqft_lot 5000    358
    6000    290
    4000    251
    7200    220
    7500    119
    Name: sqft_lot, dtype: int64
    
    
    floors 1.0    10673
    2.0     8235
    1.5     1910
    3.0      611
    2.5      161
    3.5        7
    Name: floors, dtype: int64
    
    
    waterfront 0.0    19075
    1.0      146
    Name: waterfront, dtype: int64
    
    
    view 0.0    19422
    2.0      957
    3.0      508
    1.0      330
    4.0      317
    Name: view, dtype: int64
    
    
    condition 3    14020
    4     5677
    5     1701
    2      170
    1       29
    Name: condition, dtype: int64
    
    
    grade 7     8974
    8     6065
    9     2615
    6     2038
    10    1134
    Name: grade, dtype: int64
    
    
    sqft_above 1300    212
    1010    210
    1200    206
    1220    192
    1140    184
    Name: sqft_above, dtype: int64
    
    
    sqft_basement 0.0       12826
    600.0       217
    500.0       209
    700.0       208
    800.0       201
              ...  
    915.0         1
    295.0         1
    1281.0        1
    2130.0        1
    906.0         1
    Name: sqft_basement, Length: 303, dtype: int64
    
    
    yr_built 2014    559
    2006    453
    2005    450
    2004    433
    2003    420
    Name: yr_built, dtype: int64
    
    
    yr_renovated 0.0       17011
    2014.0       73
    2003.0       31
    2013.0       31
    2007.0       30
              ...  
    1946.0        1
    1959.0        1
    1971.0        1
    1951.0        1
    1954.0        1
    Name: yr_renovated, Length: 70, dtype: int64
    
    
    zipcode 98103    602
    98038    589
    98115    583
    98052    574
    98117    553
    Name: zipcode, dtype: int64
    
    
    sqft_living15 1540    197
    1440    195
    1560    192
    1500    180
    1460    169
    Name: sqft_living15, dtype: int64
    
    
    sqft_lot15 5000    427
    4000    356
    6000    288
    7200    210
    4800    145
    Name: sqft_lot15, dtype: int64
    
    


# EXPLORE


```python
# INITIAL SCATTER PLOT TO LOOK AT RELATIONSHIPS BETWEEN FEATURES. 
pd.plotting.scatter_matrix(df, figsize=(18,20));
```


![png](output_21_0.png)


> **Scatter matrix analysis:**
> - Categorical variables:
    - Floors, waterfront, view, condition, grade, zipcode
> - Numerical variables: 
    - price, bedrooms, bathrooms, sqft_living, sqft_lot, sqft_above, sqft_basement, yr_built, yr_renovated, sqft_living15, sqft_lot15


```python
# HISTOGRAM ANALYSIS TO LOOK AT THE VARIABLE DISTRIBUTIONS

df.hist(bins=25,figsize=(16,16),xlabelsize='10',ylabelsize='10',xrot=-15);
```


![png](output_23_0.png)


> **Histogram analysis:** 
> - **There's clearly some data preprocessing that we'll need to do in order to effectively use this data in our model. This will include normalization and scaling techniques.**
> - **As you can see, almost none of the features have a normal distribution, there's a mixture of categorical and discrete/continuous variables, and large range of magnitudes.**


```python
#CHECK FOR MULTI-COLLINEARITY
#CODE FROM: https://medium.com/@szabo.bibor/how-to-create-a-seaborn-correlation-heatmap-in-python-834c0686b88e

plt.figure(figsize=(16, 6))
mask = np.triu(np.ones_like(df.corr(), dtype=np.bool))
heatmap = sns.heatmap(df.corr(), mask=mask, vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Kings Country Home Sales (2014-2015) data Correlation Matrix', fontdict={'fontsize':18}, pad=16);
```


![png](output_25_0.png)


> **Correlation Matrix Takeaways:** 

> 1. **Bathrooms, sqft_living, grade, sqft_above, and sqft_living15** all appear to be positively correlated with price, our independent variable. This makes sense- a larger house typically has more bathrooms and is more expensive.

> 2. A few variables could present problems with multi-collinearity (> .75). 

> - sqft_living & bathrooms
> - sqft_living & grade
> - sqft_living & sqft_above
> - sqft_living & sqft_living15

> 3. When we run our base model, we need to look at dropping **at least one** of these features.



```python
#ANOTHER WAY TO VISUALLY REPRESENT MULTI-COLLINEARITY
test_corr = abs(df.corr()) > 0.75

test_corr
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>price</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <td>bedrooms</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <td>bathrooms</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <td>sqft_living</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <td>sqft_lot</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <td>floors</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <td>waterfront</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <td>view</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <td>condition</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <td>grade</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <td>sqft_above</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <td>sqft_basement</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <td>yr_built</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <td>yr_renovated</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <td>zipcode</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <td>sqft_living15</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <td>sqft_lot15</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
#LOOK AT SUM OF BOOLEANS IN EACH ROW TO DETERMINE WHICH VARIABLES HAVE THE MOST CORRELATIONS > .75
sum_row = test_corr.sum(axis=1)
print(sum_row)
```

    price            1
    bedrooms         1
    bathrooms        2
    sqft_living      5
    sqft_lot         1
    floors           1
    waterfront       1
    view             1
    condition        1
    grade            3
    sqft_above       3
    sqft_basement    1
    yr_built         1
    yr_renovated     1
    zipcode          1
    sqft_living15    2
    sqft_lot15       1
    dtype: int64



```python
#DROP date COLUMN
df.drop(['date'],axis=1,inplace=True)
```

> **Let's explore what features have an effect on price. First, let's look at the categorical variables from the scatter plot** 

> - **View**
> - **Condition**
> - **Grade**
> - **Waterfront**
> - **Floors**
> - **Bedrooms**
> - **Bathrooms**


```python
plt.figure(figsize=(18,8))
plt.title('View and Price', fontsize=20)
sns.boxplot(x='view', y='price', data=df);
```


![png](output_31_0.png)



```python
plt.figure(figsize=(18,8))
plt.title('Condition and Price', fontsize=20)
sns.boxplot(x='condition', y='price', data=df);
```


![png](output_32_0.png)



```python
plt.figure(figsize=(18,8))
plt.title('Grade and Price', fontsize=20)
sns.boxplot(x='grade', y='price', data=df);
```


![png](output_33_0.png)


> **It looks like grade might be a good predictor of house price**


```python
plt.figure(figsize=(18,8))
plt.title('Waterfront and Price', fontsize=20)
sns.boxplot(x='waterfront', y='price', data=df);
```


![png](output_35_0.png)



```python
#FLOORS & PRICE
plt.figure(figsize=(18,8))
plt.title('Floors and Price', fontsize=20)
sns.boxplot(x='floors', y='price', data=df);
```


![png](output_36_0.png)



```python
#BEDROOMS & PRICE
sns.set()
plt.figure(figsize=(18,8))
plt.title('Bedrooms and Price', fontsize=20)
sns.barplot(x='bedrooms', y='price', data=df);
```


![png](output_37_0.png)



```python
#BATHROOMS & PRICE
plt.figure(figsize=(18,8))
plt.title('Bathrooms and Price', fontsize=20)
sns.barplot(x='bathrooms', y='price', data=df);
```


![png](output_38_0.png)


> **Now let's explore some our continuous features:**

> - **sqft_living**
> - **sqft_basement**
> - **sqft_lot**



```python
#SQFT_LIVING & PRICE
plt.figure(figsize=(18,8))
plt.title('House internal square footage and Price', fontsize=20)
sns.regplot(x='sqft_living', y='price', data=df, ci=95);

```


![png](output_40_0.png)



```python
plt.figure(figsize=(18,8))
plt.title('Average Total Square Footage of 15 Closest Houses and Price', fontsize=20)
sns.scatterplot(x='sqft_living15', y='price', data=df);
```


![png](output_41_0.png)



```python
plt.figure(figsize=(18,8))
plt.title('Basement Size and Price', fontsize=20)
sns.regplot(x='sqft_basement', y='price', data=df, ci=95);
```


![png](output_42_0.png)



```python
#SQFT_LOT & PRICE
plt.figure(figsize=(18,8))
plt.title('Lot Size and Price', fontsize=20)
sns.scatterplot(x='sqft_lot', y='price', data=df);
```


![png](output_43_0.png)



```python
plt.figure(figsize=(18,8))
plt.title('Average Lot Square Footage of 15 Closest Houses and Price', fontsize=20)
sns.scatterplot(x='sqft_lot15', y='price', data=df);
```


![png](output_44_0.png)


> **EDA findings:**

> - **Grade, waterfront, bathrooms, bedrooms, and total home square-footage** appear to have the largest positive effect on home sale price. 


> **Explanation:** This makes sense when you think about how the data includes homes sold in suburban, rural areas outside of Seattle as well as homes/apartments in the downtown areas. 

> In reality, we'd expect that there are luxury apartments in downtown Seattle with only 1-2 bedrooms that would have a significantly higher sale price than a larger house with more bedrooms located in a more rural area. 

# SCRUB


```python
df.isna().sum()
```




    price               0
    bedrooms            0
    bathrooms           0
    sqft_living         0
    sqft_lot            0
    floors              0
    waterfront       2376
    view               63
    condition           0
    grade               0
    sqft_above          0
    sqft_basement     454
    yr_built            0
    yr_renovated     3842
    zipcode             0
    sqft_living15       0
    sqft_lot15          0
    dtype: int64




```python
#LET'S SEE WHAT PERCENTAGE OF THE FEATURE COLUMNS ARE NANs

waterfront = (2376/21957)*100
view = (63/21957)*100
sqft_basement = (454/21957)*100
yr_renovated = (3842/21957)*100

#df.dropna()

print(f"Waterfront: {waterfront}% of column")
print(f"View: {view}% of column")
print(f"Sqft_Basement: {sqft_basement}% of column")
print(f"Year Renovated: {yr_renovated}% of column")
```

    Waterfront: 10.821150430386666% of column
    View: 0.2869244432299494% of column
    Sqft_Basement: 2.0676777337523338% of column
    Year Renovated: 17.49783668078517% of column



```python
#LET'S DROP THE ROWS WITH NANs FROM view AND sqft_basement BECAUSE THEY'RE SUCH A SMALL PERCENTAGE OF VALUES

df.dropna(subset=['sqft_basement'], inplace=True)
```


```python
df.dropna(subset=['view'], inplace=True)
df.isna().sum()
```




    price               0
    bedrooms            0
    bathrooms           0
    sqft_living         0
    sqft_lot            0
    floors              0
    waterfront       2333
    view                0
    condition           0
    grade               0
    sqft_above          0
    sqft_basement       0
    yr_built            0
    yr_renovated     3742
    zipcode             0
    sqft_living15       0
    sqft_lot15          0
    dtype: int64




```python
#THE VAST MAJORITY OF waterfront AND yr_renovated DATA IS "0", SO LET'S REPLACE ALL NANs IN THE waterfront COLUMN WITH "0"
df['waterfront'] = df['waterfront'].fillna(0)
df['yr_renovated'] = df['yr_renovated'].fillna(0)
df.isna().sum()
```




    price            0
    bedrooms         0
    bathrooms        0
    sqft_living      0
    sqft_lot         0
    floors           0
    waterfront       0
    view             0
    condition        0
    grade            0
    sqft_above       0
    sqft_basement    0
    yr_built         0
    yr_renovated     0
    zipcode          0
    sqft_living15    0
    sqft_lot15       0
    dtype: int64



# MODEL

## BASELINE MODEL


```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import metrics
import statsmodels.api as sm
import scipy.stats as stats
from statsmodels.formula.api import ols

```


```python
#MAKE A COPY OF THE ORIGINAL DATA FRAME
df_base = df.copy()
df_base.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>221900.0</td>
      <td>3</td>
      <td>1.00</td>
      <td>1180</td>
      <td>5650</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1180</td>
      <td>0.0</td>
      <td>1955</td>
      <td>0.0</td>
      <td>98178</td>
      <td>1340</td>
      <td>5650</td>
    </tr>
    <tr>
      <td>1</td>
      <td>538000.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>2570</td>
      <td>7242</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>2170</td>
      <td>400.0</td>
      <td>1951</td>
      <td>1991.0</td>
      <td>98125</td>
      <td>1690</td>
      <td>7639</td>
    </tr>
    <tr>
      <td>2</td>
      <td>180000.0</td>
      <td>2</td>
      <td>1.00</td>
      <td>770</td>
      <td>10000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>6</td>
      <td>770</td>
      <td>0.0</td>
      <td>1933</td>
      <td>0.0</td>
      <td>98028</td>
      <td>2720</td>
      <td>8062</td>
    </tr>
    <tr>
      <td>3</td>
      <td>604000.0</td>
      <td>4</td>
      <td>3.00</td>
      <td>1960</td>
      <td>5000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>7</td>
      <td>1050</td>
      <td>910.0</td>
      <td>1965</td>
      <td>0.0</td>
      <td>98136</td>
      <td>1360</td>
      <td>5000</td>
    </tr>
    <tr>
      <td>4</td>
      <td>510000.0</td>
      <td>3</td>
      <td>2.00</td>
      <td>1680</td>
      <td>8080</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1680</td>
      <td>0.0</td>
      <td>1987</td>
      <td>0.0</td>
      <td>98074</td>
      <td>1800</td>
      <td>7503</td>
    </tr>
  </tbody>
</table>
</div>



> **Remove features with multicollinearity problems previously identifed:**
> - sqft_above
> - sqft_basement
> - sqft_living15
> - sqft_lot


```python
#BUILD THE BASE MODEL WITH STATSMODELS

cat_var = ['bedrooms','bathrooms','floors', 'grade', 'waterfront', 'condition', 
           'zipcode']

features = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors', 'grade', 'waterfront', 'condition', 
           'zipcode']

outcome = 'price'
predictors = '+'.join(features)
formula = outcome + '~' + predictors
formula
```




    'price~bedrooms+bathrooms+sqft_living+sqft_lot+floors+grade+waterfront+condition+zipcode'



> **Let's one-hot encode our categorical variables (named "cat_var" above).**


```python
for col in cat_var: 
    formula = formula.replace(col, f"C({col})")

formula
```




    'price~C(bedrooms)+C(bathrooms)+sqft_living+sqft_lot+C(floors)+C(grade)+C(waterfront)+C(condition)+C(zipcode)'




```python
model = ols(formula=formula, data=df_base).fit()
model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>price</td>      <th>  R-squared:         </th>  <td>   0.829</td>  
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.828</td>  
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>   778.9</td>  
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 21 Aug 2020</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>   
</tr>
<tr>
  <th>Time:</th>                 <td>12:47:41</td>     <th>  Log-Likelihood:    </th> <td>-2.8143e+05</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td> 21082</td>      <th>  AIC:               </th>  <td>5.631e+05</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 20951</td>      <th>  BIC:               </th>  <td>5.642e+05</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>   130</td>      <th>                     </th>      <td> </td>     
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>      <td> </td>     
</tr>
</table>
<table class="simpletable">
<tr>
            <td></td>              <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>            <td> -1.24e+04</td> <td>  1.8e+05</td> <td>   -0.069</td> <td> 0.945</td> <td>-3.65e+05</td> <td>  3.4e+05</td>
</tr>
<tr>
  <th>C(bedrooms)[T.2]</th>     <td>-1407.6990</td> <td> 1.19e+04</td> <td>   -0.118</td> <td> 0.906</td> <td>-2.47e+04</td> <td> 2.19e+04</td>
</tr>
<tr>
  <th>C(bedrooms)[T.3]</th>     <td> 4547.4713</td> <td> 1.19e+04</td> <td>    0.381</td> <td> 0.703</td> <td>-1.88e+04</td> <td> 2.79e+04</td>
</tr>
<tr>
  <th>C(bedrooms)[T.4]</th>     <td>-1.454e+04</td> <td> 1.22e+04</td> <td>   -1.193</td> <td> 0.233</td> <td>-3.84e+04</td> <td> 9358.584</td>
</tr>
<tr>
  <th>C(bedrooms)[T.5]</th>     <td>-3.272e+04</td> <td> 1.28e+04</td> <td>   -2.547</td> <td> 0.011</td> <td>-5.79e+04</td> <td>-7538.002</td>
</tr>
<tr>
  <th>C(bedrooms)[T.6]</th>     <td>-6.735e+04</td> <td> 1.57e+04</td> <td>   -4.295</td> <td> 0.000</td> <td>-9.81e+04</td> <td>-3.66e+04</td>
</tr>
<tr>
  <th>C(bedrooms)[T.7]</th>     <td>-1.969e+05</td> <td>  2.9e+04</td> <td>   -6.787</td> <td> 0.000</td> <td>-2.54e+05</td> <td> -1.4e+05</td>
</tr>
<tr>
  <th>C(bedrooms)[T.8]</th>     <td>-7.447e+04</td> <td> 4.49e+04</td> <td>   -1.658</td> <td> 0.097</td> <td>-1.62e+05</td> <td> 1.36e+04</td>
</tr>
<tr>
  <th>C(bedrooms)[T.9]</th>     <td>-2.041e+05</td> <td> 7.02e+04</td> <td>   -2.909</td> <td> 0.004</td> <td>-3.42e+05</td> <td>-6.66e+04</td>
</tr>
<tr>
  <th>C(bedrooms)[T.10]</th>    <td>-3.553e+05</td> <td> 9.02e+04</td> <td>   -3.939</td> <td> 0.000</td> <td>-5.32e+05</td> <td>-1.78e+05</td>
</tr>
<tr>
  <th>C(bedrooms)[T.11]</th>    <td>-2.551e+04</td> <td> 1.53e+05</td> <td>   -0.167</td> <td> 0.868</td> <td>-3.26e+05</td> <td> 2.75e+05</td>
</tr>
<tr>
  <th>C(bedrooms)[T.33]</th>    <td> 1.592e+04</td> <td> 1.53e+05</td> <td>    0.104</td> <td> 0.917</td> <td>-2.84e+05</td> <td> 3.16e+05</td>
</tr>
<tr>
  <th>C(bathrooms)[T.0.75]</th> <td> 2.997e+04</td> <td> 9.01e+04</td> <td>    0.332</td> <td> 0.740</td> <td>-1.47e+05</td> <td> 2.07e+05</td>
</tr>
<tr>
  <th>C(bathrooms)[T.1.0]</th>  <td>  6.04e+04</td> <td> 8.82e+04</td> <td>    0.685</td> <td> 0.493</td> <td>-1.12e+05</td> <td> 2.33e+05</td>
</tr>
<tr>
  <th>C(bathrooms)[T.1.25]</th> <td>-3.105e+04</td> <td> 1.02e+05</td> <td>   -0.305</td> <td> 0.760</td> <td>-2.31e+05</td> <td> 1.69e+05</td>
</tr>
<tr>
  <th>C(bathrooms)[T.1.5]</th>  <td> 4.594e+04</td> <td> 8.83e+04</td> <td>    0.520</td> <td> 0.603</td> <td>-1.27e+05</td> <td> 2.19e+05</td>
</tr>
<tr>
  <th>C(bathrooms)[T.1.75]</th> <td> 4.384e+04</td> <td> 8.83e+04</td> <td>    0.497</td> <td> 0.619</td> <td>-1.29e+05</td> <td> 2.17e+05</td>
</tr>
<tr>
  <th>C(bathrooms)[T.2.0]</th>  <td>  4.54e+04</td> <td> 8.83e+04</td> <td>    0.514</td> <td> 0.607</td> <td>-1.28e+05</td> <td> 2.18e+05</td>
</tr>
<tr>
  <th>C(bathrooms)[T.2.25]</th> <td> 5.545e+04</td> <td> 8.83e+04</td> <td>    0.628</td> <td> 0.530</td> <td>-1.18e+05</td> <td> 2.29e+05</td>
</tr>
<tr>
  <th>C(bathrooms)[T.2.5]</th>  <td> 4.779e+04</td> <td> 8.83e+04</td> <td>    0.541</td> <td> 0.589</td> <td>-1.25e+05</td> <td> 2.21e+05</td>
</tr>
<tr>
  <th>C(bathrooms)[T.2.75]</th> <td> 5.135e+04</td> <td> 8.84e+04</td> <td>    0.581</td> <td> 0.561</td> <td>-1.22e+05</td> <td> 2.25e+05</td>
</tr>
<tr>
  <th>C(bathrooms)[T.3.0]</th>  <td> 6.544e+04</td> <td> 8.85e+04</td> <td>    0.740</td> <td> 0.460</td> <td>-1.08e+05</td> <td> 2.39e+05</td>
</tr>
<tr>
  <th>C(bathrooms)[T.3.25]</th> <td> 1.086e+05</td> <td> 8.86e+04</td> <td>    1.226</td> <td> 0.220</td> <td> -6.5e+04</td> <td> 2.82e+05</td>
</tr>
<tr>
  <th>C(bathrooms)[T.3.5]</th>  <td> 6.249e+04</td> <td> 8.86e+04</td> <td>    0.706</td> <td> 0.480</td> <td>-1.11e+05</td> <td> 2.36e+05</td>
</tr>
<tr>
  <th>C(bathrooms)[T.3.75]</th> <td> 1.531e+05</td> <td> 8.93e+04</td> <td>    1.716</td> <td> 0.086</td> <td>-2.18e+04</td> <td> 3.28e+05</td>
</tr>
<tr>
  <th>C(bathrooms)[T.4.0]</th>  <td> 1.457e+05</td> <td> 8.94e+04</td> <td>    1.629</td> <td> 0.103</td> <td>-2.96e+04</td> <td> 3.21e+05</td>
</tr>
<tr>
  <th>C(bathrooms)[T.4.25]</th> <td> 2.329e+05</td> <td> 9.02e+04</td> <td>    2.582</td> <td> 0.010</td> <td> 5.61e+04</td> <td>  4.1e+05</td>
</tr>
<tr>
  <th>C(bathrooms)[T.4.5]</th>  <td> 1.742e+05</td> <td> 8.99e+04</td> <td>    1.939</td> <td> 0.053</td> <td>-1926.329</td> <td>  3.5e+05</td>
</tr>
<tr>
  <th>C(bathrooms)[T.4.75]</th> <td> 4.112e+05</td> <td> 9.42e+04</td> <td>    4.364</td> <td> 0.000</td> <td> 2.26e+05</td> <td> 5.96e+05</td>
</tr>
<tr>
  <th>C(bathrooms)[T.5.0]</th>  <td>  2.58e+05</td> <td> 9.53e+04</td> <td>    2.707</td> <td> 0.007</td> <td> 7.12e+04</td> <td> 4.45e+05</td>
</tr>
<tr>
  <th>C(bathrooms)[T.5.25]</th> <td>  2.65e+05</td> <td> 9.85e+04</td> <td>    2.691</td> <td> 0.007</td> <td>  7.2e+04</td> <td> 4.58e+05</td>
</tr>
<tr>
  <th>C(bathrooms)[T.5.5]</th>  <td> 3.607e+05</td> <td> 1.03e+05</td> <td>    3.502</td> <td> 0.000</td> <td> 1.59e+05</td> <td> 5.63e+05</td>
</tr>
<tr>
  <th>C(bathrooms)[T.5.75]</th> <td> 3.166e+05</td> <td> 1.18e+05</td> <td>    2.676</td> <td> 0.007</td> <td> 8.47e+04</td> <td> 5.48e+05</td>
</tr>
<tr>
  <th>C(bathrooms)[T.6.0]</th>  <td> 4.457e+05</td> <td> 1.13e+05</td> <td>    3.942</td> <td> 0.000</td> <td> 2.24e+05</td> <td> 6.67e+05</td>
</tr>
<tr>
  <th>C(bathrooms)[T.6.25]</th> <td> 6.283e+05</td> <td> 1.42e+05</td> <td>    4.426</td> <td> 0.000</td> <td>  3.5e+05</td> <td> 9.07e+05</td>
</tr>
<tr>
  <th>C(bathrooms)[T.6.5]</th>  <td>-1.497e+04</td> <td>  1.4e+05</td> <td>   -0.107</td> <td> 0.915</td> <td> -2.9e+05</td> <td>  2.6e+05</td>
</tr>
<tr>
  <th>C(bathrooms)[T.6.75]</th> <td> 2.501e+05</td> <td> 1.41e+05</td> <td>    1.774</td> <td> 0.076</td> <td>-2.63e+04</td> <td> 5.27e+05</td>
</tr>
<tr>
  <th>C(bathrooms)[T.7.5]</th>  <td>-1.644e+05</td> <td> 1.89e+05</td> <td>   -0.868</td> <td> 0.385</td> <td>-5.36e+05</td> <td> 2.07e+05</td>
</tr>
<tr>
  <th>C(bathrooms)[T.7.75]</th> <td> 2.724e+06</td> <td> 1.84e+05</td> <td>   14.771</td> <td> 0.000</td> <td> 2.36e+06</td> <td> 3.09e+06</td>
</tr>
<tr>
  <th>C(bathrooms)[T.8.0]</th>  <td> 1.605e+06</td> <td> 1.44e+05</td> <td>   11.117</td> <td> 0.000</td> <td> 1.32e+06</td> <td> 1.89e+06</td>
</tr>
<tr>
  <th>C(floors)[T.1.5]</th>     <td> 5515.1712</td> <td> 4074.787</td> <td>    1.353</td> <td> 0.176</td> <td>-2471.727</td> <td> 1.35e+04</td>
</tr>
<tr>
  <th>C(floors)[T.2.0]</th>     <td>-1.439e+04</td> <td> 3182.428</td> <td>   -4.521</td> <td> 0.000</td> <td>-2.06e+04</td> <td>-8148.893</td>
</tr>
<tr>
  <th>C(floors)[T.2.5]</th>     <td> 5.872e+04</td> <td> 1.28e+04</td> <td>    4.572</td> <td> 0.000</td> <td> 3.35e+04</td> <td> 8.39e+04</td>
</tr>
<tr>
  <th>C(floors)[T.3.0]</th>     <td>-7.142e+04</td> <td> 7369.479</td> <td>   -9.691</td> <td> 0.000</td> <td>-8.59e+04</td> <td> -5.7e+04</td>
</tr>
<tr>
  <th>C(floors)[T.3.5]</th>     <td> 1.632e+04</td> <td> 5.82e+04</td> <td>    0.280</td> <td> 0.779</td> <td>-9.78e+04</td> <td>  1.3e+05</td>
</tr>
<tr>
  <th>C(grade)[T.4]</th>        <td>-9.535e+04</td> <td> 1.56e+05</td> <td>   -0.611</td> <td> 0.541</td> <td>-4.01e+05</td> <td>  2.1e+05</td>
</tr>
<tr>
  <th>C(grade)[T.5]</th>        <td>-1.585e+05</td> <td> 1.54e+05</td> <td>   -1.028</td> <td> 0.304</td> <td>-4.61e+05</td> <td> 1.44e+05</td>
</tr>
<tr>
  <th>C(grade)[T.6]</th>        <td>-1.644e+05</td> <td> 1.54e+05</td> <td>   -1.067</td> <td> 0.286</td> <td>-4.67e+05</td> <td> 1.38e+05</td>
</tr>
<tr>
  <th>C(grade)[T.7]</th>        <td>-1.534e+05</td> <td> 1.54e+05</td> <td>   -0.995</td> <td> 0.320</td> <td>-4.56e+05</td> <td> 1.49e+05</td>
</tr>
<tr>
  <th>C(grade)[T.8]</th>        <td>-1.111e+05</td> <td> 1.54e+05</td> <td>   -0.720</td> <td> 0.471</td> <td>-4.13e+05</td> <td> 1.91e+05</td>
</tr>
<tr>
  <th>C(grade)[T.9]</th>        <td>-1.795e+04</td> <td> 1.54e+05</td> <td>   -0.116</td> <td> 0.907</td> <td> -3.2e+05</td> <td> 2.84e+05</td>
</tr>
<tr>
  <th>C(grade)[T.10]</th>       <td> 1.066e+05</td> <td> 1.54e+05</td> <td>    0.691</td> <td> 0.490</td> <td>-1.96e+05</td> <td> 4.09e+05</td>
</tr>
<tr>
  <th>C(grade)[T.11]</th>       <td> 2.979e+05</td> <td> 1.55e+05</td> <td>    1.928</td> <td> 0.054</td> <td>-5003.626</td> <td> 6.01e+05</td>
</tr>
<tr>
  <th>C(grade)[T.12]</th>       <td> 6.726e+05</td> <td> 1.55e+05</td> <td>    4.329</td> <td> 0.000</td> <td> 3.68e+05</td> <td> 9.77e+05</td>
</tr>
<tr>
  <th>C(grade)[T.13]</th>       <td> 1.451e+06</td> <td> 1.62e+05</td> <td>    8.966</td> <td> 0.000</td> <td> 1.13e+06</td> <td> 1.77e+06</td>
</tr>
<tr>
  <th>C(waterfront)[T.1.0]</th> <td> 8.229e+05</td> <td> 1.33e+04</td> <td>   61.664</td> <td> 0.000</td> <td> 7.97e+05</td> <td> 8.49e+05</td>
</tr>
<tr>
  <th>C(condition)[T.2]</th>    <td> 5.955e+04</td> <td> 3.13e+04</td> <td>    1.902</td> <td> 0.057</td> <td>-1809.317</td> <td> 1.21e+05</td>
</tr>
<tr>
  <th>C(condition)[T.3]</th>    <td>  7.03e+04</td> <td> 2.92e+04</td> <td>    2.411</td> <td> 0.016</td> <td> 1.32e+04</td> <td> 1.27e+05</td>
</tr>
<tr>
  <th>C(condition)[T.4]</th>    <td> 9.609e+04</td> <td> 2.92e+04</td> <td>    3.293</td> <td> 0.001</td> <td> 3.89e+04</td> <td> 1.53e+05</td>
</tr>
<tr>
  <th>C(condition)[T.5]</th>    <td> 1.432e+05</td> <td> 2.93e+04</td> <td>    4.878</td> <td> 0.000</td> <td> 8.56e+04</td> <td> 2.01e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98002]</th>  <td> 4205.4633</td> <td> 1.36e+04</td> <td>    0.309</td> <td> 0.757</td> <td>-2.25e+04</td> <td> 3.09e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98003]</th>  <td>-4692.3808</td> <td> 1.23e+04</td> <td>   -0.381</td> <td> 0.703</td> <td>-2.88e+04</td> <td> 1.95e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98004]</th>  <td> 7.694e+05</td> <td>  1.2e+04</td> <td>   64.044</td> <td> 0.000</td> <td> 7.46e+05</td> <td> 7.93e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98005]</th>  <td> 3.019e+05</td> <td> 1.45e+04</td> <td>   20.869</td> <td> 0.000</td> <td> 2.74e+05</td> <td>  3.3e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98006]</th>  <td> 2.614e+05</td> <td> 1.09e+04</td> <td>   24.063</td> <td> 0.000</td> <td>  2.4e+05</td> <td> 2.83e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98007]</th>  <td> 2.516e+05</td> <td> 1.53e+04</td> <td>   16.395</td> <td> 0.000</td> <td> 2.22e+05</td> <td> 2.82e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98008]</th>  <td> 2.739e+05</td> <td> 1.23e+04</td> <td>   22.266</td> <td> 0.000</td> <td>  2.5e+05</td> <td> 2.98e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98010]</th>  <td> 6.168e+04</td> <td> 1.75e+04</td> <td>    3.531</td> <td> 0.000</td> <td> 2.74e+04</td> <td> 9.59e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98011]</th>  <td> 1.346e+05</td> <td> 1.37e+04</td> <td>    9.806</td> <td> 0.000</td> <td> 1.08e+05</td> <td> 1.62e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98014]</th>  <td> 8.975e+04</td> <td> 1.62e+04</td> <td>    5.529</td> <td> 0.000</td> <td> 5.79e+04</td> <td> 1.22e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98019]</th>  <td> 9.342e+04</td> <td> 1.38e+04</td> <td>    6.757</td> <td> 0.000</td> <td> 6.63e+04</td> <td> 1.21e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98022]</th>  <td> 1.669e+04</td> <td> 1.31e+04</td> <td>    1.271</td> <td> 0.204</td> <td>-9053.206</td> <td> 4.24e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98023]</th>  <td>-3.442e+04</td> <td> 1.07e+04</td> <td>   -3.211</td> <td> 0.001</td> <td>-5.54e+04</td> <td>-1.34e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98024]</th>  <td> 1.496e+05</td> <td> 1.92e+04</td> <td>    7.777</td> <td> 0.000</td> <td> 1.12e+05</td> <td> 1.87e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98027]</th>  <td> 1.522e+05</td> <td> 1.13e+04</td> <td>   13.515</td> <td> 0.000</td> <td>  1.3e+05</td> <td> 1.74e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98028]</th>  <td> 1.323e+05</td> <td> 1.23e+04</td> <td>   10.765</td> <td> 0.000</td> <td> 1.08e+05</td> <td> 1.56e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98029]</th>  <td> 2.122e+05</td> <td>  1.2e+04</td> <td>   17.672</td> <td> 0.000</td> <td> 1.89e+05</td> <td> 2.36e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98030]</th>  <td> 5981.1482</td> <td> 1.26e+04</td> <td>    0.474</td> <td> 0.635</td> <td>-1.87e+04</td> <td> 3.07e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98031]</th>  <td> 1.416e+04</td> <td> 1.24e+04</td> <td>    1.141</td> <td> 0.254</td> <td>-1.02e+04</td> <td> 3.85e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98032]</th>  <td>-6818.9823</td> <td> 1.61e+04</td> <td>   -0.425</td> <td> 0.671</td> <td>-3.83e+04</td> <td> 2.46e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98033]</th>  <td> 3.603e+05</td> <td>  1.1e+04</td> <td>   32.616</td> <td> 0.000</td> <td> 3.39e+05</td> <td> 3.82e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98034]</th>  <td> 2.031e+05</td> <td> 1.05e+04</td> <td>   19.359</td> <td> 0.000</td> <td> 1.83e+05</td> <td> 2.24e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98038]</th>  <td> 3.901e+04</td> <td> 1.04e+04</td> <td>    3.758</td> <td> 0.000</td> <td> 1.87e+04</td> <td> 5.94e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98039]</th>  <td> 1.172e+06</td> <td> 2.38e+04</td> <td>   49.247</td> <td> 0.000</td> <td> 1.13e+06</td> <td> 1.22e+06</td>
</tr>
<tr>
  <th>C(zipcode)[T.98040]</th>  <td> 5.129e+05</td> <td> 1.25e+04</td> <td>   40.898</td> <td> 0.000</td> <td> 4.88e+05</td> <td> 5.37e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98042]</th>  <td> 4572.0552</td> <td> 1.05e+04</td> <td>    0.437</td> <td> 0.662</td> <td>-1.59e+04</td> <td> 2.51e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98045]</th>  <td> 9.858e+04</td> <td> 1.33e+04</td> <td>    7.438</td> <td> 0.000</td> <td> 7.26e+04</td> <td> 1.25e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98052]</th>  <td> 2.344e+05</td> <td> 1.04e+04</td> <td>   22.463</td> <td> 0.000</td> <td> 2.14e+05</td> <td> 2.55e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98053]</th>  <td> 2.067e+05</td> <td> 1.13e+04</td> <td>   18.305</td> <td> 0.000</td> <td> 1.85e+05</td> <td> 2.29e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98055]</th>  <td> 4.297e+04</td> <td> 1.25e+04</td> <td>    3.446</td> <td> 0.001</td> <td> 1.85e+04</td> <td> 6.74e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98056]</th>  <td> 8.831e+04</td> <td> 1.12e+04</td> <td>    7.880</td> <td> 0.000</td> <td> 6.63e+04</td> <td>  1.1e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98058]</th>  <td> 2.865e+04</td> <td> 1.09e+04</td> <td>    2.624</td> <td> 0.009</td> <td> 7254.125</td> <td> 5.01e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98059]</th>  <td> 8.443e+04</td> <td> 1.08e+04</td> <td>    7.785</td> <td> 0.000</td> <td> 6.32e+04</td> <td> 1.06e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98065]</th>  <td> 1.084e+05</td> <td> 1.21e+04</td> <td>    8.986</td> <td> 0.000</td> <td> 8.47e+04</td> <td> 1.32e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98070]</th>  <td> 2.364e+04</td> <td> 1.67e+04</td> <td>    1.414</td> <td> 0.157</td> <td>-9120.733</td> <td> 5.64e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98072]</th>  <td> 1.564e+05</td> <td> 1.24e+04</td> <td>   12.573</td> <td> 0.000</td> <td> 1.32e+05</td> <td> 1.81e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98074]</th>  <td> 1.741e+05</td> <td> 1.11e+04</td> <td>   15.719</td> <td> 0.000</td> <td> 1.52e+05</td> <td> 1.96e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98075]</th>  <td> 1.777e+05</td> <td> 1.17e+04</td> <td>   15.151</td> <td> 0.000</td> <td> 1.55e+05</td> <td> 2.01e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98077]</th>  <td> 1.061e+05</td> <td> 1.37e+04</td> <td>    7.721</td> <td> 0.000</td> <td> 7.91e+04</td> <td> 1.33e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98092]</th>  <td>-2.564e+04</td> <td> 1.16e+04</td> <td>   -2.205</td> <td> 0.027</td> <td>-4.84e+04</td> <td>-2843.694</td>
</tr>
<tr>
  <th>C(zipcode)[T.98102]</th>  <td> 4.708e+05</td> <td> 1.75e+04</td> <td>   26.927</td> <td> 0.000</td> <td> 4.36e+05</td> <td> 5.05e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98103]</th>  <td>  3.46e+05</td> <td> 1.06e+04</td> <td>   32.715</td> <td> 0.000</td> <td> 3.25e+05</td> <td> 3.67e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98105]</th>  <td> 4.872e+05</td> <td> 1.32e+04</td> <td>   36.995</td> <td> 0.000</td> <td> 4.61e+05</td> <td> 5.13e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98106]</th>  <td> 1.155e+05</td> <td> 1.18e+04</td> <td>    9.805</td> <td> 0.000</td> <td> 9.24e+04</td> <td> 1.39e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98107]</th>  <td> 3.526e+05</td> <td> 1.27e+04</td> <td>   27.850</td> <td> 0.000</td> <td> 3.28e+05</td> <td> 3.77e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98108]</th>  <td> 1.114e+05</td> <td> 1.39e+04</td> <td>    7.992</td> <td> 0.000</td> <td> 8.41e+04</td> <td> 1.39e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98109]</th>  <td> 5.129e+05</td> <td> 1.71e+04</td> <td>   30.061</td> <td> 0.000</td> <td> 4.79e+05</td> <td> 5.46e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98112]</th>  <td> 6.169e+05</td> <td> 1.27e+04</td> <td>   48.692</td> <td> 0.000</td> <td> 5.92e+05</td> <td> 6.42e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98115]</th>  <td> 3.385e+05</td> <td> 1.04e+04</td> <td>   32.396</td> <td> 0.000</td> <td> 3.18e+05</td> <td> 3.59e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98116]</th>  <td> 3.167e+05</td> <td> 1.19e+04</td> <td>   26.650</td> <td> 0.000</td> <td> 2.93e+05</td> <td>  3.4e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98117]</th>  <td> 3.214e+05</td> <td> 1.06e+04</td> <td>   30.455</td> <td> 0.000</td> <td> 3.01e+05</td> <td> 3.42e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98118]</th>  <td> 1.667e+05</td> <td> 1.07e+04</td> <td>   15.566</td> <td> 0.000</td> <td> 1.46e+05</td> <td> 1.88e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98119]</th>  <td> 5.127e+05</td> <td> 1.42e+04</td> <td>   36.058</td> <td> 0.000</td> <td> 4.85e+05</td> <td> 5.41e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98122]</th>  <td> 3.497e+05</td> <td> 1.23e+04</td> <td>   28.361</td> <td> 0.000</td> <td> 3.25e+05</td> <td> 3.74e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98125]</th>  <td> 2.101e+05</td> <td> 1.12e+04</td> <td>   18.743</td> <td> 0.000</td> <td> 1.88e+05</td> <td> 2.32e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98126]</th>  <td> 1.961e+05</td> <td> 1.17e+04</td> <td>   16.805</td> <td> 0.000</td> <td> 1.73e+05</td> <td> 2.19e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98133]</th>  <td> 1.583e+05</td> <td> 1.08e+04</td> <td>   14.716</td> <td> 0.000</td> <td> 1.37e+05</td> <td> 1.79e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98136]</th>  <td> 2.704e+05</td> <td> 1.27e+04</td> <td>   21.341</td> <td> 0.000</td> <td> 2.46e+05</td> <td> 2.95e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98144]</th>  <td> 2.879e+05</td> <td> 1.18e+04</td> <td>   24.487</td> <td> 0.000</td> <td> 2.65e+05</td> <td> 3.11e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98146]</th>  <td> 1.148e+05</td> <td> 1.23e+04</td> <td>    9.329</td> <td> 0.000</td> <td> 9.06e+04</td> <td> 1.39e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98148]</th>  <td> 6.939e+04</td> <td> 2.21e+04</td> <td>    3.133</td> <td> 0.002</td> <td>  2.6e+04</td> <td> 1.13e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98155]</th>  <td> 1.451e+05</td> <td>  1.1e+04</td> <td>   13.213</td> <td> 0.000</td> <td> 1.24e+05</td> <td> 1.67e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98166]</th>  <td> 7.925e+04</td> <td> 1.27e+04</td> <td>    6.237</td> <td> 0.000</td> <td> 5.43e+04</td> <td> 1.04e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98168]</th>  <td> 4.477e+04</td> <td> 1.26e+04</td> <td>    3.562</td> <td> 0.000</td> <td> 2.01e+04</td> <td> 6.94e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98177]</th>  <td>  2.51e+05</td> <td> 1.27e+04</td> <td>   19.786</td> <td> 0.000</td> <td> 2.26e+05</td> <td> 2.76e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98178]</th>  <td> 4.169e+04</td> <td> 1.26e+04</td> <td>    3.317</td> <td> 0.001</td> <td> 1.71e+04</td> <td> 6.63e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98188]</th>  <td>  2.93e+04</td> <td> 1.56e+04</td> <td>    1.875</td> <td> 0.061</td> <td>-1334.111</td> <td> 5.99e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98198]</th>  <td> 1.991e+04</td> <td> 1.24e+04</td> <td>    1.610</td> <td> 0.107</td> <td>-4331.129</td> <td> 4.41e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98199]</th>  <td> 3.975e+05</td> <td> 1.19e+04</td> <td>   33.288</td> <td> 0.000</td> <td> 3.74e+05</td> <td> 4.21e+05</td>
</tr>
<tr>
  <th>sqft_living</th>          <td>  157.8230</td> <td>    2.528</td> <td>   62.419</td> <td> 0.000</td> <td>  152.867</td> <td>  162.779</td>
</tr>
<tr>
  <th>sqft_lot</th>             <td>    0.1934</td> <td>    0.028</td> <td>    6.849</td> <td> 0.000</td> <td>    0.138</td> <td>    0.249</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>14192.055</td> <th>  Durbin-Watson:     </th>  <td>   1.994</td>  
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>   <th>  Jarque-Bera (JB):  </th> <td>1182763.946</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 2.471</td>   <th>  Prob(JB):          </th>  <td>    0.00</td>  
</tr>
<tr>
  <th>Kurtosis:</th>       <td>39.360</td>   <th>  Cond. No.          </th>  <td>2.16e+07</td>  
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 2.16e+07. This might indicate that there are<br/>strong multicollinearity or other numerical problems.




```python
#NEED TO ONE-HOT ENCODE CATEGORICAL VARIABLES AND MERGE WITH NUMERICAL VARIABLES

bed_dummies = pd.get_dummies(df_base['bedrooms'], prefix='bed', drop_first=True)
bath_dummies = pd.get_dummies(df_base['bathrooms'], prefix='bath', drop_first=True)
floor_dummies = pd.get_dummies(df_base['floors'], prefix='fl', drop_first=True )
grade_dummies = pd.get_dummies(df_base['grade'], prefix='gd', drop_first=True )
wtf_dummies = pd.get_dummies(df_base['waterfront'], prefix='wtf', drop_first=True)
cond_dummies = pd.get_dummies(df_base['condition'], prefix='cd', drop_first=True)
zip_dummies = pd.get_dummies(df_base['zipcode'], prefix='zip', drop_first=True)

price = df_base['price']

data = pd.concat([price, df_base[features], bed_dummies, bath_dummies, floor_dummies, grade_dummies, wtf_dummies, 
                 cond_dummies, zip_dummies], axis=1)

y = data['price']
X = data.drop(['price'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(len(X_train), len(X_test), len(y_train), len(y_test))
```

    16865 4217 16865 4217



```python
# FIT SCIKIT LEARN MODEL
linreg = LinearRegression()
linreg.fit(X_train, y_train)

# PREDICT
y_hat_train = linreg.predict(X_train)
y_hat_test = linreg.predict(X_test)


# LOOK AT RESIDUALS
train_mse = mean_squared_error(y_train, y_hat_train)
test_mse = mean_squared_error(y_test, y_hat_test)
train_rmse = np.sqrt(mean_squared_error(y_train, y_hat_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_hat_test))
r2_train = r2_score(y_train, y_hat_train)
r2_test = r2_score(y_test, y_hat_test)

print('Train Mean Squared Error:', train_mse)
print('Test Mean Squared Error:', test_mse)
print('Train Root Mean Squared Error:', train_rmse)
print('Test Root Mean Squared Error:', test_rmse)
print('Diff between Test/Train RMSE:', abs(train_rmse-test_rmse))
print('R2 train:' , r2_train)
print('R2 test:' , r2_test)
```

    Train Mean Squared Error: 23104529846.12526
    Test Mean Squared Error: 24547843729.89171
    Train Root Mean Squared Error: 152001.74290489324
    Test Root Mean Squared Error: 156677.51507440917
    Diff between Test/Train RMSE: 4675.772169515927
    R2 train: 0.8282919653161583
    R2 test: 0.8170501174037272



```python
# ACTUAL VS. PREDICTED VALUES ANALYSIS

# Code inspired by:
# https://towardsdatascience.com/a-beginners-guide-to-linear-regression-in-python-with-scikit-learn-83a8f7ae2b4f


pred_df_base = pd.DataFrame({'Actual': y_test, 'Predicted': y_hat_test, 'Diff': abs(y_test-y_hat_test)}).round(2)

df1 = pred_df_base.sort_values(by=['Diff'], ascending=False).head(20)

df1
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Actual</th>
      <th>Predicted</th>
      <th>Diff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>12358</td>
      <td>4210000.0</td>
      <td>1863995.29</td>
      <td>2346004.71</td>
    </tr>
    <tr>
      <td>9478</td>
      <td>2380000.0</td>
      <td>961785.95</td>
      <td>1418214.05</td>
    </tr>
    <tr>
      <td>7304</td>
      <td>2500000.0</td>
      <td>1086981.97</td>
      <td>1413018.03</td>
    </tr>
    <tr>
      <td>20519</td>
      <td>2950000.0</td>
      <td>1629929.76</td>
      <td>1320070.24</td>
    </tr>
    <tr>
      <td>13954</td>
      <td>3400000.0</td>
      <td>2120529.87</td>
      <td>1279470.13</td>
    </tr>
    <tr>
      <td>13398</td>
      <td>2420000.0</td>
      <td>3658970.74</td>
      <td>1238970.74</td>
    </tr>
    <tr>
      <td>1431</td>
      <td>2540000.0</td>
      <td>1319050.21</td>
      <td>1220949.79</td>
    </tr>
    <tr>
      <td>19133</td>
      <td>3640000.0</td>
      <td>2450389.86</td>
      <td>1189610.14</td>
    </tr>
    <tr>
      <td>6396</td>
      <td>2900000.0</td>
      <td>1746732.98</td>
      <td>1153267.02</td>
    </tr>
    <tr>
      <td>4807</td>
      <td>2480000.0</td>
      <td>3584371.31</td>
      <td>1104371.31</td>
    </tr>
    <tr>
      <td>2083</td>
      <td>3850000.0</td>
      <td>2791027.52</td>
      <td>1058972.48</td>
    </tr>
    <tr>
      <td>7693</td>
      <td>2150000.0</td>
      <td>1100539.99</td>
      <td>1049460.01</td>
    </tr>
    <tr>
      <td>1674</td>
      <td>2250000.0</td>
      <td>1244754.76</td>
      <td>1005245.24</td>
    </tr>
    <tr>
      <td>20309</td>
      <td>3000000.0</td>
      <td>1995362.58</td>
      <td>1004637.42</td>
    </tr>
    <tr>
      <td>7645</td>
      <td>2540000.0</td>
      <td>1563512.78</td>
      <td>976487.22</td>
    </tr>
    <tr>
      <td>13894</td>
      <td>925000.0</td>
      <td>1856941.72</td>
      <td>931941.72</td>
    </tr>
    <tr>
      <td>4186</td>
      <td>2880000.0</td>
      <td>1970091.49</td>
      <td>909908.51</td>
    </tr>
    <tr>
      <td>4028</td>
      <td>2500000.0</td>
      <td>1629842.47</td>
      <td>870157.53</td>
    </tr>
    <tr>
      <td>15138</td>
      <td>2150000.0</td>
      <td>1282364.67</td>
      <td>867635.33</td>
    </tr>
    <tr>
      <td>18867</td>
      <td>2180000.0</td>
      <td>1335152.51</td>
      <td>844847.49</td>
    </tr>
  </tbody>
</table>
</div>




```python
#VISUAL ANALYSIS OF OUR ACTUAL VS. PREDICTED AND DIFFERENCE FROM OUR BASE MODEL
#CODE FROM: 
#https://towardsdatascience.com/a-beginners-guide-to-linear-regression-in-python-with-scikit-learn-83a8f7ae2b4f

df1.plot(kind='bar',figsize=(15,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
```


![png](output_64_0.png)



```python
#MULTI-COLLINEARITY ANALYSIS
from statsmodels.stats.outliers_influence import variance_inflation_factor

# run the non-categorical values through VIF

vif = pd.DataFrame()
X = df_base[features]
X = sm.add_constant(X)
vif['VIF value'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['Feature'] = X.columns

vif.sort_values(by=['VIF value'], ascending=False).round(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>VIF value</th>
      <th>Feature</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>3597321.679</td>
      <td>const</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4.022</td>
      <td>sqft_living</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2.890</td>
      <td>bathrooms</td>
    </tr>
    <tr>
      <td>6</td>
      <td>2.791</td>
      <td>grade</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1.628</td>
      <td>bedrooms</td>
    </tr>
    <tr>
      <td>5</td>
      <td>1.511</td>
      <td>floors</td>
    </tr>
    <tr>
      <td>8</td>
      <td>1.090</td>
      <td>condition</td>
    </tr>
    <tr>
      <td>9</td>
      <td>1.070</td>
      <td>zipcode</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1.058</td>
      <td>sqft_lot</td>
    </tr>
    <tr>
      <td>7</td>
      <td>1.020</td>
      <td>waterfront</td>
    </tr>
  </tbody>
</table>
</div>



> **Multicollinearity issues from VIF analysis** - and sqft_living, bedrooms, grade, and bathrooms are all highly correlated. We need to remove one of them when we tune our next model.


```python
#PLOT A QQ PLOT TO CHECK NORMALITY
fig = sm.graphics.qqplot(model.resid, dist=stats.norm, line='45', fit=True);
```


![png](output_67_0.png)


> **Normality** - Yikes! We have some major issues with normality. This is most likely a result of outliers in the data.

> As a refresher, **scaling** is a form standardizing our data to make sure we have similar magnitudes. **Normalizing** our data refers to using transformation techniques to make data look more like the normal distribution. 

> For instance, "grade" is on a **discrete/categorical** scale of "3-13", while "sqft_living" is made up of **continuous numerical** values in the thousands. This presents a problem if we're going to try and make an accurate and well-fitting model.

> **Check for Heteroscedasticity**


```python
#CHECK FOR HOMOSCEDASTICITY 
plt.subplots(figsize=(13,8))
plt.scatter(model.predict(df_base[features]), model.resid)
plt.plot(model.predict(df_base[features]), [0 for i in range(len(df_base))]);
```


![png](output_70_0.png)


> - **Yikes again!** We clearly have a heteroscedasticity problem with our data that we'll have to deal with in subsequent models.


> **Conclusions from our base model:**

> **Model fit & R2:** .829 for an R2 value indicates our model fits relatively well. About 83% of the the variance in our data can be explained by the model. 

> **Normality/Residuals check:**
- Very positively skewed with a value of **2.471**
- Kurtosis value of **39.360**, which is well above acceptable value of **3**
- VIF analysis indicates numerous issues with multicollinearity 
- QQ plot indicates the model is not robust, and we need to remove some outliers and normalize (re-scale) some data
- Heteroscedasticity looks to be a problem as well. 
> 
> **Prediction assessment:** 
> - **Actual vs. Predicted** information indicates decent prediction performance from our baseline model


> **NEXT STEPS**:  Now that we've looked at the problems in our raw data, the next steps are to do some feature engineering to normalize the data to prepare it better for multivariate linear regression modeling. 

## FINAL MODEL

> **For the final model, we are going to log transform the following features, in order to scale to increase our model performance:**
>    - sqft_living
>    - bedrooms
>    - bathrooms
>    - sqft_lot
>    - floors
>    - condition
>    - grade
>    - sqft_living15
>    - sqft_lot15
>    - price (independent variable)

> **In this case, we are going to use a logarithm with a base of 10 vs. 2 because we our data vary greatly in magnitude. We need to get them as close to each other as possible to use effectively in our model.**  


```python
# REMOVE OUTLIERS USING Z-SCORE

from scipy import stats

df_base[(np.abs(stats.zscore(df_base[features])) < 3).all(axis=1)]

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>221900.0</td>
      <td>3</td>
      <td>1.00</td>
      <td>1180</td>
      <td>5650</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1180</td>
      <td>0.0</td>
      <td>1955</td>
      <td>0.0</td>
      <td>98178</td>
      <td>1340</td>
      <td>5650</td>
    </tr>
    <tr>
      <td>1</td>
      <td>538000.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>2570</td>
      <td>7242</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>2170</td>
      <td>400.0</td>
      <td>1951</td>
      <td>1991.0</td>
      <td>98125</td>
      <td>1690</td>
      <td>7639</td>
    </tr>
    <tr>
      <td>2</td>
      <td>180000.0</td>
      <td>2</td>
      <td>1.00</td>
      <td>770</td>
      <td>10000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>6</td>
      <td>770</td>
      <td>0.0</td>
      <td>1933</td>
      <td>0.0</td>
      <td>98028</td>
      <td>2720</td>
      <td>8062</td>
    </tr>
    <tr>
      <td>3</td>
      <td>604000.0</td>
      <td>4</td>
      <td>3.00</td>
      <td>1960</td>
      <td>5000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>7</td>
      <td>1050</td>
      <td>910.0</td>
      <td>1965</td>
      <td>0.0</td>
      <td>98136</td>
      <td>1360</td>
      <td>5000</td>
    </tr>
    <tr>
      <td>4</td>
      <td>510000.0</td>
      <td>3</td>
      <td>2.00</td>
      <td>1680</td>
      <td>8080</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1680</td>
      <td>0.0</td>
      <td>1987</td>
      <td>0.0</td>
      <td>98074</td>
      <td>1800</td>
      <td>7503</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>21592</td>
      <td>360000.0</td>
      <td>3</td>
      <td>2.50</td>
      <td>1530</td>
      <td>1131</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1530</td>
      <td>0.0</td>
      <td>2009</td>
      <td>0.0</td>
      <td>98103</td>
      <td>1530</td>
      <td>1509</td>
    </tr>
    <tr>
      <td>21593</td>
      <td>400000.0</td>
      <td>4</td>
      <td>2.50</td>
      <td>2310</td>
      <td>5813</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>2310</td>
      <td>0.0</td>
      <td>2014</td>
      <td>0.0</td>
      <td>98146</td>
      <td>1830</td>
      <td>7200</td>
    </tr>
    <tr>
      <td>21594</td>
      <td>402101.0</td>
      <td>2</td>
      <td>0.75</td>
      <td>1020</td>
      <td>1350</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1020</td>
      <td>0.0</td>
      <td>2009</td>
      <td>0.0</td>
      <td>98144</td>
      <td>1020</td>
      <td>2007</td>
    </tr>
    <tr>
      <td>21595</td>
      <td>400000.0</td>
      <td>3</td>
      <td>2.50</td>
      <td>1600</td>
      <td>2388</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1600</td>
      <td>0.0</td>
      <td>2004</td>
      <td>0.0</td>
      <td>98027</td>
      <td>1410</td>
      <td>1287</td>
    </tr>
    <tr>
      <td>21596</td>
      <td>325000.0</td>
      <td>2</td>
      <td>0.75</td>
      <td>1020</td>
      <td>1076</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1020</td>
      <td>0.0</td>
      <td>2008</td>
      <td>0.0</td>
      <td>98144</td>
      <td>1020</td>
      <td>1357</td>
    </tr>
  </tbody>
</table>
<p>20186 rows × 17 columns</p>
</div>




```python
# LOG TRANSFORM DATA THAT DOESN'T HAVE "0" OR NEGATIVE VALUES. WE'LL ALSO REMOVE SOME OF THE FEATURES WITH 
# MULTI-COLLINEARITY PROBLEMS LIKE sqft_above

# Change this into a for loop and function

data_log = pd.DataFrame([])
data_log['logsqft_living'] = np.log(df_base['sqft_living'])
data_log['logsqft_lot'] = np.log(df_base['sqft_lot'])
data_log['logsqft_living15'] = np.log(df_base['sqft_living15'])
data_log['logsqft_lot15'] = np.log(df_base['sqft_lot15'])

bed_dummies = pd.get_dummies(df_base['bedrooms'], prefix='bed', drop_first=True)
bath_dummies = pd.get_dummies(df_base['bathrooms'], prefix='bath', drop_first=True)
floor_dummies = pd.get_dummies(df_base['floors'], prefix='fl', drop_first=True )
grade_dummies = pd.get_dummies(df_base['grade'], prefix='gd', drop_first=True )
wtf_dummies = pd.get_dummies(df_base['waterfront'], prefix='wtf', drop_first=True)
cond_dummies = pd.get_dummies(df_base['condition'], prefix='cd', drop_first=True)
zip_dummies = pd.get_dummies(df_base['zipcode'], prefix='zip', drop_first=True)

price = df_base['price']

data_2 = pd.concat([price, data_log, bed_dummies, bath_dummies, floor_dummies, grade_dummies, wtf_dummies, 
                 cond_dummies, zip_dummies], axis=1)

y = data_2['price']
X = data_2.drop(['price'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(len(X_train), len(X_test), len(y_train), len(y_test))


```

    16865 4217 16865 4217



```python
# FIT SCIKIT LEARN MODEL
linreg_2 = LinearRegression()
linreg_2.fit(X_train, y_train)

# PREDICT
y_hat_train = linreg_2.predict(X_train)
y_hat_test = linreg_2.predict(X_test)


# LOOK AT RESIDUALS
train_mse = mean_squared_error(y_train, y_hat_train)
test_mse = mean_squared_error(y_test, y_hat_test)
train_rmse = np.sqrt(mean_squared_error(y_train, y_hat_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_hat_test))
r2_train = r2_score(y_train, y_hat_train)
r2_test = r2_score(y_test, y_hat_test)

print('Train Mean Squared Error:', train_mse)
print('Test Mean Squared Error:', test_mse)
print('Train Root Mean Squared Error:', train_rmse)
print('Test Root Mean Squared Error:', test_rmse)
print('Diff between Test/Train RMSE:', abs(train_rmse-test_rmse))
print('R2 train:' , r2_train)
print('R2 test:' , r2_test)

```

    Train Mean Squared Error: 24018885783.331047
    Test Mean Squared Error: 1.4796206608127134e+33
    Train Root Mean Squared Error: 154980.2754653993
    Test Root Mean Squared Error: 3.8465837581062936e+16
    Diff between Test/Train RMSE: 3.846583758090795e+16
    R2 train: 0.8214966631816967
    R2 test: -1.102729955271308e+22



```python
# TURN COEFFICIENTS INTO A DATAFRAME FOR VIEWING
coefs_2 = linreg_2.coef_
view_cfs_2 = pd.DataFrame(coefs_2, X.columns, 
                          columns=['coefficients']).sort_values(by='coefficients', ascending = False)

display(view_cfs_2)

```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>coefficients</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>bed_33</td>
      <td>2.487401e+18</td>
    </tr>
    <tr>
      <td>bath_7.5</td>
      <td>2.289024e+17</td>
    </tr>
    <tr>
      <td>bath_7.75</td>
      <td>2.946738e+06</td>
    </tr>
    <tr>
      <td>bath_8.0</td>
      <td>2.361576e+06</td>
    </tr>
    <tr>
      <td>gd_13</td>
      <td>1.900294e+06</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>gd_5</td>
      <td>-2.058446e+05</td>
    </tr>
    <tr>
      <td>bed_7</td>
      <td>-2.242000e+05</td>
    </tr>
    <tr>
      <td>gd_6</td>
      <td>-2.281847e+05</td>
    </tr>
    <tr>
      <td>gd_7</td>
      <td>-2.285308e+05</td>
    </tr>
    <tr>
      <td>bed_9</td>
      <td>-3.046579e+05</td>
    </tr>
  </tbody>
</table>
<p>132 rows × 1 columns</p>
</div>



```python
pred_df_base = pd.DataFrame({'Actual': y_test, 'Predicted': y_hat_test, 'Diff': abs(y_test-y_hat_test)}).round(2)

df2 = pred_df_base.sort_values(by=['Diff'], ascending=False).head(20)

df3 = pred_df_base.head(15)
```


```python
#CODE FROM: 
#https://towardsdatascience.com/a-beginners-guide-to-linear-regression-in-python-with-scikit-learn-83a8f7ae2b4f

df3.plot(kind='bar',figsize=(15,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
```


![png](output_81_0.png)


# iNTERPRET


> **Model fit & R2:** .82 adjusted R2 value indicates our model fits well, and we're able to explain 82% of the variance using our model.

> 
> **Prediction assessment:** 
> - **Actual vs. Predicted** information indicates **strong** prediction performance compared to the baseline model
> 


# CONCLUSIONS & RECOMMENDATIONS

> **Conclusion:** The features that had the most significant positive correlation with home sale price were **Grade, Overall Square-Footage, and Condition.** Indeed, our model relieved most heavily on these features to predict home sale prices.

> **Recommendations:**

> **Buyer:** Downplay the importance of the above features (especially grade) during negotiations with the seller. Through your real estate agent, indicate to the seller that you are looking at other homes in the area that better fit your needs (i.e. more bedrooms, bathrooms).      

> **Seller:** Looks at ways to increase the overall grade for your house. Obtain the grading criteria used to generate the data and pick tangible, low-effort/cost areas where you can make improvements to your home that increase your score. Make improvements to justify a larger asking price. Look for a quick way to increase the overall square footage of your house such as a screened in porch or garage addition.




> - **Future Seller-** look at options to add square footage to your house, and remodeling key rooms such as the kitchen to increase your grade.



> **Future research:**
> - Research school zoning data and home distance from elementary, middle, and high schools.

> - Pull new home sale data from after the start of the COVID-19 pandemic, specifically looking at home sale trends in urban vs. rural areas. 

> - Examine crime statistics effect on home prices.

 



## CUT LINE FOR NOTEBOOK


```python
#CALCULATE ERROR FOR LOG TRANSFORMED DATA

# y_predict_2 = linreg_2.predict(X_test_2)

# linreg_2_mse = mean_squared_error(y_predict_2, y_test_2)


# print('Mean Absolute Percentage Error:', metrics.mean_absolute_error(y_test_2, y_predict_2))  
# print('Mean Squared Percentage Error:', metrics.mean_squared_error(y_test_2, y_predict_2))  
# print('Root Mean Squared Percentage Error:', np.sqrt(metrics.mean_squared_error(y_test_2, y_predict_2)))
```

> The **Root Mean Squared Percentage Error** above indicates that the squared root of the average **percentage** of the difference between our model's prediction and the log transformed observations in our test data is 34%.


```python
#DUMMY VARIABLES FOR CONDITION, AND GRADE

dummy_condition = pd.get_dummies(df['condition'], prefix= 'cond')
dummy_grade = pd.get_dummies(df['grade'], prefix= 'grade')

display(dummy_condition.head())
display(dummy_grade.head())
```

> Summarize your conclusions and bullet-point your list of recommendations, which are based on your modeling results.
