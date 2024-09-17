
# Title for the Project


## Description


## Data Cleaning

###Function:

```python

def cleaning_null (df):

```


There were 3 columns with missing values:

 - 'fuel_type'
 - 'accident'
 - 'clean_title'

For each of them, the process to fill the missing values was different:

### 'fuel_type'

Using the data from the 'engine', we attributed to all of the cars with engine containing 'electric' or 'battery' the value of electric.
Same thought process for cars of with 'brand' equal to 'Tesla', all got the value of electric.
Then for the rest, we checked if there is 'gasoline' or 'diesel' and filled with the corresponding values.
Lastly we search for information on the few missing engines on google to know the fuel they consume.

### 'accident'

For the Nan values in 'accident' we replaced them with 'None reported', as there was no correlation between any other data and we assume that if there is missing the value, probably there is no none reported.

### 'clean_title'

As there are only 'Yes' values we considered the Nan values as 'No'


## Data Set

[KAGGLE - Regression of Used Car Prices ](https://www.kaggle.com/competitions/playground-series-s4e9/leaderboard)
