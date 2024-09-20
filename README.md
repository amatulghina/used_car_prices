

# Determining Key Factors in Used Car Pricing: An In-Depth Analysis


## Description

Using a dataset of used cars that includes a wide range of characteristics and prices provides an excellent opportunity to train on predictive models. The dataset encompasses various attributes, such as technical specifications, engine power, accident history, and aesthetic features.

Why is this important?
On the surface, it is difficult to intuitively determine which characteristics most significantly affect a car's price. Additionally, consumers who lack extensive knowledge of automobiles may find it challenging to assess whether a car's price is fair.

How does this model help?
This predictive model is designed to help identify and understand the factors that matter most when pricing a used car. By analyzing the data, it becomes clear which attributes play a critical role in determining the car's value and which ones are less significant. This insight is invaluable for both sellers wanting to price their vehicles appropriately and buyers looking to make informed purchasing decisions.

## Data Cleaning


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




## Data filtering 

Some data has values that are alike so we grouped them and for the visualization we need to create dummies or classes. 

### 'fuel_type'
Some values were the same but written "differently" so we changed those names to the correct one :

```python
{'Plug-In Hybrid': 'Hybrid',
 'E85 Flex Fuel': 'Gasoline'})
```

### 'engine'
Separated the engine by its components, Horsepower, liters and cylinders:
```python
def extract_engine_info(engine, fuel_type):
        hp_search = re.search(r'(\d+(\.\d+)?)HP', engine)
        litre_search = re.search(r'(\d+(\.\d+)?)L', engine)
        cylinders_search = re.search(r'(\d+) Cylinder', engine)
        
        hp = float(hp_search.group(1)) if hp_search else ''
        litres = float(litre_search.group(1)) if litre_search else ''
        cylinders = int(cylinders_search.group(1)) if cylinders_search else ''
        
        return pd.Series([hp, litres, cylinders])
```



### 'brand'
Generated a mean of the brand price to minimize the number of columns needed to create  dummies of brand:
```python
mean_price_by_brand = df.groupby('brand')[['price']].mean().reset_index().sort_values(by='price', ascending=False)
        mean_price_by_brand['brand_ratio'] = mean_price_by_brand['price']/mean_price_by_brand.iloc[-1,-1]
```

### 'ext_col' & 'int_col'
```python
Changed the values to numbers, so the Data Frame get a bit smaller and easier to compute:
df.loc[(df[col].str.contains('Black', case=False, na=False)), col] = '1'
    df.loc[(df[col].str.contains('Noir', case=False, na=False)), col] = '2'
    df.loc[(df[col].str.contains('Blue', case=False, na=False)), col] = '3'
    df.loc[(df[col].str.contains('Blu', case=False, na=False)), col] = '4'
    df.loc[(df[col].str.contains('Red', case=False, na=False)), col] = '5'
    df.loc[(df[col].str.contains('White', case=False, na=False)), col] = '6'
    df.loc[(df[col].str.contains('Green', case=False, na=False)), col] = '7'
    df.loc[(df[col].str.contains('Gray', case=False, na=False)), col] = '8'
    df.loc[(df[col].str.contains('Grey', case=False, na=False)), col] = '9'
    df.loc[(df[col].str.contains('Silver', case=False, na=False)), col] = '10'
    df.loc[(df[col].str.contains('Metallic', case=False, na=False)), col] = '11'
    df.loc[(df[col].str.contains('Yellow', case=False, na=False)), col] = '12'
    df.loc[(df[col].str.contains('Orange', case=False, na=False)), col] = '13'
    df.loc[(df[col].str.contains('Brown', case=False, na=False)), col] = '14'
    df.loc[(df[col].str.contains('Beige', case=False, na=False)), col] = '15'

    colors = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15']

    # Use .str.contains() to create a mask for rows containing any word from words_list
    # We use '|'.join to create a regular expression that matches any word in words_list
    pattern = '|'.join(colors)
    mask = df[col].str.contains(pattern, case=False, regex=True)
    # Change the value if the text does NOT contain any word from the list
    df.loc[~mask, col] = '16'
```    


### 'transmission' 
Grouped the data by the type of transmission:

```python
transmission_mapping = {
    'Automatic': ['A/T', 'Automatic', '6-Speed A/T', '8-Speed A/T', '10-Speed Automatic', 
                  'Transmission w/Dual Shift Mode', '6-Speed Automatic', '7-Speed A/T', 
                  '5-Speed Automatic', '8-Speed Automatic', '9-Speed A/T', '10-Speed A/T', 
                  '9-Speed Automatic', '5-Speed A/T', '1-Speed A/T', '4-Speed A/T',
                  '9-Speed Automatic with Auto-Shift', '8-Speed Automatic with Auto-Shift', 
                  '10-Speed Automatic with Overdrive', '1-Speed Automatic', '2-Speed Automatic', 
                  '7-Speed Automatic', '7-Speed Automatic with Auto-Shift', '6-Speed Automatic with Auto-Shift',
                  'Single-Speed Fixed Gear', '8-SPEED A/T', '7-Speed DCT Automatic', '2-Speed A/T', '4-Speed Automatic'],
    
    'Manual': ['M/T', 'Manual', '5-Speed M/T', '6-Speed M/T', '7-Speed M/T', 
               '6-Speed Manual', '7-Speed Manual', '8-Speed Manual', '6 Speed Mt', '7-Speed'],
    
    'CVT': ['Automatic CVT', 'CVT Transmission', 'Variable', 'CVT-F'],
    
    'Other': ['F', '2', '–', 'SCHEDULED FOR OR IN PRODUCTION', 'Transmission Overdrive Switch']
}

```

## Functions:

```python
def cleaning_null (df):
```
Function to clean all the null values of the Data Sets

```python
def format_columns (df):
```
Function to format all the data in the data set on smaller groups and in dummies columns.


## Ensemble Method 

After running all the possible models with the formatted data, we found that the most accurate model was the Random Florest  



## Data Sources

From Kaggle we found a Playground Prediction Competition. There is a data set with car prices, and we need to do a machine learning  model to predict the prices of the cares, we are given a train data set, a test data set and a target data set.

[DS:  Regression of Used Car Prices ](https://www.kaggle.com/competitions/playground-series-s4e9/leaderboard)


## Slides

[Presentation](https://docs.google.com/presentation/d/1i4jDMUIB0a-p9x-t8T-v6WQOGErrYs0Owi0uBb2RZGE/edit?usp=sharing)





