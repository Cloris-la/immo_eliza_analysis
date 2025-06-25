import warnings
warnings.filterwarnings('ignore')
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from correlationFunctions import corPlot, corMatrix, pairPlot

# Load the dataset
# This gets the directory where mainCor.py is located
dir_path = Path(__file__).parent
file_path = dir_path / 'Data' / 'Cleaned - immoweb-dataset.csv'
df = pd.read_csv(file_path)

numeric_vars = ['bedroomCount',
 'bathroomCount',
 'habitableSurface',
 'landSurface',
 'parkingCountIndoor',
 'parkingCountOutdoor',
 'toiletCount',
 'price']


#############################################################################################
# CORRELATION
#############################################################################################

#### PREPARING THE DATA

# List of relevant numeric-like columns (including object types)
features = [col for col in numeric_vars if col != 'price']

# Copy the data and convert object columns to numeric (coerce "no info" to NaN)
df_cleaned = df[features + ['price', 'type']].copy()
for col in features:
    df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')

# Create 2 subsets for House and Apartment
df_house = df_cleaned[df_cleaned['type'] == 'House'].copy()
df_apartment = df_cleaned[df_cleaned['type'] == 'Apartment'].copy()

# Drop the type column for each
df_cleaned = df_cleaned.drop(columns=['type'])
df_house = df_house.drop(columns=['type'])
df_apartment = df_apartment.drop(columns=['type', 'landSurface'])  # also landSurface as Apt dont have a land

# Create the output directory if it doesn't exist
output_dir = 'plots/correlation'
os.makedirs(output_dir, exist_ok=True)


#### CALLING THE PLOT FUNCTIONS

corPlot(df_cleaned, "All properties")
corPlot(df_house, "Only Houses")
corPlot(df_apartment, "Only Apartments")

corMatrix(df_cleaned, "All properties")
corMatrix(df_house, "Only Houses")
corMatrix(df_apartment, "Only Apartments")

# Pairplots for only the top 3 most correlated
top3_features = df_cleaned.corr()['price'].drop('price').sort_values(ascending=False).head(3).index.tolist()
pairPlot(df_cleaned, top3_features, "All properties")

top3_house = df_house.corr()['price'].drop('price').sort_values(ascending=False).head(3).index.tolist()
pairPlot(df_house, top3_house, "Only Houses")

top3_apartment = df_apartment.corr()['price'].drop('price').sort_values(ascending=False).head(3).index.tolist()
pairPlot(df_apartment, top3_apartment, "Only Apartments")


#### DRAWING CONCLUSIONS ABOUT THE CORRELATION

# Retrieve the max and min correlations against the price
maxPriceCor = max(
    df_cleaned.corr()['price'].drop('price').sort_values(ascending=False)
)

minPriceCor = min(
    df_cleaned.corr()['price'].drop('price').sort_values(ascending=False)
)

# Retrieve the top2 correlation against the price for house and apartment
PriceCorHouse = df_house.corr()['price'].drop('price').sort_values(ascending=False).head(2)
PriceCorApt = df_apartment.corr()['price'].drop('price').sort_values(ascending=False).head(2)

# Retrieve the max correlations excluding the price
maxCorHouse = df_house.corr().where(~np.eye(df_house.corr().shape[0], dtype=bool)).max().max()
maxCorApt = df_apartment.corr().where(~np.eye(df_apartment.corr().shape[0], dtype=bool)).max().max()

# Checking the correlation between price and parking combined
df_cleaned['parking'] = df_cleaned.parkingCountIndoor + df_cleaned.parkingCountOutdoor
parkingCor = df_cleaned.corr()['price']['parking']

print("""
    ==============================================================================================
                 CORRELATION  
    ==============================================================================================
    
    All plots related to correlation are saved in the folder plots/correlation
      """)

print(f"""
      Looking at correlation against price, we don't find very strong correlations.
      The highest one being with bedroomCount {maxPriceCor:.2f}.
 
      If we split the data between houses and apartments, we find stronger correlations:
      - For Houses, the correlation with bedroomCount ({PriceCorHouse[1]:.2f}) is lower than with toiletCount ({PriceCorHouse[0]:.2f}).
      - For Apartments, the correlation with bedroomCount ({PriceCorApt[1]:.2f}) is lower than with bathroomCount ({PriceCorApt[0]:.2f}).
      """)


print(f"""
      Looking at other variables correlation, we still have no strong correlations 
      but when we split the data:
      - For Houses, the correlation between bedroomCount and toiletCount is {maxCorHouse:.2f}.
      - For Apartments, the correlation between bedroomCount and bathroomCount is {maxCorApt:.2f}.
      """)

print(f"""
      Another insteresting aspect is the almost null correlation between price and parking variables ({minPriceCor:.2f}).
      If we combine indoor and outdoor parking, then the correlation with price increases to {parkingCor:.2f}
      Which still is very low.
      """)

print("""
      For our model, we may think about combining bathroom and toilet counts
      """)
