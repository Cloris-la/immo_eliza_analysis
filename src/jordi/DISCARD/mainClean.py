# IMMOWEB PROJECT: EXPLORATORY DATA ANALYSIS

## LIBRARIES AND SETTINGS
# This is for reading the locality name properly. In avoiding the encoding error
import csv
import os
from pathlib import Path

import numpy as np

import warnings
warnings.filterwarnings('ignore')

# Pandas options for data wrangling and output set-up 
import pandas as pd
pd.set_option('display.max_columns', None) # display all columns
pd.set_option('display.expand_frame_repr', False) # print all columns and in the same line
pd.set_option('display.max_colwidth', None) # display the full content of each cell
pd.set_option('display.float_format', lambda x: '%.2f' %x) # floats to be displayed with 2 decimal places

## FUNCTIONS USED RECURRENTLY
# Function to run basic data frame description
def Descriptives(df):
    print("Looking at the shape to see the number of records:", df.shape)
    print("\nDescription of the file to check values range:")
    print(df.describe().transpose())
    print("\nAttributes and respective data types:")
    print(df.info())

def MemOptimisation(df):
    print(f"\nAmount of memory used by all attributes: {df.memory_usage(deep=True).sum()}\n")
    
    # Optimise memory usage
    for i in categoric_cols:
        df[i] = df[i].astype('category')
    for i in numeric_cols:
        df[i] = pd.to_numeric(df[i], downcast='integer')
        df[i] = pd.to_numeric(df[i], downcast='float')    

    print(df.info(memory_usage='deep'))
    print("\nAmount of memory used now by all attributes: ",df.memory_usage(deep=True).sum())
    
# Function to check the missing values (NaNs)
def Missing(df):
    try:
        assert df.notnull().all().all()
        print("Good news! There are no missing values.")
    
    except AssertionError:
        print("Count of missing values:")
        print(df.isna().sum())
        print("\nPercentage of missing values:")
        print(df.isna().mean().round(4)*100, "\n")

def clear_terminal():
    # For Windows
    if os.name == 'nt':
        os.system('cls')
    # For Linux/macOS
    else:
        os.system('clear')

## START OF ANALYSIS: LOAD DATA & CLEAN THE VARIABLES

# Load the dataset
# This gets the directory where mainClean.py is located
dir_path = Path(__file__).parent
file_path = dir_path / 'Data' / 'immoweb-dataset.csv'
df = pd.read_csv(file_path)

# Ensure no leading/trailing spaces in the column names
df.columns = [col.strip() for col in df.columns]
# Show information about the data
Descriptives(df)

### TARGET VARIABLE: PRICE

# Check the missing values in price
Missing(df['price'])

# Price is the target variable, we cannot input a value for the missing values
# as we cannot check the reliability of the ML model againt an inputted value: model would be biased
df = df.dropna(subset=['price'])

Missing(df)

# We see there are variables with 100% missing values, so nothing to do with them: DROP
# Start in list with all variables to drop
dropVar = []
dropVar = df.columns[df.isna().sum() == len(df)].tolist()
dropVar

### UNNAMED
# This is an index variable, as presumabily the data comes from a dataframe that was saved as CSV keeping the index column
dropVar += ['Unnamed: 0']
dropVar

### ID & URL
# These variables have no value for predicting price: they can be dropped
# But, first checking if there are duplicated records
df.duplicated(subset=['id','url'], keep=False).sum()
dropVar += ['id', 'url']
dropVar

### TYPE & SUBTYPE
# Checking the different values in type
df.type.value_counts()
ad_types = df[['type', 'subtype']]
ad_types.groupby(['type', 'subtype']).value_counts()

# Looks correct
# Apply sentence case to remain consistent with the rest of string variables
df[['type', 'subtype']] = df[['type', 'subtype']].apply(lambda col: col.str.strip().str.capitalize())
Missing(df[['type', 'subtype']])

### GEO VARIABLES: PROVINCE, LOCALITY, POSTCODE & REGION
# Create a function to assign Regions to the locality
def map_region(row):
    loc = str(row["locality"]).strip()
    prov = row["province"]
    if loc in german_towns:
        return "German-speaking Community"
    if prov == "Brussels":
        return "Brussels"
    if prov in flemish:
        return "Flanders"
    if prov in walloon:
        return "Wallonia"
    return "Unknown"

# Add Region column
flemish = ["Antwerp", "Limburg", "East Flanders", "Flemish Brabant", "West Flanders"]
walloon = ["Hainaut", "Liège", "Luxembourg", "Namur", "Walloon Brabant"]
german_towns = [
        "Eupen", "Kelmis", "Raeren", "Lontzen", "Bütgenbach",
        "Büllingen", "Amel", "Burg-Reuland", "St. Vith"
    ]

df["region"] = df.apply(map_region, axis=1)
df.region.value_counts()
df.postCode.value_counts()

# Move "Region" column to left of "province"
region_series = df.pop("region")
province_idx = df.columns.get_loc("province")
df.insert(province_idx, "region", region_series)

# Strip whitespace in locality, region, province, postCode
df[["locality", "region", "province", "postCode"]] = df[["locality", "region", "province", "postCode"]].astype(str).apply(lambda x: x.str.strip())
Missing(df[["locality", "region", "province", "postCode"]])

# Then drop 'locality' which is postcode
dropVar += ['postcode']
dropVar

# Discard locality as it is the same as postCode
dropVar += ["locality"]
dropVar

### ROOMS COUNT

#### BEDROOMCOUNT & ROOMCOUNT
# These variables are very similar (advertisers often do not distinguish between bedroom and room)
# and roomCount present high number of missing values
Missing(df[['bedroomCount', 'roomCount']])

# for records with missing bedroomCount, take roomCount if exists
df['bedroomCount'] = df['bedroomCount'].fillna(df['roomCount'])

# Nan remaining filled with No info
df['bedroomCount'] = df['bedroomCount'].fillna('No info')

# Then discard roomCount
dropVar += ['roomCount']
dropVar

#### HASDRESSINGROOM, DININGROOMSURFACE & HASDININGROOM
Missing(df[['hasDressingRoom', 'diningRoomSurface', 'hasDiningRoom','hasLivingRoom','livingRoomSurface']])

# Discard the variables
dropVar += ['hasDressingRoom', 'diningRoomSurface', 'hasDiningRoom','hasLivingRoom','livingRoomSurface']
dropVar

#### HASOFFICE
# hasoffice may be a nice
# but how to differenciate with nuumber of rooms and bedrooms?
# Is it counted twice?
# also shows many missing values
Missing(df['hasOffice'])

# Discard the variables
dropVar += ['hasOffice']
dropVar

#### KITCHENSURFACE & KITCHEN TYPE

Missing(df[['kitchenSurface', 'kitchenType']])

# kitchenType can be relevant even though it has a large number of Nan
# Recode Nan as No info category
df['kitchenType'] = df['kitchenType'].fillna('No info')

# Discard the variables
dropVar += ['kitchenSurface']
dropVar

#### BATHROOMCOUNT & TOILETCOUNT
# These variables are very similar
# and there is no information about separate or not toilet
Missing(df[['bathroomCount', 'toiletCount']])

# First we assume that any property that has missing value for bathroomCount and has 1 toiletCount, has 1 bathroom
mask = df['bathroomCount'].isna() & (df['toiletCount'] == 1)
df.loc[mask, 'bathroomCount'] = 1

# For the moment, we keep toiletCount as may be used to fill bathroomCount
# Nan remaining as No info
df['bathroomCount'] = df['bathroomCount'].fillna('No info')
df['toiletCount'] = df['toiletCount'].fillna('No info')
Missing(df[['bathroomCount', 'toiletCount']])

### PARKINGCOUNTINDOOR & PARKINGCOUNTOUTDOOR

Missing(df[['parkingCountIndoor', 'parkingCountOutdoor']])

# Parking is a valuable asset, if seller dont specify then we assume there is no parking
df['parkingCountIndoor'] = df['parkingCountIndoor'].fillna(0)
df['parkingCountOutdoor'] = df['parkingCountOutdoor'].fillna(0)
Missing(df[['parkingCountIndoor', 'parkingCountOutdoor']])

### HASGARDEN, GARDENSURFACE, GARDENORIENTATION & HASTERRACE, TERRACESURFACE, TERRACEORIENTATION
Missing(df[['hasGarden', 'gardenSurface', 'gardenOrientation', 'hasTerrace', 'terraceSurface', 'terraceOrientation']])
print(df['hasGarden'].value_counts())
print(df['hasTerrace'].value_counts())

# Recode to 1 (has) and O (don't)
df['hasGarden'] = df['hasGarden'].apply(lambda x: 1 if x == True else 0)
df['hasTerrace'] = df['hasTerrace'].apply(lambda x: 1 if x == True else 0)

# If no garden / no terrace then surface is 0
df.loc[df['hasGarden'] == 0, 'gardenSurface'] = 0
df.loc[df['hasTerrace'] == 0, 'terraceSurface'] = 0

# If no garden / no terrace then Orientation is empty
df.loc[df['hasGarden'] == 0, 'gardenOrientation'] = "No garden"
df.loc[df['hasTerrace'] == 0, 'terraceOrientation'] = "No terrace"

df['gardenSurface'] = df['gardenSurface'].fillna('No info')
df['terraceSurface'] = df['terraceSurface'].fillna('No info')

# Discard the variables
dropVar += ['gardenSurface','terraceSurface','gardenOrientation', 'terraceOrientation']
dropVar

# Finally we decide to remove Surface and Orientation as they have lower importance
dropVar += ['gardenSurface', 'gardenOrientation', 'terraceSurface', 'terraceOrientation']
dropVar
'hasGarden', 'gardenSurface', 'gardenOrientation', 'hasTerrace', 'terraceSurface', 'terraceOrientation'

### HASSWIMMINGPOOL
df['hasSwimmingPool'].value_counts()

# hasswimmingpool has many missing values,but in Belgium it is rare to have a swimming pool
# we can assume that missing values mean there is no swimming pool
# and that having swimming pool correlates with price
df['hasSwimmingPool'] = df['hasSwimmingPool'].apply(lambda x: 1 if x == True else 0)

### HASVISIOPHONE & HASARMOREDDOOR & HASAIRCONDITIONING & HASFIREPLACE

# These may be considered as extras but probably not one that a buyer would consider
# and shows many missing values
# and a feature that a seller would not care about informing
Missing(df[['hasArmoredDoor', 'hasVisiophone', 'hasAirConditioning', 'hasFireplace']])

# Discard the variables
dropVar += ['hasArmoredDoor', 'hasVisiophone', 'hasAirConditioning', 'hasFireplace']
dropVar

### FLOOD ZONE TYPE
Missing(df['floodZoneType'])
# Whether the property is in a potential flooding area is relevant
# Recode Nan to No info
df['floodZoneType'] = df['floodZoneType'].fillna('No info')

### HASATTIC & HASBASEMENT
Missing(df[['hasAttic','hasBasement']])
# There are many Nan but this is a feature a seller would highlight
# Recoding 0/1 and assuming Nan means no attic or basement
df['hasAttic'] = df['hasAttic'].apply(lambda x: 1 if x == True else 0)
df['hasBasement'] = df['hasBasement'].apply(lambda x: 1 if x == True else 0)

### HASLIFT
Missing(df.loc[df['type'] == 'House', 'hasLift'])

# Most houses dont have a lift
df.loc[(df['type'] == 'House') & (df['hasLift'].isna()), 'hasLift'] = 0
Missing(df.loc[df['type'] == 'Apartment', 'hasLift'])

# Apartments often have a lift we can assume that 41% Nan mean No lift, for 59% with lift
df.loc[(df['type'] == 'Apartment') & (df['hasLift'].isna()), 'hasLift'] = 0

### HEATING
Missing(df[['heatingType']])
Missing(df[['hasHeatPump', 'hasPhotovoltaicPanels', 'hasThermicPanels']])

# 🛠 Convertir les colonnes en 0/1 (si ce sont des floats)
for col in ["hasHeatPump", "hasPhotovoltaicPanels", "hasThermicPanels"]:
    df[col] = df[col].fillna(0).astype(int)
# 🔆 Créer un masque pour les lignes où au moins une source solaire est activée
solar_mask = (
    (df["hasThermicPanels"] == 1) | (df["hasPhotovoltaicPanels"] == 1)
)

# ✏️ Modifier uniquement les lignes où heatingType est manquant
modif_count = df.loc[solar_mask & df["heatingType"].isna(), "heatingType"].shape[0]
df.loc[solar_mask & df["heatingType"].isna(), "heatingType"] = "SOLAR"
print(f"✅ {modif_count} lignes modifiées avec 'SOLAR' dans 'heatingtype'.")

# heatingType fill Nan with Heat pump where True
df.loc[(df["heatingType"].isna()) & (df["hasHeatPump"] == 1), "heatingType"] = "Heat Pump"
df["heatingType"].value_counts()

# Recode Nan to No info
df['heatingType'] = df['heatingType'].fillna('No info')

# Drop variables used to fill heatingType
dropVar += ["hasHeatPump", "hasPhotovoltaicPanels", "hasThermicPanels"]
dropVar

### BUILDINGCONDITION & BUILDINGCONSTRUCTIONYEAR
Missing(df[['buildingCondition', 'buildingConstructionYear']])

df['buildingCondition'] = df['buildingCondition'].fillna('No info')
df['buildingConstructionYear'] = df['buildingConstructionYear'].fillna('No info')

### FACEDECOUNT, FLOORCOUNT, STREETFACADEWIDTH
Missing(df[['facedeCount', 'floorCount', 'streetFacadeWidth']])

# There is a high amount of Nan
# and variables less relevant
# Drop variables
dropVar += ['facedeCount', 'floorCount', 'streetFacadeWidth']
dropVar
Missing(df.loc[df['type'] == 'House', 'hasLift'])

# Most houses dont have a lift
df.loc[(df['type'] == 'House') & (df['hasLift'].isna()), 'hasLift'] = 0
Missing(df.loc[df['type'] == 'Apartment', 'hasLift'])

# Apartments often have a lift we can assume that 41% Nan mean No lift, for 59% with lift
df.loc[(df['type'] == 'Apartment') & (df['hasLift'].isna()), 'hasLift'] = 0

### HEATING
Missing(df[['heatingType']])
Missing(df[['hasHeatPump', 'hasPhotovoltaicPanels', 'hasThermicPanels']])

# 🛠 Convertir les colonnes en 0/1 (si ce sont des floats)
for col in ["hasHeatPump", "hasPhotovoltaicPanels", "hasThermicPanels"]:
    df[col] = df[col].fillna(0).astype(int)
# 🔆 Créer un masque pour les lignes où au moins une source solaire est activée
solar_mask = (
    (df["hasThermicPanels"] == 1) | (df["hasPhotovoltaicPanels"] == 1)
)

# ✏️ Modifier uniquement les lignes où heatingType est manquant
modif_count = df.loc[solar_mask & df["heatingType"].isna(), "heatingType"].shape[0]
df.loc[solar_mask & df["heatingType"].isna(), "heatingType"] = "SOLAR"
print(f"✅ {modif_count} lignes modifiées avec 'SOLAR' dans 'heatingtype'.")

# heatingType fill Nan with Heat pump where True
df.loc[(df["heatingType"].isna()) & (df["hasHeatPump"] == 1), "heatingType"] = "Heat Pump"
df["heatingType"].value_counts()

# Recode Nan to No info
df['heatingType'] = df['heatingType'].fillna('No info')

# Drop variables used to fill heatingType
dropVar += ["hasHeatPump", "hasPhotovoltaicPanels", "hasThermicPanels"]
dropVar

### BUILDINGCONDITION & BUILDINGCONSTRUCTIONYEAR
Missing(df[['buildingCondition', 'buildingConstructionYear']])

df['buildingCondition'] = df['buildingCondition'].fillna('No info')
df['buildingConstructionYear'] = df['buildingConstructionYear'].fillna('No info')

### FACEDECOUNT, FLOORCOUNT, STREETFACADEWIDTH
Missing(df[['facedeCount', 'floorCount', 'streetFacadeWidth']])

# There is a high amount of Nan
# and variables less relevant
# Drop variables
dropVar += ['facedeCount', 'floorCount', 'streetFacadeWidth']
dropVar

### HABITABLE SURFACE
Missing(df['habitableSurface'])

# Very relevant, recode Nan to No info
df['habitableSurface'] = df['habitableSurface'].fillna('No info')

### LAND SURFACE
Missing(df['landSurface'])

# Many Nan but expected as Apartments should have No land
df['landSurface'] = df['landSurface'].astype('object')
df.loc[(df['type'] == 'Apartment') & (df['landSurface'].isna()), 'landSurface'] = "Apt: no land"

# For houses, fillNA with No info on land
df.loc[(df['type'] == 'House') & (df['landSurface'].isna()), 'landSurface'] = "No info on land"

### EPC SCORE
Missing(df['epcScore'])

df['epcScore'].value_counts()

# epcScore is relevant
# It has some wrong values and 15% nan: replace with No info
correct = ['A', 'A+', 'A++', 'B', 'C', 'D', 'E', 'F', 'G']
df['epcScore'] = df['epcScore'].where(df['epcScore'].isin(correct), 'No info')

## OUTPUT DATA
# Clear the clear_terminal
clear_terminal()

print("""
===========================================================================================
    IMMOWEB DATASET CLEANED: RESULTS SUMMARY
===========================================================================================
""")

print(f"Cleaning ends by dropping {len(dropVar)} columns")

# Drop the colums
df = df.drop(dropVar, axis=1)
# Check no missing values
Missing(df)

# Split the variables into numerical and categorical cols, will be useful later for the analysis
numeric_cols = df.select_dtypes(include=np.number).columns
numeric_cols

categoric_cols = df.select_dtypes(exclude=np.number).columns
categoric_cols

# Reduce the memory used
MemOptimisation(df)

# Display info about the data remaininig
Descriptives(df)
# Generate the report
profile = ProfileReport(df,title="Immoweb: Data Profile")

# Save the data cleaned
df.to_csv("Data/Cleaned - immoweb-dataset.csv", index=False, encoding="utf-8-sig")
print("Cleaned data is saved to: Data/Cleaned - immoweb-dataset.csv")

