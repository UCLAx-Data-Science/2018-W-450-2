import numpy as np
import pandas as pd

zoning_df = pd.read_csv('data/zoning.csv')
listing_df = pd.read_csv('data/listing.csv')
sale_df = pd.read_csv('data/sale.csv')

housing_df = pd.merge(zoning_df, listing_df, left_on="Id", right_on="Id")
housing_df = pd.merge(housing_df, sale_df, left_on="Id", right_on="Id")

housing_df.set_index("Id", inplace=True)

for column in housing_df.select_dtypes(['object']).columns:
    housing_df[column] = housing_df[column].astype('category')

housing_df.MSSubClass = housing_df.MSSubClass.astype('category')
housing_df.OverallQual = housing_df.OverallQual.astype('category')
housing_df.OverallCond = housing_df.OverallCond.astype('category')
housing_df.BsmtFullBath = housing_df.BsmtFullBath.astype('category')
housing_df.BsmtHalfBath = housing_df.BsmtHalfBath.astype('category')
housing_df.FullBath = housing_df.FullBath.astype('category')
housing_df.HalfBath = housing_df.HalfBath.astype('category')
housing_df.BedroomAbvGr = housing_df.BedroomAbvGr.astype('category')
housing_df.KitchenAbvGr = housing_df.KitchenAbvGr.astype('category')
housing_df.TotRmsAbvGrd = housing_df.TotRmsAbvGrd.astype('category')
housing_df.Fireplaces = housing_df.Fireplaces.astype('category')
housing_df.GarageCars = housing_df.GarageCars.astype('category')
housing_df.MoSold = housing_df.MoSold.astype('category')

housing_df.LotFrontage.fillna(housing_df.LotFrontage.mean(), inplace=True)
housing_df.MasVnrArea.fillna(housing_df.MasVnrArea.mean(), inplace=True)
housing_df.GarageYrBlt.fillna(housing_df.GarageYrBlt.mean(), inplace=True)


empty_means_without = ["Alley","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1",
                        "BsmtFinType2", "FireplaceQu","GarageType","GarageFinish",
                        "GarageQual","GarageCond","PoolQC","Fence","MiscFeature"]

def replace_empty(feature, value):
    housing_df[feature].cat.add_categories([value], inplace=True)
    housing_df[feature].fillna(value, inplace=True)


for feature in empty_means_without:
    replace_empty(feature, "None")

housing_df.dropna(inplace=True)

numeric_df = housing_df.select_dtypes(exclude=['category']).copy()
numeric_df.drop("SalePrice", axis=1, inplace=True)
numeric_sc_df = (numeric_df - numeric_df.mean())/numeric_df.std()
numeric_log_df = np.log(numeric_df + 1)
numeric_log_sc_df = (numeric_df - numeric_df.mean())/numeric_df.std()

outliers = [198, 524, 1174, 1183, 1299, 186, 692, 770, 
            179, 225, 804, 889, 1387, 497]
numeric_final_df = numeric_log_sc_df.drop(outliers, axis=0)
housing_ouliers_removed_df = housing_df.drop(outliers, axis=0)