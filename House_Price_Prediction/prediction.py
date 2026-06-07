
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
# Load dataset and check basic info
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# check for NA values
columns = list(train_df.columns)

test_columns = list(test_df.columns)
#print('All Columns:\n', test_columns)


columns_na = {}

# For training data
for col in columns:
    if train_df[col].isna().sum() > 0:
        columns_na.update({col: train_df[col].isna().sum()})


test_columns_na = {}
#for test data
for col in test_columns:
    if test_df[col].isna().sum() > 0:
        test_columns_na.update({col: test_df[col].isna().sum()})



# Dropping columns which has missing values more than 50% because they wont give any useful info to model
# In our case theya are [MasVnrType,FireplaceQu,PoolQC,Fence,MiscFeature]

cols_to_drop = ['MasVnrType', 'Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature']
train_df.drop(columns=cols_to_drop, axis=1, inplace=True)
#print('DF shape after dropping NaN columns :', train_df.shape)

# for test data, drop columns
test_cols_to_drop = ['MasVnrType', 'Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature']
test_df.drop(columns=test_cols_to_drop, axis=1, inplace=True)
print('Test DF shape after dropping NaN columns :', test_df.shape)

# Impute remaining columns with NaN values
# column 'LotFrontage' => we can take mean of the values and replace NaNa values with it
#train_df['LotFrontage'].hist()
#test_df['LotFrontage'].hist()
# plt.show()
# Histogram shows that data for 'LotFrontage' is not biased an has normal distribution, so we can use mean
LotFrontage_mean = train_df['LotFrontage'].mean()
test_LotFrontage_mean = test_df['LotFrontage'].mean()
train_df['LotFrontage'] = train_df['LotFrontage'].fillna(LotFrontage_mean)
test_df['LotFrontage'] = test_df['LotFrontage'].fillna(test_LotFrontage_mean)


# column 'MasVnrArea' => Histogram shows that data is not normally distributed and is right skewed, so we will use median to replace values
MasVnrArea_median = train_df['MasVnrArea'].median()
train_df['MasVnrArea'] = train_df['MasVnrArea'].fillna(MasVnrArea_median)


# For Test DF :
test_MasVnrArea_median = test_df['MasVnrArea'].median()
test_df['MasVnrArea'] = test_df['MasVnrArea'].fillna(test_MasVnrArea_median)


# column 'BsmtQual' => From value counts we can see that 'TA' has the most count, its safe to replace NaN values with 'TA'
# print(train_df['BsmtQual'].value_counts())
train_df['BsmtQual'] = train_df['BsmtQual'].fillna('TA')


# For Test DF:
test_df['BsmtQual'] = test_df['BsmtQual'].fillna('TA')

# columns 'BsmtCond' => From value counts we can see that 'TA' has the most count, its safe to replace NaN values with 'TA'

train_df['BsmtCond'] = train_df['BsmtCond'].fillna('TA')
test_df['BsmtCond'] = test_df['BsmtCond'].fillna('TA')

# column 'BsmtExposure' => From value counts we can see that 'NO' has the most count, its safe to replace NaN values with 'NO'

train_df['BsmtExposure'] = train_df['BsmtExposure'].fillna('NO')
test_df['BsmtExposure'] = test_df['BsmtExposure'].fillna('NO')

# column  'BsmtFinType1' => From value counts we can see that 'NO' has the most count, its safe to replace NaN values with 'Unf'

train_df['BsmtFinType1'] = train_df['BsmtFinType1'].fillna('Unf')
test_df['BsmtFinType1'] = test_df['BsmtFinType1'].fillna('Unf')

# columns 'BsmtFinType2' => From value counts we can see that 'NO' has the most count, its safe to replace NaN values with 'Unf'

train_df['BsmtFinType2'] = train_df['BsmtFinType2'].fillna('Unf')
test_df['BsmtFinType2'] = test_df['BsmtFinType2'].fillna('Unf')

# columns 'Electrical' => From value counts we can see that 'NO' has the most count, its safe to replace NaN values with 'SBrkr'

train_df['Electrical'] = train_df['Electrical'].fillna('SBrkr')
test_df['Electrical'] = test_df['Electrical'].fillna('SBrkr')

# Columns 'GarageType' => From value counts we can see that 'NO' has the most count, its safe to replace NaN values with 'Attchd'

train_df['GarageType'] = train_df['GarageType'].fillna('Attchd')
test_df['GarageType'] = test_df['GarageType'].fillna('Attchd')

# columns 'GarageYrBlt' => Dstribution for this column is left skewed, so replace missing values by Median

GarageYrBlt_median = train_df['GarageYrBlt'].median()
train_df['GarageYrBlt'] = train_df['GarageYrBlt'].fillna(GarageYrBlt_median)

test_GarageYrBlt_median = test_df['GarageYrBlt'].median()
test_df['GarageYrBlt'] = test_df['GarageYrBlt'].fillna(test_GarageYrBlt_median)

# columns 'GarageFinish' => From value counts we can see that 'NO' has the most count, its safe to replace NaN values with 'Unf'

train_df['GarageFinish'] = train_df['GarageFinish'].fillna('Unf')
test_df['GarageFinish'] = test_df['GarageFinish'].fillna('Unf')

# column 'GarageQual' => From value counts we can see that 'NO' has the most count, its safe to replace NaN values with 'TA'

train_df['GarageQual'] = train_df['GarageQual'].fillna('TA')
test_df['GarageQual'] = test_df['GarageQual'].fillna('TA')

# column 'GarageCond' => From value counts we can see that 'NO' has the most count, its safe to replace NaN values with 'TA'

train_df['GarageCond'] = train_df['GarageCond'].fillna('TA')
test_df['GarageCond'] = test_df['GarageCond'].fillna('TA')

print(test_df.isna().any())
#######################################################################################################################
def calculate_iqr(column):
    q1 = train_df[column].quantile(0.25)
    q3 = train_df[column].quantile(0.75)
    iqr = 1.5 * (q3 - q1)

    upper_limit = q3 + iqr
    lower_limit = q1 - iqr

    return upper_limit, lower_limit


# 'MSSubClass'
upper, lower = calculate_iqr('MSSubClass')
MSSubClass_indices = list(train_df[(train_df['MSSubClass'] > upper) | (train_df['MSSubClass'] < lower)].index)
train_df.drop(MSSubClass_indices, inplace=True)


#Test
test_MSSubClass_indices = list(test_df[(test_df['MSSubClass'] > upper) | (test_df['MSSubClass'] < lower)].index)
test_df.drop(test_MSSubClass_indices, inplace=True)

# LotArea
upper, lower = calculate_iqr('LotArea')
lotarea_indices = list(train_df[(train_df['LotArea'] > upper) | (train_df['LotArea'] < lower)].index)
train_df.drop(lotarea_indices, inplace=True)
test_lotarea_indices = list(test_df[(test_df['LotArea'] > upper) | (test_df['LotArea'] < lower)].index)
test_df.drop(test_lotarea_indices, inplace=True)


# YearBuilt
upper, lower = calculate_iqr('YearBuilt')
yearbuilt_indices = list(train_df[(train_df['YearBuilt'] > upper) | (train_df['YearBuilt'] < lower)].index)
train_df.drop(yearbuilt_indices, inplace=True)
test_yearbuilt_indices = list(test_df[(test_df['YearBuilt'] > upper) | (test_df['YearBuilt'] < lower)].index)
test_df.drop(test_yearbuilt_indices, inplace=True)


# BsmtFinSF2
upper, lower = calculate_iqr('BsmtFinSF2')
BsmtFinSF2_indices = list(train_df[(train_df['BsmtFinSF2'] > upper) | (train_df['BsmtFinSF2'] < lower)].index)
train_df.drop(BsmtFinSF2_indices, inplace=True)
test_BsmtFinSF2_indices = list(test_df[(test_df['BsmtFinSF2'] > upper) | (test_df['BsmtFinSF2'] < lower)].index)
test_df.drop(test_BsmtFinSF2_indices, inplace=True)


# BsmtUnfSF
upper, lower = calculate_iqr('BsmtUnfSF')
BsmtUnfSF_indices = list(train_df[(train_df['BsmtUnfSF'] > upper) | (train_df['BsmtUnfSF'] < lower)].index)
train_df.drop(BsmtUnfSF_indices, inplace=True)
test_BsmtUnfSF_indices = list(test_df[(test_df['BsmtUnfSF'] > upper) | (test_df['BsmtUnfSF'] < lower)].index)
test_df.drop(test_BsmtUnfSF_indices, inplace=True)


# TotalBsmtSF
upper, lower = calculate_iqr('TotalBsmtSF')
TotalBsmtSF_indices = list(train_df[(train_df['TotalBsmtSF'] > upper) | (train_df['TotalBsmtSF'] < lower)].index)
train_df.drop(TotalBsmtSF_indices, inplace=True)
test_TotalBsmtSF_indices = list(test_df[(test_df['TotalBsmtSF'] > upper) | (test_df['TotalBsmtSF'] < lower)].index)
test_df.drop(test_TotalBsmtSF_indices, inplace=True)


# 1stFlrSF
upper, lower = calculate_iqr('1stFlrSF')
firstFlrSF_indices = list(train_df[(train_df['1stFlrSF'] > upper) | (train_df['1stFlrSF'] < lower)].index)
train_df.drop(firstFlrSF_indices, inplace=True)
test_firstFlrSF_indices = list(test_df[(test_df['1stFlrSF'] > upper) | (test_df['1stFlrSF'] < lower)].index)
test_df.drop(test_firstFlrSF_indices, inplace=True)


# LowQualFinSF
upper, lower = calculate_iqr('LowQualFinSF')
LowQualFinSF_indices = list(train_df[(train_df['LowQualFinSF'] > upper) | (train_df['LowQualFinSF'] < lower)].index)
train_df.drop(LowQualFinSF_indices, inplace=True)
test_LowQualFinSF_indices = list(test_df[(test_df['LowQualFinSF'] > upper) | (test_df['LowQualFinSF'] < lower)].index)
test_df.drop(test_LowQualFinSF_indices, inplace=True)


# GRLivArea
upper, lower = calculate_iqr('GrLivArea')
GRLivArea_indices = list(train_df[(train_df['GrLivArea'] > upper) | (train_df['GrLivArea'] < lower)].index)
train_df.drop(GRLivArea_indices, inplace=True)
test_GRLivArea_indices = list(test_df[(test_df['GrLivArea'] > upper) | (test_df['GrLivArea'] < lower)].index)
test_df.drop(test_GRLivArea_indices, inplace=True)



# GarageArea
upper, lower = calculate_iqr('GarageArea')
garagearea_indices = list(train_df[(train_df['GarageArea'] > upper) | (train_df['GarageArea'] < lower)].index)
train_df.drop(garagearea_indices, inplace=True)
test_garagearea_indices = list(test_df[(test_df['GarageArea'] > upper) | (test_df['GarageArea'] < lower)].index)
test_df.drop(test_garagearea_indices, inplace=True)



# WoodDeckSF
upper, lower = calculate_iqr('WoodDeckSF')
WoodDeckSF_indices = list(train_df[(train_df['WoodDeckSF'] > upper) | (train_df['WoodDeckSF'] < lower)].index)
train_df.drop(WoodDeckSF_indices, inplace=True)
test_WoodDeckSF_indices = list(test_df[(test_df['WoodDeckSF'] > upper) | (test_df['WoodDeckSF'] < lower)].index)
test_df.drop(test_WoodDeckSF_indices, inplace=True)


# OpenPorchSF
upper, lower = calculate_iqr('OpenPorchSF')
OpenPorchSF_indices = list(train_df[(train_df['OpenPorchSF'] > upper) | (train_df['OpenPorchSF'] < lower)].index)
train_df.drop(OpenPorchSF_indices, inplace=True)
test_OpenPorchSF_indices = list(test_df[(test_df['OpenPorchSF'] > upper) | (test_df['OpenPorchSF'] < lower)].index)
test_df.drop(test_OpenPorchSF_indices, inplace=True)


# EnclosedPorch
upper, lower = calculate_iqr('EnclosedPorch')
EnclosedPorch_indices = list(train_df[(train_df['EnclosedPorch'] > upper) | (train_df['EnclosedPorch'] < lower)].index)
train_df.drop(EnclosedPorch_indices, inplace=True)
test_EnclosedPorch_indices = list(test_df[(test_df['EnclosedPorch'] > upper) | (test_df['EnclosedPorch'] < lower)].index)
test_df.drop(test_EnclosedPorch_indices, inplace=True)



# 3SsnPorch
upper, lower = calculate_iqr('3SsnPorch')
SsnPorch_indices = list(train_df[(train_df['3SsnPorch'] > upper) | (train_df['3SsnPorch'] < lower)].index)
train_df.drop(SsnPorch_indices, inplace=True)
test_SsnPorch_indices = list(test_df[(test_df['3SsnPorch'] > upper) | (test_df['3SsnPorch'] < lower)].index)
test_df.drop(test_SsnPorch_indices, inplace=True)


# ScreenPorch
upper, lower = calculate_iqr('ScreenPorch')
ScreenPorch_indices = list(train_df[(train_df['ScreenPorch'] > upper) | (train_df['ScreenPorch'] < lower)].index)
train_df.drop(ScreenPorch_indices, inplace=True)
test_ScreenPorch_indices = list(test_df[(test_df['ScreenPorch'] > upper) | (test_df['ScreenPorch'] < lower)].index)
test_df.drop(test_ScreenPorch_indices, inplace=True)


# MiscVal
upper, lower = calculate_iqr('MiscVal')
MiscVal_indices = list(train_df[(train_df['MiscVal'] > upper) | (train_df['MiscVal'] < lower)].index)
train_df.drop(MiscVal_indices, inplace=True)
test_MiscVal_indices = list(test_df[(test_df['MiscVal'] > upper) | (test_df['MiscVal'] < lower)].index)
test_df.drop(test_MiscVal_indices, inplace=True)

#########################################################################################################################


# train_df.drop(['BsmtFinType2'], inplace=True)
train_df = train_df.drop(columns=['Street', 'Utilities', 'BsmtFinType2', 'Condition2', 'Exterior2nd',
                                  'GarageFinish', 'LandSlope', 'ExterCond', 'BsmtCond', 'BsmtExposure', 'GarageCond'],
                         axis=1)

test_df = test_df.drop(columns=['Street', 'Utilities', 'BsmtFinType2', 'Condition2', 'Exterior2nd',
                                  'GarageFinish', 'LandSlope', 'ExterCond', 'BsmtCond', 'BsmtExposure', 'GarageCond'],
                         axis=1)


# check if the below columns have Nominal or Ordinal Data
nominal_col = ['MSZoning', 'LotShape', 'LandContour', 'LotConfig', 'Neighborhood', 'Condition1', 'RoofStyle',
               'RoofMatl',
               'Exterior1st', 'Exterior2nd', 'Foundation', 'Heating', 'CentralAir', 'Electrical', 'Functional',
               'GarageType', 'GarageFinish', 'PavedDrive', 'SaleType', 'SaleCondition', 'BldgType']
ordinal_col = ['ExterQual', 'BsmtQual', 'BsmtFinType1', 'HeatingQC', 'KitchenQual', 'GarageQual', 'GarageCond']


# For nominal columns, try to simplify the values and do one hot encoding
# MSZoning
mszone_map = {'RL': 'Residential', 'RM': 'Residential', 'FV': 'Residential', 'RH': 'Residential',
              'C (all)': 'Commercial'}
train_df.replace({'MSZoning': mszone_map}, inplace=True)
test_df.replace({'MSZoning': mszone_map}, inplace=True)
mszone_dummy = pd.get_dummies(train_df['MSZoning'], prefix='mszone', drop_first=True, dtype=int)
test_mszone_dummy = pd.get_dummies(test_df['MSZoning'], prefix='mszone', drop_first=True, dtype=int)
train_df = pd.concat([train_df, mszone_dummy], axis=1)
test_df = pd.concat([test_df, test_mszone_dummy], axis=1)
train_df.drop(columns=['MSZoning'], inplace=True)
test_df.drop(columns=['MSZoning'], inplace=True)


# LotShape
lotshape_map = {'Reg': 'Regular', 'IR1': 'Irregular', 'IR2': 'Irregular', 'IR3': 'Irregular'}
train_df.replace({'LotShape': lotshape_map}, inplace=True)
test_df.replace({'LotShape': lotshape_map}, inplace=True)
lotshape_dummy = pd.get_dummies(train_df['LotShape'], prefix='lotshape', drop_first=True, dtype=int)
test_lotshape_dummy = pd.get_dummies(test_df['LotShape'], prefix='lotshape', drop_first=True, dtype=int)
train_df = pd.concat([train_df, lotshape_dummy], axis=1)
test_df = pd.concat([test_df, test_lotshape_dummy], axis=1)
train_df.drop(columns=['LotShape'], inplace=True)
test_df.drop(columns=['LotShape'], inplace=True)

# LandContour
lotcontour_dummy = pd.get_dummies(train_df['LandContour'], prefix='lotcontour', drop_first=True, dtype=int)
test_lotcontour_dummy = pd.get_dummies(test_df['LandContour'], prefix='lotcontour', drop_first=True, dtype=int)
train_df = pd.concat([train_df, lotcontour_dummy], axis=1)
test_df = pd.concat([test_df, test_lotcontour_dummy], axis=1)
train_df.drop(columns=['LandContour'], inplace=True)
test_df.drop(columns=['LandContour'], inplace=True)

# LotConfig
lotconfig_map = {'FR2': 'Front', 'FR3': 'Front'}
train_df.replace({'LotConfig': lotconfig_map}, inplace=True)
test_df.replace({'LotConfig': lotconfig_map}, inplace=True)
lotconfig_dummy = pd.get_dummies(train_df['LotConfig'], prefix='lotconfig', drop_first=True, dtype=int)
test_lotconfig_dummy = pd.get_dummies(test_df['LotConfig'], prefix='lotconfig', drop_first=True, dtype=int)
train_df = pd.concat([train_df, lotconfig_dummy], axis=1)
test_df = pd.concat([test_df, test_lotconfig_dummy], axis=1)
train_df.drop(columns=['LotConfig'], inplace=True)
test_df.drop(columns=['LotConfig'], inplace=True)

# Neighborhood
neighborhood_map = {'SawyerW':'Sawyer'}
train_df.replace({'Neighborhood': neighborhood_map}, inplace=True)
test_df.replace({'Neighborhood': neighborhood_map}, inplace=True)
neighborhood_dummy = pd.get_dummies(train_df['Neighborhood'], prefix='neighborhood', drop_first=True, dtype=int)
test_neighborhood_dummy = pd.get_dummies(test_df['Neighborhood'], prefix='neighborhood', drop_first=True, dtype=int)
train_df = pd.concat([train_df,neighborhood_dummy], axis=1)
test_df = pd.concat([test_df,test_neighborhood_dummy], axis=1)
train_df.drop(columns=['Neighborhood'], inplace=True)
test_df.drop(columns=['Neighborhood'], inplace=True)

# Condition1
condition1_map = {'RRAe':'Railroad', 'RRAn': 'Railroad', 'RRNe':'Railroad', 'RRNn': 'Railroad'}
train_df.replace({'Condition1': condition1_map}, inplace=True)
test_df.replace({'Condition1': condition1_map}, inplace=True)
condition1_dummy = pd.get_dummies(train_df['Condition1'], prefix='condition1', drop_first=True, dtype=int)
test_condition1_dummy = pd.get_dummies(test_df['Condition1'], prefix='condition1', drop_first=True, dtype=int)
train_df = pd.concat([train_df,condition1_dummy], axis=1)
test_df = pd.concat([test_df,test_condition1_dummy], axis=1)
train_df.drop(columns=['Condition1'], inplace=True)
test_df.drop(columns=['Condition1'], inplace=True)

# RoofStyle
roofstyle_dummy = pd.get_dummies(train_df['RoofStyle'], prefix='roofstyle', drop_first=True, dtype=int)
test_roofstyle_dummy = pd.get_dummies(test_df['RoofStyle'], prefix='roofstyle', drop_first=True, dtype=int)
train_df = pd.concat([train_df,roofstyle_dummy], axis=1)
test_df = pd.concat([test_df,test_roofstyle_dummy], axis=1)
train_df.drop(columns=['RoofStyle'], inplace=True)
test_df.drop(columns=['RoofStyle'], inplace=True)


# RoofMatl
roofmatl_map = {'CompShg':'Composite', 'Tar&Grv': 'Tar_Gravel'}
train_df.replace({'RoofMatl': roofmatl_map}, inplace=True)
test_df.replace({'RoofMatl': roofmatl_map}, inplace=True)
roofmatl_dummy = pd.get_dummies(train_df['RoofMatl'], prefix='roofmatl', drop_first=True, dtype=int)
test_roofmatl_dummy = pd.get_dummies(test_df['RoofMatl'], prefix='roofmatl', drop_first=True, dtype=int)
train_df = pd.concat([train_df,roofmatl_dummy], axis=1)
test_df = pd.concat([test_df,test_roofmatl_dummy], axis=1)
train_df.drop(columns=['RoofMatl'], inplace=True)
test_df.drop(columns=['RoofMatl'], inplace=True)

# Exterior1st
exterior1_map = {'WdShing':'Wood', 'Wd Sdng': 'Wood', 'BrkFace': 'Brick', 'BrkComm': 'Brick', 'Plywood': 'Wood'}
train_df.replace({'Exterior1st': exterior1_map}, inplace=True)
test_df.replace({'Exterior1st': exterior1_map}, inplace=True)
exterior1_dummy = pd.get_dummies(train_df['Exterior1st'], prefix='exterior1st', drop_first=True, dtype=int)
test_exterior1_dummy = pd.get_dummies(train_df['Exterior1st'], prefix='exterior1st', drop_first=True, dtype=int)
train_df = pd.concat([train_df,exterior1_dummy], axis=1)
test_df = pd.concat([test_df,test_exterior1_dummy], axis=1)
train_df.drop(columns=['Exterior1st'], inplace=True)
test_df.drop(columns=['Exterior1st'], inplace=True)

# Foundation
foundation_dummy = pd.get_dummies(train_df['Foundation'], prefix='foundation', drop_first=True, dtype=int)
test_foundation_dummy = pd.get_dummies(test_df['Foundation'], prefix='foundation', drop_first=True, dtype=int)
train_df = pd.concat([train_df,foundation_dummy], axis=1)
test_df = pd.concat([test_df,test_foundation_dummy], axis=1)
train_df.drop(columns=['Foundation'], inplace=True)
test_df.drop(columns=['Foundation'], inplace=True)

# Heating
heating_dummy = pd.get_dummies(train_df['Heating'], prefix='heating', drop_first=True, dtype=int)
test_heating_dummy = pd.get_dummies(test_df['Heating'], prefix='heating', drop_first=True, dtype=int)
train_df = pd.concat([train_df,heating_dummy], axis =1)
test_df = pd.concat([test_df,test_heating_dummy], axis =1)
train_df.drop(columns=['Heating'], inplace=True)
test_df.drop(columns=['Heating'], inplace=True)

# CentralAir
centralair_dummy = pd.get_dummies(train_df['CentralAir'], prefix='centralair', drop_first=True, dtype=int)
test_centralair_dummy = pd.get_dummies(test_df['CentralAir'], prefix='centralair', drop_first=True, dtype=int)
train_df = pd.concat([train_df,centralair_dummy], axis=1)
test_df = pd.concat([test_df,test_centralair_dummy], axis=1)
train_df.drop(columns=['CentralAir'], inplace=True)
test_df.drop(columns=['CentralAir'], inplace=True)

# Electrical
electrical_map = {'FuseA':'Fuse', 'FuseF': 'Fuse', 'FuseP': 'Fuse'}
train_df.replace({'Electrical': electrical_map}, inplace=True)
test_df.replace({'Electrical': electrical_map}, inplace=True)
electrical_dummy = pd.get_dummies(train_df['Electrical'], prefix='electrical', drop_first=True, dtype=int)
test_electrical_dummy = pd.get_dummies(test_df['Electrical'], prefix='electrical', drop_first=True, dtype=int)
train_df = pd.concat([train_df,electrical_dummy], axis=1)
test_df = pd.concat([test_df,test_electrical_dummy], axis=1)
train_df.drop(columns=['Electrical'], inplace=True)
test_df.drop(columns=['Electrical'], inplace=True)

# Functional
funcational_map = {'Min1':'Minor', 'Min2': 'Minor', 'Maj2': 'Major', 'Maj1':'Major'}
train_df.replace({'Functional': funcational_map}, inplace=True)
test_df.replace({'Functional': funcational_map}, inplace=True)
functional_dummy = pd.get_dummies(train_df['Functional'], prefix='functional', drop_first=True, dtype=int)
test_functional_dummy = pd.get_dummies(test_df['Functional'], prefix='functional', drop_first=True, dtype=int)
train_df = pd.concat([train_df,functional_dummy], axis=1)
test_df = pd.concat([test_df,test_functional_dummy], axis=1)
train_df.drop(columns=['Functional'], inplace=True)
test_df.drop(columns=['Functional'], inplace=True)

#GarageType
garagetype_dummy = pd.get_dummies(train_df['GarageType'], prefix='garagetype', drop_first=True, dtype=int)
test_garagetype_dummy = pd.get_dummies(test_df['GarageType'], prefix='garagetype', drop_first=True, dtype=int)
train_df = pd.concat([train_df,garagetype_dummy], axis=1)
test_df = pd.concat([test_df,test_garagetype_dummy], axis=1)
train_df.drop(columns=['GarageType'], inplace=True)
test_df.drop(columns=['GarageType'], inplace=True)

# PavedDrive
paveddrive_dummy = pd.get_dummies(train_df['PavedDrive'], prefix='paveddrive', drop_first=True, dtype=int)
test_paveddrive_dummy = pd.get_dummies(test_df['PavedDrive'], prefix='paveddrive', drop_first=True, dtype=int)
train_df = pd.concat([train_df,paveddrive_dummy], axis=1)
test_df = pd.concat([test_df,test_paveddrive_dummy], axis=1)
train_df.drop(columns=['PavedDrive'], inplace=True)
test_df.drop(columns=['PavedDrive'], inplace=True)

# SaleType
saletype_map = {'WD':'Warranty', 'CWD': 'Warranty', 'ConLI': 'Contract', 'ConLw':'Contract', 'ConLD': 'Contract', 'Con': 'Contract'}
train_df.replace({'SaleType': saletype_map}, inplace=True)
test_df.replace({'SaleType': saletype_map}, inplace=True)
saletype_dummy = pd.get_dummies(train_df['SaleType'], prefix='saletype', drop_first=True, dtype=int)
test_saletype_dummy = pd.get_dummies(test_df['SaleType'], prefix='saletype', drop_first=True, dtype=int)
train_df = pd.concat([train_df,saletype_dummy], axis=1)
test_df = pd.concat([test_df,test_saletype_dummy], axis=1)
train_df.drop(columns=['SaleType'], inplace=True)
test_df.drop(columns=['SaleType'], inplace=True)

# SaleCondition
salecondition_dummy = pd.get_dummies(train_df['SaleCondition'], prefix='salecondition', drop_first=True, dtype=int)
test_salecondition_dummy = pd.get_dummies(test_df['SaleCondition'], prefix='salecondition', drop_first=True, dtype=int)
train_df = pd.concat([train_df,salecondition_dummy], axis=1)
test_df = pd.concat([test_df,test_salecondition_dummy], axis=1)
train_df.drop(columns=['SaleCondition'], inplace=True)
test_df.drop(columns=['SaleCondition'], inplace=True)

# BldgType
bldngtype_map = {'1Fam': 'Single', '2fmCon': 'Double', 'Duplex': 'Double', 'TwnhsE': 'Townhouse', 'Twnhs': 'Townhouse'}
train_df.replace({'BldgType': bldngtype_map}, inplace=True)
test_df.replace({'BldgType': bldngtype_map}, inplace=True)
bldgtype_dummy = pd.get_dummies(train_df['BldgType'], prefix='bldgtype', drop_first=True, dtype=int)
test_bldgtype_dummy = pd.get_dummies(test_df['BldgType'], prefix='bldgtype', drop_first=True, dtype=int)
train_df = pd.concat([train_df,bldgtype_dummy], axis=1)
test_df = pd.concat([test_df,test_bldgtype_dummy], axis=1)
train_df.drop(columns=['BldgType'], inplace=True)
test_df.drop(columns=['BldgType'], inplace=True)


###############################################
# Handling ordinal values => First make them Categorical dtypes and then give them a number resepectively

# ExterQual
exterqual_map = {'Ex': 'Excellent', 'Gd':'Good', 'TA': 'Average', 'Fa':'Fair'}
train_df.replace({'ExterQual': exterqual_map}, inplace=True)
test_df.replace({'ExterQual': exterqual_map}, inplace=True)
external_quality_map = {'Average', 'Good', 'Fair', 'Excellent'}
train_df['ExterQual'] = pd.Categorical(train_df['ExterQual'], categories=external_quality_map, ordered=True)
test_df['ExterQual'] = pd.Categorical(test_df['ExterQual'], categories=external_quality_map, ordered=True)
ext_quality_mapping = {'Average': 1, 'Good': 2, 'Fair': 3, 'Excellent':4}
train_df['ExterQual'] = train_df['ExterQual'].map(ext_quality_mapping)
test_df['ExterQual'] = test_df['ExterQual'].map(ext_quality_mapping)

# BsmtQual
bsmtqual_map = {'Ex': 'Excellent', 'Gd':'Good', 'TA': 'Average', 'Fa':'Fair'}
train_df.replace({'BsmtQual': bsmtqual_map}, inplace=True)
test_df.replace({'BsmtQual': bsmtqual_map}, inplace=True)
bsmt_quality_map = ['Average', 'Good', 'Fair', 'Excellent']
train_df['BsmtQual'] = pd.Categorical(train_df['BsmtQual'], categories=bsmt_quality_map, ordered=True)
test_df['BsmtQual'] = pd.Categorical(test_df['BsmtQual'], categories=bsmt_quality_map, ordered=True)
bsmt_quality_mapping = {'Average': 1, 'Good': 2, 'Fair': 3, 'Excellent': 4}
train_df['BsmtQual'] = train_df['BsmtQual'].map(bsmt_quality_mapping)
test_df['BsmtQual'] = test_df['BsmtQual'].map(bsmt_quality_mapping)

# BsmtFinType1

bsmtfintype1_map = {'GLQ':'Good', 'ALQ': 'Average', 'Unf': 'Poor', 'Rec':'Average', 'BLQ': 'Poor', 'LwQ':'Poor'}
train_df.replace({'BsmtFinType1': bsmtfintype1_map}, inplace=True)
test_df.replace({'BsmtFinType1': bsmtfintype1_map}, inplace=True)
bsmnttype_map = ['Poor', 'Average', 'Good']
train_df['BsmtFinType1'] = pd.Categorical(train_df['BsmtFinType1'], categories=bsmnttype_map, ordered=True)
test_df['BsmtFinType1'] = pd.Categorical(test_df['BsmtFinType1'], categories=bsmnttype_map, ordered=True)
bsmt_finish_mapping = {'Poor': 1, 'Average': 2, 'Good': 3}
train_df['BsmtFinType1'] = train_df['BsmtFinType1'].map(bsmt_finish_mapping)
test_df['BsmtFinType1'] = test_df['BsmtFinType1'].map(bsmt_finish_mapping)

# HeatingQC
heating_map = {'Ex': 'Excellent', 'Gd':'Good', 'TA': 'Average', 'Fa':'Fair', 'Po':'Poor'}
train_df.replace({'HeatingQC': heating_map}, inplace=True)
test_df.replace({'HeatingQC': heating_map}, inplace=True)
heatqc_map = ['Poor', 'Average', 'Good', 'Fair', 'Excellent']
train_df['HeatingQC'] = pd.Categorical(train_df['HeatingQC'], categories=heatqc_map, ordered=True)
test_df['HeatingQC'] = pd.Categorical(test_df['HeatingQC'], categories=heatqc_map, ordered=True)
heatqc_mapping = {'Poor': 1, 'Average': 2, 'Good': 3, 'Fair': 4, 'Excellent':5}
train_df['HeatingQC'] = train_df['HeatingQC'].map(heatqc_mapping)
test_df['HeatingQC'] = test_df['HeatingQC'].map(heatqc_mapping)

# KitchenQual
kitchen_map = {'Ex': 'Excellent', 'Gd':'Good', 'TA': 'Average', 'Fa':'Fair'}
train_df.replace({'KitchenQual': kitchen_map}, inplace=True)
test_df.replace({'KitchenQual': kitchen_map}, inplace=True)
heatqc_map = ['Average', 'Good', 'Fair', 'Excellent']
train_df['KitchenQual'] = pd.Categorical(train_df['KitchenQual'], categories=heatqc_map, ordered=True)
test_df['KitchenQual'] = pd.Categorical(test_df['KitchenQual'], categories=heatqc_map, ordered=True)
kitchenqc_mapping = {'Average': 1, 'Good': 2, 'Fair': 3, 'Excellent':4}
train_df['KitchenQual'] = train_df['KitchenQual'].map(kitchenqc_mapping)
test_df['KitchenQual'] = test_df['KitchenQual'].map(kitchenqc_mapping)


# GarageQual
garagequal_map = {'Ex': 'Excellent', 'Gd':'Good', 'TA': 'Average', 'Fa':'Fair', 'Po':'Poor'}
train_df.replace({'GarageQual': garagequal_map}, inplace=True)
test_df.replace({'GarageQual': garagequal_map}, inplace=True)
garageq_map = ['Poor', 'Average', 'Good', 'Fair', 'Excellent']
train_df['GarageQual'] = pd.Categorical(train_df['GarageQual'], categories=garageq_map, ordered=True)
test_df['GarageQual'] = pd.Categorical(test_df['GarageQual'], categories=garageq_map, ordered=True)
garageq_mapping = {'Poor': 1, 'Average': 2, 'Good': 3, 'Fair': 4, 'Excellent':5}
train_df['GarageQual'] = train_df['GarageQual'].map(garageq_mapping)
test_df['GarageQual'] = test_df['GarageQual'].map(garageq_mapping)

# HouseStyle
housestyle_map = {'2.5Unf': 'Extra Large', '2Story':'Large', '1.5Fin': 'Medium', '1.5Unf':'Medium', '1Story':'Small', 'SLvl': 'Extra Small', 'SFoyer': 'Extra Small'}
train_df.replace({'HouseStyle': housestyle_map}, inplace=True)
test_df.replace({'HouseStyle': housestyle_map}, inplace=True)
house_map = ['Extra Small', 'Small', 'Medium', 'Large', 'Extra Large']
train_df['HouseStyle'] = pd.Categorical(train_df['HouseStyle'], categories=house_map, ordered=True)
test_df['HouseStyle'] = pd.Categorical(test_df['HouseStyle'], categories=house_map, ordered=True)
house_mapping = {'Extra Small': 1, 'Small': 2, 'Medium': 3, 'Large': 4, 'Extra Large':5}
train_df['HouseStyle'] = train_df['HouseStyle'].map(house_mapping)
test_df['HouseStyle'] = test_df['HouseStyle'].map(house_mapping)


#################################################################################################################
# Train the model using linear regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X = train_df.drop(columns=['SalePrice']).values
y = train_df['SalePrice'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)

reg = LinearRegression()
reg.fit(X_train, y_train)
train_pred_val = reg.predict(X_test)
rmse = mean_squared_error(train_pred_val, y_test)
print('RMSE values for Training set: ', rmse)
print('R2 score: ', r2_score(y_test, train_pred_val))
prediction_df = pd.DataFrame(columns=['Training_Predicted_values', 'Training_Actual_Values'])
prediction_df['Training_Predicted_values'] = train_pred_val
prediction_df['Training_Actual_Values'] = y_test
prediction_df['Training_Predicted_values'] = prediction_df['Training_Predicted_values'].astype(int)

# Ridge Regression
from sklearn.linear_model import Ridge
scores = []
for alpha in [0.1,1.0,10.0,100.0,1000.0]:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    ridge_pred = ridge.predict(X_test)
    scores.append(ridge.score(X_test, y_test))
print('Ridge Scores: ', scores)

import seaborn as sns
sns.regplot(data=prediction_df, x='Training_Predicted_values', y='Training_Actual_Values', color='blue')
plt.show()


test_df['electrical_Mix'] = train_df['electrical_Mix']
test_df = test_df.dropna()
#print(test_df.info(max_cols=125))
test_pred_val = reg.predict(test_df.values)
test_prediction_df = pd.DataFrame(columns=['Id', 'SalePrice'])
test_prediction_df['Id'] = test_df['Id']
test_prediction_df['SalePrice'] = test_pred_val

test_prediction_df['SalePrice'] = test_prediction_df['SalePrice'].astype(int)
test_prediction_df['Id'] = test_prediction_df['Id'].astype(int)
print(test_prediction_df.shape)

submission_csv = test_prediction_df.to_csv('test_house_prediction.csv', index=False)

