# housing-price
# Importing required libraries
import pandas as pd
import numpy as np

# Loading the excel file and replacing NaN with np.nan values
df= pd.read_excel("C://Users//ujjwala.gupta//Desktop//HSdataset.xlsx", na_values= 'nan')
df.isna().sum(axis=0)

# Deleting the features with minimum variablitity after examining their value counts
del df["Id"]
del df['Street']
del df["Alley"]
del df["Utilities"]


#Creating new variable representing the age of house, renovated house and garage at the time of sale
df["Ageatsale"]= df['YrSold'] - df['YearBuilt']
df["Ageatremod"]= df['YearRemodAdd'] -df['YearBuilt']

# Ordering the 77 features according to 4 broad categories constructed such as Physical charactertics, space, Location  and quality 
new_order=[0,2,4,11,12,15,16,77,78,18,17,19,20,21,25,35,37,38,43,44,45,46,47,48,50,52,54,55,56,57,61,70,71,72,73,1,3,22,30,32,34,33,39,40,41,42,58,62,63,64,65,66,67, 5,7,6,9,10, 13,14,23,24,26,27,28,29,31,36,49,51,53,60,59,68,69,74,75,76,8]
df = df[df.columns[new_order]]

df["Garageageatsale"]= df["YrSold"]-df["GarageYrBlt"]
#deleting repeated features which are captured in other features replicating them
del df["Condition2"]
del df["RoofMatl"]
del df["Exterior2nd"]

# Label endcoding keeping into account the order of categories in each feature specially quality features
# 0 is assigned if that house doesnot have that facility and for nan values ,1 is assigned to poor quality and moving on to 5 
#that is assigned for excellent quality implying as quality increases , labeling value used for categories increases
df["MSZoning"].value_counts()
MSZoning_cat={"RL": 1,"RM" : 2, "RH": 2, "FV" : 3, "C (all)" : 4}
df["MSZoning"]=df["MSZoning"].map(MSZoning_cat)

df["MSSubClass"].value_counts()
df["LotFrontage"].value_counts()
df["LotArea"].value_counts()
df["LotFrontage"].fillna(0, inplace=True)

df["LotShape"].value_counts()
LotShape_cat={"Reg": 1,"IR1" : 2, "IR2": 2, "IR3" : 2}
df["LotShape"]=df["LotShape"].map(LotShape_cat)

df["LandContour"].value_counts()
LandContour_cat={"Lvl": 1,"Bnk" : 2, "HLS": 3, "Low" : 4}
df["LandContour"]=df["LandContour"].map(LandContour_cat)

df["LotConfig"].value_counts()
LotConfig_cat={"Inside": 1,"Corner" : 2, "CulDSac": 3, "FR2" : 4, "FR3" : 4}
df["LotConfig"]=df["LotConfig"].map(LotConfig_cat)

df["LandSlope"].value_counts()
LandSlope_cat={"Gtl": 1,"Mod" : 2, "Sev": 3}
df["LandSlope"]=df["LandSlope"].map(LandSlope_cat)

df["Condition1"].value_counts()
Condition1_cat={"Norm": 1,"Feedr" : 2, "Artery": 3, "RRAn" : 4,"PosN" : 5, "RRAe" : 4, "PosA" : 5, "RRNn" : 4, "RRNe" : 4}
df["Condition1"]=df["Condition1"].map(Condition1_cat)
df["Condition1"].value_counts()

df["BldgType"].value_counts()
BldgType_cat={"1Fam": 1,"TwnhsE" : 2, "Duplex": 3, "Twnhs" : 4,"2fmCon" : 5}
df["BldgType"]=df["BldgType"].map(BldgType_cat)

df["HouseStyle"].value_counts()
HouseStyle_cat={"1Story": 1,"2Story" : 2, "1.5Fin": 3, "SLvl" : 4,"SFoyer" : 5, "1.5Unf" : 6, "2.5Unf" : 7, "2.5Fin" : 8}
df["HouseStyle"]=df["HouseStyle"].map(HouseStyle_cat)

df["RoofStyle"].value_counts()
RoofStyle_cat={"Gable": 1,"Hip" : 2, "Flat": 3, "Gambrel" : 4,"Mansard" : 5, "Shed" : 6}
df["RoofStyle"]=df["RoofStyle"].map(RoofStyle_cat)

df["Exterior1st"].value_counts()
Exterior1st_cat={"VinylSd": 1,"HdBoard" : 2, "MetalSd": 3, "Wd Sdng" : 4,"Plywood" : 5, "CemntBd" : 6, "BrkFace" : 7, "WdShing" : 8, "Stucco" : 9, "AsbShng" : 10, "Stone" : 11, "BrkComm" : 7, "AsphShn" : 10, "CBlock" : 12, "ImStucc" : 13}
df["Exterior1st"]=df["Exterior1st"].map(Exterior1st_cat)
df["Exterior1st"].value_counts()

df["MasVnrType"].value_counts()
MasVnrType_cat={"None": 0,"BrkFace" : 1, "Stone": 2, "BrkCmn" : 1}
df["MasVnrType"]=df["MasVnrType"].map(MasVnrType_cat)
sum(pd.isnull(df['MasVnrType']))
df["MasVnrType"].fillna(0, inplace=True)
df["MasVnrType"].value_counts()

df["MasVnrArea"].value_counts()
df["MasVnrArea"].fillna(0, inplace=True)
sum(pd.isnull(df['MasVnrArea']))

df["ExterQual"].value_counts()
sum(pd.isnull(df['ExterQual']))
ExterQual_cat={"TA": 3,"Gd" : 4, "Ex": 5, "Fa" : 2}
df["ExterQual"]=df["ExterQual"].map(ExterQual_cat)
df["ExterCond"].value_counts()

sum(pd.isnull(df['ExterCond']))
ExterCond_cat={"TA": 3,"Gd" : 4, "Ex": 5, "Fa" : 2, "Po" : 1}
df["ExterCond"]=df["ExterCond"].map(ExterCond_cat)
df["ExterCond"].value_counts()

df['PoolQC'].value_counts()
PoolQC_cat={np.nan:0,'Ex':5,'Gd':4,'Fa':2}
df['PoolQC']= df['PoolQC'].map(PoolQC_cat)
df['PoolQC'].value_counts()

df['GarageQual'].value_counts()
GarageQual_cat={np.nan:0,'Ex':5,'Gd':4,'Fa':2,'TA':3,'Po':1}
df['GarageQual']= df['GarageQual'].map(GarageQual_cat)
sum(pd.isnull(df['GarageQual']))

df['MiscFeature'].value_counts()
MiscFeature_cat={np.nan:0,'Elev':1,'Gar2':2,'Othr':3,'Shed':4,'TenC':5}
df['MiscFeature']= df['MiscFeature'].map(MiscFeature_cat)
sum(pd.isnull(df['MiscFeature']))

df["Foundation"].value_counts()
Foundation_cat={"CBlock": 2,"PConc" : 1, "BrkTil": 3, "Slab" : 4, "Stone" : 5, "Wood" : 6}
df["Foundation"]=df["Foundation"].map(Foundation_cat)
df["Foundation"].value_counts()

df['Fence'].value_counts()
Fence_cat={np.nan:0,'GdPrv':1,'MnPrv':2,'GdWo':3,'MnWw':4}
df['Fence']= df['Fence'].map(Fence_cat)
sum(pd.isnull(df['Fence']))

df['GarageCond'].value_counts()
GarageCond_cat={np.nan:0,'Ex':5,'Gd':4,'Fa':2,'TA':3,'Po':1}
df['GarageCond']= df['GarageCond'].map(GarageCond_cat)
sum(pd.isnull(df['GarageCond']))

df['KitchenQual'].value_counts()
KitchenQual_cat={np.nan:0,'Ex':5,'Gd':4,'TA':3,'Fa':2}
df['KitchenQual']= df['KitchenQual'].map(KitchenQual_cat)
sum(pd.isnull(df['KitchenQual']))

df['HeatingQC'].value_counts()
HeatingQC_cat={np.nan:0,'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1}
df['HeatingQC']= df['HeatingQC'].map(HeatingQC_cat)
sum(pd.isnull(df['HeatingQC']))

df['PavedDrive'].value_counts()
PavedDrive_cat={np.nan:0,'Y':1,'N':2,'P':3}
df['PavedDrive']= df['PavedDrive'].map(PavedDrive_cat)
sum(pd.isnull(df['PavedDrive']))

df['GarageFinish'].value_counts()
GarageFinish_cat={np.nan:0,'Unf':1,'Fin':3,'RFn':2}
df['GarageFinish']=df['GarageFinish'].map(GarageFinish_cat)
sum(pd.isnull(df['GarageFinish']))

df['FireplaceQu'].value_counts()
FireplaceQu_cat={np.nan:0,'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1}
df['FireplaceQu']= df['FireplaceQu'].map(FireplaceQu_cat)
sum(pd.isnull(df['FireplaceQu']))

df['CentralAir'].value_counts()
CentralAir_cat={np.nan:0,'Y':1,'N':2}
df['CentralAir']=df['CentralAir'].map(CentralAir_cat)
sum(pd.isnull(df['CentralAir']))

df['SaleCondition'].value_counts()
SaleCondition_cat={np.nan:0,'Normal':1,'Partial':2,'Abnorml':3,'Family':4,'Alloca':5, "AdjLand" :  6}
df['SaleCondition']= df['SaleCondition'].map(SaleCondition_cat)
sum(pd.isnull(df['SaleCondition']))

df['SaleType'].value_counts()
SaleType_cat={np.nan:0,'WD':1,'New':2,'COD':3,'ConLD':4,'ConLI':4, "ConLw" :  4, "CWD" : 1, "Oth" : 5, "Con" : 4}
df['SaleType']= df['SaleType'].map(SaleType_cat)
sum(pd.isnull(df['SaleType']))

df['GarageType'].value_counts()
GarageType_cat={np.nan:0,'Attchd':1,'Detchd':2,'BuiltIn':3, "Basment" : 4, "CarPort": 5, "2Types" : 6}
df['GarageType']=df['GarageType'].map(GarageType_cat)
sum(pd.isnull(df['GarageType']))

df['Functional'].value_counts()
Functional_cat={np.nan:0,'Typ':5,'Min1':4,'Min2':4, "Mod" : 3, "Maj1": 2, "Maj2" : 2, "Sev" : 1}
df['Functional']=df['Functional'].map(Functional_cat)
sum(pd.isnull(df['Functional']))
df["Functional"].fillna(0, inplace=True)
df["Functional"].value_counts()

df['Electrical'].value_counts()
Electrical_cat={np.nan:0,'SBrKr':1,'FuseA':2,'FuseF':3, "FuseP" : 4, "Mix": 5}
df['Electrical']=df['Electrical'].map(Electrical_cat)
sum(pd.isnull(df['Electrical']))
df["Electrical"].fillna(0, inplace=True)
df["Electrical"].value_counts()

df['Heating'].value_counts()
Heating_cat={np.nan:0,'GasA':1,'GasW':2,'Grav':3, "Wall" : 4, "OthW": 5, "Floor" : 6}
df['Heating']=df['Heating'].map(Heating_cat)
sum(pd.isnull(df['Heating']))

df['BsmtQual'].value_counts()
sum(pd.isnull(df['BsmtQual']))
BsmtQual_cat={np.nan:0,'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1}
df['BsmtQual']= df["BsmtQual"].map(BsmtQual_cat)
sum(pd.isnull(df['BsmtQual']))

df['BsmtCond'].value_counts()
sum(pd.isnull(df['BsmtCond']))
BsmtCond_cat={np.nan:0,'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1}
df['BsmtCond']= df["BsmtCond"].map(BsmtCond_cat)
sum(pd.isnull(df['BsmtCond']))

df['BsmtExposure'].value_counts()
sum(pd.isnull(df['BsmtExposure']))
BsmtExposure_cat={np.nan:0, "No" : 1, 'Mn':2, 'Av':3, 'Gd':4}
df['BsmtExposure']= df["BsmtExposure"].map(BsmtExposure_cat)
sum(pd.isnull(df['BsmtExposure']))

df['BsmtFinType1'].value_counts()
sum(pd.isnull(df['BsmtFinType1']))
BsmtFinType1_cat={np.nan:0, "Unf" : 1, 'LwQ':2, 'Rec':3, 'BLQ':4, "ALQ" : 5, "GLQ" : 6}
df['BsmtFinType1']= df["BsmtFinType1"].map(BsmtFinType1_cat)
sum(pd.isnull(df['BsmtFinType1']))

df['BsmtFinType2'].value_counts()
sum(pd.isnull(df['BsmtFinType2']))
BsmtFinType2_cat={np.nan:0, "Unf" : 1, 'LwQ':2, 'Rec':3, 'BLQ':4, "ALQ" : 5, "GLQ" : 6}
df['BsmtFinType2']= df["BsmtFinType2"].map(BsmtFinType2_cat)
sum(pd.isnull(df['BsmtFinType2']))

#Label encoding: Labelling each neighborhood the respective zipcode to reduce 25 categories (neighbourhood)  to just 2 category(zipcode)
df['Neighborhood'].value_counts()
Neighborhood_cat={np.nan:0, "NAmes" : 50010, 'CollgCr':50014, 'OldTown':50010, 'Edwards':50014, "Somerst" : 50010, "Gilbert" : 50014, "NridgHt" : 50010, "Sawyer": 50014 ,"NWAmes": 50014 , "SawyerW" : 50014 , "BrkSide" : 50010  , "Crawfor" : 50014, "Mitchel" : 50010  , "NoRidge" : 50010  , "Timber": 50014  , "IDOTRR": 50010  , "ClearCr":50014 , "StoneBr": 50010 ,"SWISU": 50011 , "Blmngtn": 50010   , "MeadowV": 50014  , "BrDale": 50010  , "Veenker": 50011  ,"NPkVill": 50010 , "Blueste": 50014 }
df['Neighborhood']= df["Neighborhood"].map(Neighborhood_cat)
sum(pd.isnull(df['Neighborhood']))

# Counting nan values for remaining columns and replacing them with 0
df.isna().sum(axis=0)
df["Exterior1st"].value_counts()
df["Exterior1st"].fillna(0, inplace=True)
df["BsmtFullBath"].fillna(0, inplace=True)
df["BsmtHalfBath"].fillna(0, inplace=True)
df["GarageYrBlt"].fillna(0, inplace=True)
df["Garageageatsale"].fillna(0, inplace=True)
df["GarageCars"].fillna(0, inplace=True)
df["MoSold"].fillna(0, inplace=True)
df["MSZoning"].fillna(0, inplace=True)
df["LotArea"].fillna(0, inplace=True)
df["MasVnrArea"].fillna(0, inplace=True)
df["LowQualFinSF"].fillna(0, inplace=True)
df["GrLivArea"].fillna(0, inplace=True)
df["GarageArea"].fillna(0, inplace=True)
df["WoodDeckSF"].fillna(0, inplace=True)
df["OpenPorchSF"].fillna(0, inplace=True)
df["1stFlrSF"].fillna(0, inplace=True)
df["2ndFlrSF"].fillna(0, inplace=True)
df["TotalBsmtSF"].fillna(0, inplace=True)
df["BsmtUnfSF"].fillna(0, inplace=True)
df["BsmtFinSF1"].fillna(0, inplace=True)
df["BsmtFinSF2"].fillna(0, inplace=True)
df["1stFlrSF"].fillna(0, inplace=True)
df["2ndFlrSF"].fillna(0, inplace=True)
df["TotalBsmtSF"].fillna(0, inplace=True)
df["BsmtUnfSF"].fillna(0, inplace=True)

# After all replacement of nan values, final count of nan values in whole file
df.isnull().values.any()
df.isnull().sum().sum()

##########################Exploratory data analysis####################################


# importing required libraries
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy import stats

# normal and log normal Distribution seaborn graph for Sale price( Dependent variable)
sns.set_style('darkgrid')
plt.figure(1); plt.title('Normal')
sns.distplot(df["SalePrice"], fit=stats.norm, kde=False)
sns.distplot(df["SalePrice"], fit=stats.norm, kde=True)
plt.figure(2); plt.title('Log Normal')
sns.distplot(df["SalePrice"], kde=False, fit=stats.lognorm)

#describing the dependent variable
df["SalePrice"].describe()
plt.hist(df['SalePrice'],orientation = 'vertical',histtype = 'bar', color ='blue')

# finding out outliers in saleprice by plotting box plot and then detecting the exact id to be transformed
sns.boxplot(df['SalePrice'])
z = np.abs(stats.zscore(df["SalePrice"]))
print(z)
threshold = 3
print(np.where(z > 3))

#Creation of quality index from all quality and condition defining features
x = df[df.columns[-22:-5]]
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
Quality= pd.DataFrame(x_scaled)
Quality["quality_index"]=  Quality.mean(axis=1)*100

#splitting the dataset into training and test dataset
from sklearn.model_selection import train_test_split
feature_names_df = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual',
       'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',
       'BsmtFinSF2', 'BsmtUnfSF', '1stFlrSF', '2ndFlrSF', "TotalBsmtSF",
       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
       'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF',
       'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
       'MiscVal', 'YrSold','MSZoning', 'LotShape', 'LandContour','LotConfig', 'LandSlope', 'Neighborhood', 'Condition1',
       'BldgType', 'HouseStyle', 'RoofStyle', 'Exterior1st','MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
       'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
       'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',
       'SaleType', 'SaleCondition','Garageageatsale', 'MoSold', 'Ageatsale', 'Ageatremod']
X_df = df[feature_names_df]
y_df = df['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, random_state=0)
