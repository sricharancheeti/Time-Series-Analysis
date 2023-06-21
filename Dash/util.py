import pymongo
import pandas as pd
import copy
from tqdm import tqdm
import numpy as np

# Replace the placeholders with your database credentials
# username = "Chashivmad"
# password = "GS5iR2Heom7qmz9q"
# hostname = "timeseries.zx4n7pp.mongodb.net"
# port = ""
database_name = "TimeSeries_prod"

# Create a connection string using the format mongodb://username:password@hostname:port/database_name
connection_string = f"mongodb+srv://Chashivmad:GS5iR2Heom7qmz9q@timeseries.zx4n7pp.mongodb.net/?retryWrites=true&w=majority"

# Connect to MongoDB using the connection string
client = pymongo.MongoClient(connection_string)

# Access a specific database and collection
db = client[database_name]

# update period for each ind (Y=Yearly, Q=Quarterly, M=Monthly, W=Weekly, D=Daily)
INDICATORS_PERIODS = {'GDP': 'Q', 'GDPC1': 'Q', 'GDPPOT': 'Q', 'NYGDPMKTPCDWLD': 'Y',                 # 1. Growth
                      # 2. Prices and Inflation
                      'CPIAUCSL': 'M', 'CPILFESL': 'M', 'GDPDEF': 'Q',
                      # 3. Money Supply
                      'M1SL': 'M', 'WM1NS': 'W', 'WM2NS': 'W', 'M1V': 'Q', 'M2V': 'Q', 'WALCL': 'W',
                      # 4. Employment
                      'UNRATE': 'M', 'NROU': 'Q', 'CIVPART': 'M', 'EMRATIO': 'M',
                      # 4. Employment
                      'UNEMPLOY': 'M', 'PAYEMS': 'M', 'MANEMP': 'M', 'ICSA': 'W', 'IC4WSA': 'W',
                      # 5. Income and Expenditure
                      'CDSP': 'Q', 'MDSP': 'Q', 'FODSP': 'Q', 'DSPIC96': 'M', 'PCE': 'M', 'PCEDG': 'M',
                      # 5. Income and Expenditure
                      'PSAVERT': 'M', 'DSPI': 'M', 'RSXFS': 'M',
                      # 6. Gov-t debt
                      'GFDEBTN': 'Q', 'GFDEGDQ188S': 'Q',
                      # 7. ETF
                      'VDE.US': 'D', 'VHT.US': 'D'
                      }

def get_macro_shift_transformation(macro_indicators_dict):
        """Add shifted (growth) values to the data_repo.macro_indicators before joining them together, remove non-stationary time series"""
        
        # Define historical periods in days
        HISTORICAL_PERIODS_DAYS = [1, 3, 7, 30, 90, 365]
        
        # Different types of transformations for daily, weekly, monthly, indicators
        DoD_ind = [k for (k, v) in INDICATORS_PERIODS.items() if v == 'D']
        WoW_ind = [k for (k, v) in INDICATORS_PERIODS.items() if v == 'W']
        MoM_ind = [k for (k, v) in INDICATORS_PERIODS.items() if v == 'M']
        QoQ_ind = [k for (k, v) in INDICATORS_PERIODS.items() if v == 'Q']
        YoY_ind = [k for (k, v) in INDICATORS_PERIODS.items() if v == 'Y']  
       
        # Process indexes (VHT.US and VDE.US)
        macro_indicators_dict['VHT.US'].drop(
            ['Open', 'High', 'Low', 'Volume'], axis=1, inplace=True)
        macro_indicators_dict['VHT.US'].rename(
            columns={'Close': 'VHT.US'}, inplace=True)
        macro_indicators_dict['VDE.US'].drop(
            ['Open', 'High', 'Low', 'Volume'], axis=1, inplace=True)
        macro_indicators_dict['VDE.US'].rename(
            columns={'Close': 'VDE.US'}, inplace=True)

        # Add shifted (growth) values for daily indicators
        for ind in DoD_ind:
           # Only perform transformation for specified indicators
            if not ind in {'VDE.US','VHT.US'}:
                continue
            for i in HISTORICAL_PERIODS_DAYS:
                df = macro_indicators_dict[ind]
                ind_transformed = ind + '_growth_' + str(i)+'d'
                df[ind_transformed] = df[ind]/df[ind].shift(i)-1

        # Add future growth stats for VHT.US and VDE.US
        for ind in ['VHT.US','VDE.US']:
          for i in HISTORICAL_PERIODS_DAYS:
            df = macro_indicators_dict[ind]
            ind_transformed = ind + '_future_growth_' + str(i)+'d'
            df[ind_transformed] = df[ind].shift(-i-1)/df[ind].shift(-1)-1

        # Add week-over-week and month-over-month growth stats for weekly indicators
        for ind in WoW_ind:
            df = macro_indicators_dict[ind]
            ind_transformed = ind + '_wow'
            df[ind_transformed] = df[ind]/df[ind].shift(1)-1
            ind_transformed = ind + '_mom'
            df[ind_transformed] = df[ind]/df[ind].shift(5)-1
            # Drop original "ind" column if series are non-stationary
            # Do not drop original ind for FinStressIndex (STLFSI2), and Long-term Mortgage rates
            if not ind in {'STLFSI2', 'MORTGAGE30US'}:
                macro_indicators_dict[ind].drop(
                    [ind], axis=1, inplace=True)

        # Add month-over-month and year-over-year growth stats for monthly indicators
        for ind in MoM_ind:
            df = macro_indicators_dict[ind]
            ind_transformed = ind + '_mom'
            df[ind_transformed] = df[ind]/df[ind].shift(1)-1
            ind_transformed = ind + '_yoy'
            df[ind_transformed] = df[ind]/df[ind].shift(12)-1
            # drop original "ind" column if series are non-stationary
            # do not drop original ind for all indicators that are 'ratios' or 'percentages'
            if not ind in {'UNRATE', 'CIVPART', 'EMRATIO', 'PSAVERT', 'INDPRO', 'TCU', 'SPCS20RSA', 'MULTPL_SHILLER_PE_RATIO_MONTH'}:
                macro_indicators_dict[ind].drop(
                    [ind], axis=1, inplace=True)

        # Loop through each indicator in the QoQ_ind list
        for ind in QoQ_ind:
            # Get the DataFrame for the current indicator from the dictionary of macro indicators
            df = macro_indicators_dict[ind]
            # Create a new column name by appending '_qoq' to the original indicator name
            ind_transformed = ind + '_qoq'
            # Calculate the quarter-over-quarter (QoQ) growth rate by dividing the current value by the value from the previous quarter, and subtracting 1
            df[ind_transformed] = df[ind] / df[ind].shift(1) - 1
            # Create another new column name by appending '_yoy' to the original indicator name
            ind_transformed = ind + '_yoy'
            # Calculate the year-over-year (YoY) growth rate by dividing the current value by the value from the same quarter in the previous year, and subtracting 1
            df[ind_transformed] = df[ind] / df[ind].shift(4) - 1
            # Drop the original "ind" column from the DataFrame if the series are non-stationary, except for indicators that are 'ratios' or 'percentages'
            if not ind in {'GDPDEF', 'M1V', 'M2V', 'NROU', 'CDSP', 'MDSP', 'FODSP', 'GFDEGDQ188S'}:
                macro_indicators_dict[ind].drop([ind], axis=1, inplace=True)

        # Loop through each indicator in the YoY_ind list
        for ind in YoY_ind:
            # Get the DataFrame for the current indicator from the dictionary of macro indicators
            df = macro_indicators_dict[ind]
            # Create a new column name by appending '_yoy' to the original indicator name
            ind_transformed = ind + '_yoy'
            # Calculate the year-over-year (YoY) growth rate by dividing the current value by the value from the same quarter in the previous year, and subtracting 1
            df[ind_transformed] = df[ind] / df[ind].shift(1) - 1
            # Drop the original "ind" column from the DataFrame
            macro_indicators_dict[ind].drop([ind], axis=1, inplace=True)

# As we have data that ranges over various periods such as Month, Quarter and Year
# We are making the 
def get_daily_macro_stats_df(daily_df, macro_ind_df, regime='LAST'):
        """take Time from daily_df, and apply that to macro_ind_df, (LAST=take last observation, PREVIOUS=take previous) """
        # 
        ticker_dates = daily_df.Date.sort_values().unique()
        new_column_dict = {}

        for elem in ticker_dates:
            ts = pd.to_datetime(str(elem))
            d = ts.strftime('%Y-%m-%d')
            # all potential records to merge
            options_to_merge = macro_ind_df[macro_ind_df.index <= d]
            if len(options_to_merge) == 0:
                continue
            last_value = options_to_merge.tail(1).values.tolist()[0]
            prev_value = options_to_merge.tail(2).values.tolist()[0]
            if regime == 'PREVIOUS':
                if prev_value is not None:
                    new_column_dict[d] = prev_value
            elif regime == 'LAST':
                if last_value is not None:
                    new_column_dict[d] = last_value
            else:
                raise("Regime should be in ('PREVIOUS','LAST')")
        return pd.DataFrame.from_dict(new_column_dict, orient='index', columns = options_to_merge.keys())

collec = db.list_collection_names()
macro_indicators = dict()
for index in collec:
    # print(index)
    collection = db[index]
    df = pd.DataFrame(list(collection.find({},{"_id":0})))
    if index in ['VDE.US','VHT.US']:
        pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        # df.index = pd.to_datetime(df['Date'].index)
    else:
        pd.to_datetime(df['DATE'])
        df = df.set_index('DATE')
        # print(df)
        # df.index = pd.to_datetime(df['DATE'].index)
    macro_indicators[index] = df

# need to have a deep copy of macro indicators to make local transformations not changing the datarepo
macro_indicators_dict = copy.deepcopy(macro_indicators)

# Transform the macroindicators
get_macro_shift_transformation(macro_indicators_dict)

# Extract the dates from VDE.US dataframe
dates = pd.DataFrame(macro_indicators_dict['VDE.US'].index.sort_values().unique(), columns=['Date'])

# 2) Create a DAILY macro stats dataset
# Different types of joins for daily, weekly, monthly, indicators
  # join on the last available date
lastday_ind = [k for (k, v) in INDICATORS_PERIODS.items() if v not in ('M', 'Q', 'Y')]
  # join on the previous available date (M,Q,Y stats write say '2021-01-01' - but they represent the whole M,Q,Y STARTING at this date)
firstday_ind = [k for (k, v) in INDICATORS_PERIODS.items() if v in ('M', 'Q', 'Y')]

  # start from all dates we need to have macro stats
dates = pd.DataFrame(macro_indicators_dict['VDE.US'].index.sort_values().unique(), columns=['Date'])

macro_data_df = None
#  iterate over all transformed series in self.macro_indicators_dict and join one by one
# all non-Monthly indicators are
print(lastday_ind)
tq_last_day = tqdm(lastday_ind)
tq_last_day.set_description("Merging LAST Day indicators")
for ind in tq_last_day:
  df_to_merge = get_daily_macro_stats_df(dates, macro_indicators_dict[ind], regime='LAST')
  if macro_data_df is None:
    macro_data_df = df_to_merge
  else:
    macro_data_df = macro_data_df.join(df_to_merge)

tq_first_day = tqdm(firstday_ind)
tq_first_day.set_description("Merging FIRST Day indicators")
#  some stats  have first day of period date (e.g. '2020-06-01' instead of '2020-06-30'), so we need to get PREVIOUS available macro param
for ind in tq_first_day:
  df_to_merge = get_daily_macro_stats_df(dates, macro_indicators_dict[ind], regime='PREVIOUS')
  if macro_data_df is None:
    macro_data_df = df_to_merge
  else:
    macro_data_df = macro_data_df.join(df_to_merge)

# Future growth indicators are mostly correlated with each other
future_ind = []
for ind in macro_data_df.keys():
  if 'future' in ind:
    future_ind.append(ind)
  
# print(future_ind)

# include all features 
X_keys = macro_data_df.keys()
# do not use future ind to predict
X_keys = X_keys.drop(future_ind)

# deep copy of the dataframe not to change the original df
macro_copy = macro_data_df.copy(deep=True)

# replace bad values with np.nan
macro_copy.replace([np.inf, -np.inf], np.nan, inplace=True)

macro_copy.fillna(0,inplace=True)

X = macro_copy[X_keys]
y = macro_copy['VDE.US_future_growth_90d']

# Calculate daily returns for ETFs
returns_vde = macro_copy["VDE.US"].pct_change().dropna()
returns_vht = macro_copy["VHT.US"].pct_change().dropna()

# Combine returns into one dataframe
df_returns = pd.concat([returns_vde, returns_vht], axis=1)
df_returns.columns = ["VDE Returns", "VHT Returns"]

# Calculate daily changes for macroeconomic data
df_macro_changes = macro_copy.diff().dropna()
# Merge macroeconomic data and ETF returns
df_merged = pd.concat([df_returns, df_macro_changes],axis=1)

from sklearn.linear_model import LinearRegression

# Separate the XLV and XLE data
X_vde = df_merged.iloc[:, 2:]
y_vde = df_merged.iloc[:, 0]

X_vht = df_merged.iloc[:, 2:]
y_vht = df_merged.iloc[:, 1]

# Create the linear regression models
model_vde = LinearRegression().fit(X_vde, y_vde)
model_vht = LinearRegression().fit(X_vht, y_vht)

# Calculate feature importance for XLV
feature_importance_xlv = pd.DataFrame(model_vde.coef_, index=X_vde.columns, columns=["Importance"])
feature_importance_xlv = feature_importance_xlv.abs().sort_values(by="Importance", ascending=False)
# print("VDE Feature Importance:\n", feature_importance_xlv)

# Calculate feature importance for XLE
feature_importance_xle = pd.DataFrame(model_vht.coef_, index=X_vht.columns, columns=["Importance"])
feature_importance_xle = feature_importance_xle.abs().sort_values(by="Importance", ascending=False)
# print("VHT Feature Importance:\n", feature_importance_xle)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

from sklearn.ensemble import RandomForestRegressor

clf = RandomForestRegressor(max_depth=8, random_state=0)
clf.fit(X_train, y_train)

y_pred_clf = clf.predict(X_test)
df2 = pd.DataFrame({'Actual': y_test, 'Predicted':y_pred_clf})
df2.reset_index(inplace=True)
data_list_ins = df2.to_dict(orient='records')
# temp = []
collection = db.create_collection("VDE.US_PRED")
collection.insert_many(data_list_ins)

# Classification for VHT
X_VHT = macro_copy[X_keys]
y_vht = macro_copy['VHT.US_future_growth_90d']


from sklearn.model_selection import train_test_split

X_train_vht, X_test_vht, y_train_vht, y_test_vht = train_test_split(X_VHT, y_vht, test_size=0.2, shuffle=False)
from sklearn.ensemble import RandomForestRegressor

clf = RandomForestRegressor(max_depth=8, random_state=0)
clf.fit(X_train_vht, y_train_vht)

y_pred_clf_vht = clf.predict(X_test)
df3 = pd.DataFrame({'Actual': y_test_vht, 'Predicted':y_pred_clf_vht})
df3.reset_index(inplace=True)
data_list_ins = df3.to_dict(orient='records')
# temp = []
collection = db.create_collection("VHT.US_PRED")
collection.insert_many(data_list_ins)

# # Random Forest regressor : Actual vs. Predicted graph

# import matplotlib.ticker as mtick

# ax = df2.plot(figsize=(20,6), grid=True)

# ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))