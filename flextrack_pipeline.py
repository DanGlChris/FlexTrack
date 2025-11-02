import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import confusion_matrix, mean_absolute_error, f1_score, mean_squared_error
from sklearn.isotonic import IsotonicRegression
import warnings
import os
from scipy.stats import linregress
from xgboost import XGBClassifier, XGBRegressor
warnings.filterwarnings('ignore')

# --- 0. Configuration & Setup ---
USE_CALIBRATION = False
RUN_GRID_SEARCH = False 
# --- Data Loading ---
print("Loading data...")
try:
    train_df = pd.read_csv('/kaggle/input/flextrack/flextrack-2025-training-data-v0.2.csv', encoding="utf-8")
    test_df = pd.read_csv('/kaggle/input/flextrack/flextrack-2025-public-test-data-v0.3.csv', encoding="utf-8") 
    sample_submission_df = pd.read_csv('/kaggle/input/flextrack/flextrack-2025-random-prediction-data-v0.2.csv', encoding="utf-8")
except FileNotFoundError:
    print("Kaggle paths not found. Falling back to local paths.")
    train_df = pd.read_csv('data/flextrack-2025-training-data-v0.2.csv')
    test_df = pd.read_csv('data/flextrack-2025-public-test-data-v0.3.csv')
    sample_submission_df = pd.read_csv('data/flextrack-2025-random-prediction-data-v0.2.csv')

# --- Column Definitions ---
COL_SITE = 'Site'
COL_TIME = 'Timestamp_Local'
COL_TEMP = 'Dry_Bulb_Temperature_C'
COL_RAD = 'Global_Horizontal_Radiation_W/m2'
COL_POWER = 'Building_Power_kW'
COL_FLAG = 'Demand_Response_Flag'
COL_CAP = 'Demand_Response_Capacity_kW'


# --- 1. Feature Engineering & Selection Functions ---

def calculate_rolling_slope(series):
    x = np.arange(len(series))
    valid_indices = ~np.isnan(series)
    if np.sum(valid_indices) < 2: return np.nan
    slope, _, _, _, _ = linregress(x[valid_indices], series[valid_indices])
    return slope

# Function to dynamically determine site archetypes
def get_site_archetypes(df):
    """
    Analyzes the full dataset to classify sites into archetypes based on power consumption.
    """
    # Calculate the mean power consumption for each site
    site_power_stats = df.groupby(COL_SITE)[COL_POWER].mean()
    
    # Define a threshold to separate small and large consumers
    # A threshold of 100 kW seems appropriate based on your data description.
    power_threshold = 60 
    
    archetype_map = {
        site: 'large' if power > power_threshold else 'small'
        for site, power in site_power_stats.items()
    }
    print("Generated Site Archetypes:", archetype_map)
    return archetype_map

SITE_ARCHETYPE_MAP = get_site_archetypes(test_df)

def create_features(df, baseline=False):
    """
    MODIFIED: Now uses the dynamic SITE_ARCHETYPE_MAP for all sites.
    """
    df[COL_TIME] = pd.to_datetime(df[COL_TIME])
    df = df.set_index(COL_TIME)
    
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['dayofyear'] = df.index.dayofyear
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
    
    lags = [1, 2, 4, 8, 16, 24, 48, 56, 96, 144]
    for lag in lags:
        df[f'power_lag_{lag}'] = df.groupby(COL_SITE)[COL_POWER].shift(lag)
        df[f'temp_lag_{lag}'] = df.groupby(COL_SITE)[COL_TEMP].shift(lag)
        
    windows = [4, 12, 24, 48, 56, 96, 144]
    for window in windows:
        df[f'power_roll_mean_{window}'] = df.groupby(COL_SITE)[COL_POWER].transform(lambda x: x.rolling(window).mean())
        df[f'power_roll_std_{window}'] = df.groupby(COL_SITE)[COL_POWER].transform(lambda x: x.rolling(window).std())
        
    comfort_temp = 18.0
    df['heating_degree'] = np.maximum(0, comfort_temp - df[COL_TEMP])
    df['cooling_degree'] = np.maximum(0, df[COL_TEMP] - comfort_temp)
    
    df['temp_rad_interaction'] = df[COL_TEMP] * df[COL_RAD]
    df['temp_squared'] = df[COL_TEMP] ** 2
    df['power_diff_1'] = df.groupby(COL_SITE)[COL_POWER].diff(1)
    df['power_cumsum_4'] = df.groupby(COL_SITE)[COL_POWER].transform(lambda x: x.rolling(4).sum())
    
    df['is_business_hours'] = ((df['hour'] >= 7) & (df['hour'] <= 18) & (df['is_weekend'] == 0)).astype(int)
    df['is_peak_hours'] = ((df['hour'] >= 13) & (df['hour'] <= 20)).astype(int)
    df['is_winter'] = df['month'].isin([6, 7, 8]).astype(int)
    df['is_summer'] = df['month'].isin([12, 1, 2]).astype(int)
    df['is_heating_likely'] = ((df['is_winter'] == 1) & ((df['is_weekend'] == 1) | (df['heating_degree'] > 0))).astype(int)
    
  
    df['site_archetype'] = df[COL_SITE].map(SITE_ARCHETYPE_MAP)
    all_archetypes = ['small', 'large'] 
    df['site_archetype'] = pd.Categorical(df['site_archetype'], categories=all_archetypes)
    df = pd.get_dummies(df, columns=['site_archetype'], prefix='archetype')
    
    for archetype in ['archetype_small', 'archetype_large']:
        if archetype in df.columns:
            df[f'{archetype}_temp_interaction'] = df[archetype] * df[COL_TEMP]
            df[f'{archetype}_power_interaction'] = df[archetype] * df[COL_POWER]
   
    df['power_diff_96'] = df.groupby(COL_SITE)[COL_POWER].diff(96)
    df['temp_diff_96'] = df.groupby(COL_SITE)[COL_TEMP].diff(96)
    df['power_vs_roll_mean_96'] = df[COL_POWER] - df[f'power_roll_mean_96']

    if(not baseline):
        df['day_of_year_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365)
        df['is_lunch_hour'] = ((df['hour'] >= 12) & (df['hour'] < 13) & (df['is_weekend'] == 0)).astype(int)
        df['is_overnight'] = ((df['hour'] >= 22) | (df['hour'] < 5)).astype(int)
        df['temp_deviation_from_comfort'] = np.abs(df[COL_TEMP] - 18.0)
        df['temp_bins'] = pd.cut(df[COL_TEMP], bins=[-np.inf, 10, 18, 25, np.inf], labels=['cold', 'mild', 'warm', 'hot'])
        for span in [8, 24]:
            df[f'power_ewm_span_{span}'] = df.groupby(COL_SITE)[COL_POWER].transform(lambda x: x.ewm(span=span, adjust=False).mean())
            df[f'temp_ewm_span_{span}'] = df.groupby(COL_SITE)[COL_TEMP].transform(lambda x: x.ewm(span=span, adjust=False).mean())
        df[f'power_slope_4'] = df.groupby(COL_SITE)[COL_POWER].transform(lambda x: x.rolling(window=4, min_periods=2).apply(calculate_rolling_slope, raw=False))
        df['temp_x_hour_sin'] = df[COL_TEMP] * np.sin(2 * np.pi * df['hour'] / 24)
        df['temp_x_hour_cos'] = df[COL_TEMP] * np.cos(2 * np.pi * df['hour'] / 24)
        df['temp_x_power_lag_96'] = df[COL_TEMP] * df['power_lag_96']
        df = pd.get_dummies(df, columns=['temp_bins'], prefix='temp_bins')
        df['day_of_month'] = df.index.day
        df['week_of_year'] = df.index.isocalendar().week.astype(int)
        df['time_of_day_minute'] = df.index.hour * 60 + df.index.minute
        df['time_of_day_fraction'] = df['time_of_day_minute'] / 1440.0
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['is_weekday'] = (df['dayofweek'] < 5).astype(int)
        df['is_work_hours'] = ((df['hour'] >= 8) & (df['hour'] < 18) & (df['is_weekend'] == 0)).astype(int)
        df['is_morning_ramp_up'] = ((df['hour'] >= 6) & (df['hour'] < 9) & (df['is_weekend'] == 0)).astype(int)
        df['is_evening_ramp_down'] = ((df['hour'] >= 17) & (df['hour'] < 20) & (df['is_weekend'] == 0)).astype(int)
        df['hour_of_week'] = df['dayofweek'] * 24 + df['hour']
        df['is_first_work_hour'] = ((df['hour'] == 8) & (df['is_weekend'] == 0)).astype(int)
        df['is_last_work_hour'] = ((df['hour'] == 17) & (df['is_weekend'] == 0)).astype(int)
        df['season'] = (df['month'] % 12 + 3) // 3
        conditions = [ (df['is_work_hours'] == 1), ((df['is_weekday'] == 1) & (df['is_work_hours'] == 0)), ((df['is_weekend'] == 1) & (df['hour'].between(8, 18))), ((df['is_weekend'] == 1) & ~(df['hour'].between(8, 18))) ]
        choices = ['Workday_Hours', 'Workday_OffHours', 'Weekend_Day', 'Weekend_Night']
        df['occupancy_state'] = np.select(conditions, choices, default='Unknown')
        df['temp_cubed'] = df[COL_TEMP] ** 3
        df['radiation_is_zero'] = (df[COL_RAD] == 0).astype(int)
        df['radiation_sqrt'] = np.sqrt(df[COL_RAD])
        df['is_mild_temp_workday'] = ((df[COL_TEMP] > 19) & (df[COL_TEMP] < 23) & (df['is_work_hours'] == 1)).astype(int)
        df['is_extreme_heat_workday'] = ((df[COL_TEMP] > 30) & (df['is_work_hours'] == 1)).astype(int)
        df['is_extreme_cold_workday'] = ((df[COL_TEMP] < 10) & (df['is_work_hours'] == 1)).astype(int)
        for window in [4, 8, 12, 24, 48, 96, 288, 672]:
            df[f'temp_roll_mean_{window}'] = df.groupby(COL_SITE)[COL_TEMP].transform(lambda x: x.rolling(window, min_periods=min(window//4, 1)).mean())
            df[f'temp_roll_std_{window}'] = df.groupby(COL_SITE)[COL_TEMP].transform(lambda x: x.rolling(window, min_periods=min(window//4, 1)).std())
        for window in [24, 96]:
            df[f'temp_roll_min_{window}'] = df.groupby(COL_SITE)[COL_TEMP].transform(lambda x: x.rolling(window, min_periods=1).min())
            df[f'temp_roll_max_{window}'] = df.groupby(COL_SITE)[COL_TEMP].transform(lambda x: x.rolling(window, min_periods=1).max())
            df[f'temp_roll_range_{window}'] = df[f'temp_roll_max_{window}'] - df[f'temp_roll_min_{window}']
        for diff_period in [1, 4, 24, 96]:
            if f'temp_diff_{diff_period}' not in df.columns: df[f'temp_diff_{diff_period}'] = df.groupby(COL_SITE)[COL_TEMP].diff(diff_period)
            df[f'rad_diff_{diff_period}'] = df.groupby(COL_SITE)[COL_RAD].diff(diff_period)
        for window in [4, 12, 24, 48, 96, 144]:
            df[f'power_roll_min_{window}'] = df.groupby(COL_SITE)[COL_POWER].transform(lambda x: x.rolling(window, min_periods=1).min())
            df[f'power_roll_max_{window}'] = df.groupby(COL_SITE)[COL_POWER].transform(lambda x: x.rolling(window, min_periods=1).max())
            df[f'power_roll_median_{window}'] = df.groupby(COL_SITE)[COL_POWER].transform(lambda x: x.rolling(window, min_periods=1).median())
        for window in [24, 96]:
            df[f'power_roll_skew_{window}'] = df.groupby(COL_SITE)[COL_POWER].transform(lambda x: x.rolling(window, min_periods=4).skew())
            df[f'power_roll_kurt_{window}'] = df.groupby(COL_SITE)[COL_POWER].transform(lambda x: x.rolling(window, min_periods=4).kurt())
        for diff_period in [1, 4, 24, 96, 672]:
            if f'power_diff_{diff_period}' not in df.columns: df[f'power_diff_{diff_period}'] = df.groupby(COL_SITE)[COL_POWER].diff(diff_period)
            df[f'power_pct_change_{diff_period}'] = df.groupby(COL_SITE)[COL_POWER].pct_change(diff_period)
        df['occupancy_state_lag'] = df.groupby(COL_SITE)['occupancy_state'].shift(1)
        df['state_changed'] = (df['occupancy_state'] != df['occupancy_state_lag']).astype(int)
        df['is_work_hours_lag'] = df.groupby(COL_SITE)['is_work_hours'].shift(1)
        df['is_transition_to_work'] = ((df['is_work_hours'] == 1) & (df['is_work_hours_lag'] == 0)).astype(int)
        df['is_transition_from_work'] = ((df['is_work_hours'] == 0) & (df['is_work_hours_lag'] == 1)).astype(int)
        df['time_since_last_transition'] = df.groupby([COL_SITE, df['state_changed'].cumsum()]).cumcount()
        df = pd.get_dummies(df, columns=['occupancy_state'], prefix='state')
        df = df.drop(columns=['occupancy_state_lag'])
        df['temp_x_is_work_hours'] = df[COL_TEMP] * df['is_work_hours']
        df['temp_x_is_weekend'] = df[COL_TEMP] * df['is_weekend']
        df['rad_x_is_work_hours'] = df[COL_RAD] * df['is_work_hours']
        df['rad_x_is_weekend'] = df[COL_RAD] * df['is_weekend']
        df['rad_x_hour_sin'] = df[COL_RAD] * df['hour_sin']
        df['rad_x_hour_cos'] = df[COL_RAD] * df['hour_cos']
        df['effective_temp_load'] = df[COL_TEMP] + 0.01 * df[COL_RAD]
        df['temp_x_season'] = df[COL_TEMP] * df['season']
        df['temp_x_month_sin'] = df[COL_TEMP] * df['month_sin']
        df['temp_x_month_cos'] = df[COL_TEMP] * df['month_cos']
        df['hour_x_power_roll_mean_24'] = df['hour'] * df['power_roll_mean_24']
        df['power_lag_96_x_dayofweek'] = df['power_lag_96'] * df['dayofweek']
        df['rad_x_power_lag_96'] = df[COL_RAD] * df['power_lag_96']
        df['temp_x_power_roll_mean_12'] = df[COL_TEMP] * df['power_roll_mean_12']
        for window in [12, 24, 48]:
            df[f'power_roll_sum_{window}'] = df.groupby(COL_SITE)[COL_POWER].transform(lambda x: x.rolling(window, min_periods=1).sum())
            df[f'temp_roll_sum_{window}'] = df.groupby(COL_SITE)[COL_TEMP].transform(lambda x: x.rolling(window, min_periods=1).sum())
            df[f'rad_roll_sum_{window}'] = df.groupby(COL_SITE)[COL_RAD].transform(lambda x: x.rolling(window, min_periods=1).sum())
        if 'power_roll_mean_96' in df.columns and 'power_roll_max_96' in df.columns:
            df['power_vs_daily_avg'] = df[COL_POWER] / (df['power_roll_mean_96'] + 1e-6)
            df['power_vs_daily_max'] = df[COL_POWER] / (df['power_roll_max_96'] + 1e-6)
        for window in [24, 96]:
            mean_col = f'power_roll_mean_{window}'
            std_col = f'power_roll_std_{window}'
            if mean_col in df.columns and std_col in df.columns:
                df[f'power_zscore_{window}'] = (df[COL_POWER] - df[mean_col]) / (df[std_col] + 1e-6)
        for window in [96]:
            df[f'power_percentile_{window}'] = df.groupby(COL_SITE)[COL_POWER].transform(
                lambda x: x.rolling(window, min_periods=4).apply(
                    lambda y: (y.iloc[-1] <= y).mean() if len(y) > 0 else np.nan, raw=False))
            
    return df.reset_index()

def generate_and_select_features(train_df, test_df):
    print("  - Stage 1: Creating baseline and full feature sets...")
    train_df_baseline, test_df_baseline = create_features(train_df.copy(), baseline=True), create_features(test_df.copy(), baseline=True)
    train_df_full, test_df_full = create_features(train_df.copy(), baseline=False), create_features(test_df.copy(), baseline=False)
    train_cols = train_df_full.columns
    test_df_full, test_df_baseline = test_df_full.reindex(columns=train_cols, fill_value=0), test_df_baseline.reindex(columns=train_df_baseline.columns, fill_value=0)

    print("  - Stage 2: Selecting stable baseline features via correlation...")
    features_to_exclude = [COL_SITE, COL_TIME, COL_FLAG, COL_CAP, 'demand_response_flag_mapped']
    baseline_features_raw = [col for col in train_df_baseline.columns if col not in features_to_exclude]
    corr_matrix = train_df_baseline[baseline_features_raw].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop_corr = [column for column in upper.columns if any(upper[column] > 0.98)]
    stable_baseline_features = [f for f in baseline_features_raw if (f not in to_drop_corr) and (f not in ['power_roll_std_56', 'power_roll_mean_144'])]
    print(f"    Stable baseline feature count: {len(stable_baseline_features)}")

    print("  - Stage 3: Building specialized feature lists for models...")
    CLASSIFIER_FEATURES_TO_REMOVE = ['month', 'power_roll_mean_96', 'power_roll_std_56', 'power_roll_mean_144', 'power_lag_96', 'temp_lag_48', 'power_roll_std_4', 'temp_lag_144', 'archetype_large', 'archetype_small']
    CLASSIFIER_FEATURES_TO_ADD = ['hour_cos', 'day_of_week_cos', 'is_last_work_hour', 'is_mild_temp_workday', 'temp_roll_min_24', 'rad_diff_1', 'rad_diff_4', 'is_transition_to_work', 'is_transition_from_work', 'state_Workday_OffHours', 'rad_roll_sum_48', 'power_zscore_96']
    
    REGRESSOR_FEATURES_TO_REMOVE = ["archetype_small_power_interaction", "Dry_Bulb_Temperature_C", 'power_roll_std_56', 'power_roll_mean_144']
    REGRESSOR_FEATURES_TO_ADD = ['temp_ewm_span_24', 'temp_x_power_lag_96', 'temp_x_hour_sin', 'archetype_large_power_interaction', 'power_ewm_span_8', 'temp_bins_warm', 'temp_roll_std_8', 'temp_roll_min_24', 'temp_diff_24', 'power_pct_change_4', 'temp_x_is_weekend', 'rad_x_hour_sin', 'power_vs_daily_max', 'temp_roll_std_4', 'rad_diff_4']
    
    features_clf = [f for f in stable_baseline_features if f not in CLASSIFIER_FEATURES_TO_REMOVE]
    for f in (CLASSIFIER_FEATURES_TO_ADD):
        if f not in features_clf and f in train_df_full.columns: features_clf.append(f)
    features_reg = [f for f in stable_baseline_features if f not in REGRESSOR_FEATURES_TO_REMOVE]
    for f in (REGRESSOR_FEATURES_TO_ADD):
        if f not in features_reg and f in train_df_full.columns: features_reg.append(f)
    return train_df_full, test_df_full, features_clf, features_reg


# --- 2. Modular Training & Prediction Pipeline ---
# This function remains largely the same, as it was already well-structured.
def train_and_predict_pipeline(train_df, test_df, features_clf, features_reg, run_grid_search=False, params=None):
    # This function is well-designed and does not need major changes.
    print("  - Starting training pipeline...")
    flag_mapping = {-1: 0, 0: 1, 1: 2}
    train_df['demand_response_flag_mapped'] = train_df[COL_FLAG].map(flag_mapping)
    # Using a simple time split. For single-site models, this is fine.
    #split_date = train_df[COL_TIME].quantile(0.8, interpolation='nearest')
    split_date = train_df[COL_TIME].max() - pd.Timedelta(days=31)
    train_split, val_split = train_df[train_df[COL_TIME] < split_date], train_df[train_df[COL_TIME] >= split_date]
    X_train_clf, y_train_flag = train_split[features_clf], train_split['demand_response_flag_mapped']
    X_val_clf, y_val_flag_mapped = val_split[features_clf], val_split['demand_response_flag_mapped']
    
    if run_grid_search:
        print("    EXECUTING GRID SEARCH...")
        # Classifier Search
        clf_param_grid = {'n_estimators': [1500], 'learning_rate': [0.01, 0.02, 0.03], 'num_leaves': [31, 41, 51, 61], 'colsample_bytree': [0.8], 'min_child_samples': [20], 'reg_alpha': [0.1, 0.5], 'reg_lambda': [0.1, 0.5]}
        lgbm_clf = lgb.LGBMClassifier(objective='multiclass', num_class=3, random_state=42, verbosity=-1, device="gpu", class_weight='balanced')
        grid_search_clf = GridSearchCV(estimator=lgbm_clf, param_grid=clf_param_grid, scoring='f1_macro', cv=TimeSeriesSplit(n_splits=3), n_jobs=-1, verbose=2).fit(X_train_clf, y_train_flag)
        best_clf_params = grid_search_clf.best_params_
        print(f"    Best CLF Params Found: {best_clf_params}")

        # Regressor Search
        train_reg_mask = train_split[COL_FLAG] != 0
        X_train_reg, y_train_reg = train_split[train_reg_mask][features_reg], train_split[train_reg_mask][COL_CAP]
        reg_param_grid = {'n_estimators': [1500], 'learning_rate': [0.01, 0.03, 0.05], 'num_leaves': [40, 50, 60, 80], 'colsample_bytree': [0.8], 'min_child_samples': [20], 'reg_alpha': [0.1, 0.5], 'reg_lambda': [0.1, 0.5]}
        lgbm_reg = lgb.LGBMRegressor(objective='regression_l1', random_state=42, verbosity=-1, device="gpu")
        grid_search_reg = GridSearchCV(estimator=lgbm_reg, param_grid=reg_param_grid, scoring='neg_mean_absolute_error', cv=TimeSeriesSplit(n_splits=3), n_jobs=-1, verbose=2).fit(X_train_reg, y_train_reg)
        best_reg_params = grid_search_reg.best_params_
        print(f"    Best REG Params Found: {best_reg_params}")
    else:
        print("    Skipping GridSearchCV. Using pre-defined best parameters.")
        if params:
            best_clf_params = params["CLF"]
            best_reg_params = params["REG"]
        else: # Default fallback
            best_clf_params = {'colsample_bytree': 0.8, 'learning_rate': 0.02, 'min_child_samples': 20, 'n_estimators': 1500, 'num_leaves': 41, 'reg_alpha': 0.5, 'reg_lambda': 1.0}
            best_reg_params= {'colsample_bytree': 0.8, 'learning_rate': 0.05, 'min_child_samples': 20, 'n_estimators': 1500, 'num_leaves': 60, 'reg_alpha': 0.5, 'reg_lambda': 0.5}
        print(f"    Using CLF Params: {best_clf_params}")
        print(f"    Using REG Params: {best_reg_params}")

    print("    Training final models on full data...")
    clf_params = {**best_clf_params, 'objective': 'multiclass', 'num_class': 3, 'random_state': 42, 'verbosity': -1, 'device': "gpu", 'class_weight': 'balanced'}
    final_classifier = lgb.LGBMClassifier(**clf_params).fit(train_df[features_clf], train_df['demand_response_flag_mapped'], eval_set=[(X_val_clf, y_val_flag_mapped)], eval_metric='multi_logloss', callbacks=[lgb.early_stopping(250, verbose=False)])
    clf_params_xb = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'n_estimators': best_clf_params.get('n_estimators', 2500),
        'learning_rate': best_clf_params.get('learning_rate', 0.026),
        'max_depth': best_clf_params.get('num_leaves', 29) // 2,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': best_clf_params.get('reg_alpha', 0.5),
        'reg_lambda': best_clf_params.get('reg_lambda', 0.5),
        'random_state': 42,
        'eval_metric': 'mlogloss',
        'early_stopping_rounds': 250,
        'tree_method': 'gpu_hist', 
        'predictor': 'gpu_predictor' 
    }
    final_classifier_xb = XGBClassifier(**clf_params_xb)
    final_classifier_xb.fit(
        train_df[features_clf],
        train_df['demand_response_flag_mapped'],
        eval_set=[(X_val_clf, y_val_flag_mapped)],
        verbose=False
    )
    
    train_reg_mask, val_reg_mask = train_df[COL_FLAG] != 0, val_split[COL_FLAG] != 0
    X_full_train_reg, y_full_train_reg = train_df[train_reg_mask][features_reg], train_df[train_reg_mask][COL_CAP]
    X_val_reg, y_val_reg_true = val_split[val_reg_mask][features_reg], val_split[val_reg_mask][COL_CAP]
    reg_params = {**best_reg_params, 'objective': 'huber',  "boosting_type" : 'dart', "random_state" : 155, 'verbosity': -1, 'device': "gpu"}
    final_regressor = lgb.LGBMRegressor(**reg_params).fit(X_full_train_reg, y_full_train_reg, eval_set=[(X_val_reg, y_val_reg_true)], eval_metric='mae', callbacks=[lgb.early_stopping(250, verbose=False)])

    reg_params_xb = {
        'objective': 'reg:absoluteerror', 
        'n_estimators': best_reg_params.get('n_estimators', 10000),
        'learning_rate': best_reg_params.get('learning_rate', 0.0175),
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.3,
        'reg_lambda': 0.2,
        'random_state': 155,
        'eval_metric': 'mae',
        'early_stopping_rounds': 250,
        'tree_method': 'gpu_hist', 
        'predictor': 'gpu_predictor'
    }
    
    final_regressor_xb = XGBRegressor(**reg_params_xb)
    final_regressor_xb.fit(
        X_full_train_reg, y_full_train_reg,
        eval_set=[(X_val_reg, y_val_reg_true)],
        verbose=False
    )    
    # --- GENERATE ENSEMBLED PREDICTIONS ---
    print("    Generating ensemble predictions...")
    
    # Classifier: average probabilities
    flag_probs_lgb = final_classifier.predict_proba(test_df[features_clf])
    flag_probs_xgb = final_classifier_xb.predict_proba(test_df[features_clf])
    flag_probs = 0.7 * flag_probs_lgb + 0.3 * flag_probs_xgb
    
    flag_preds_mapped = np.argmax(flag_probs, axis=1)
    
    # Regressor: average raw predictions
    capacity_preds = np.zeros(len(test_df))
    event_indices = np.where(flag_preds_mapped != 1)[0] 
    
    if len(event_indices) > 0:
        X_test_events = test_df.iloc[event_indices][features_reg]
        lgb_preds = final_regressor.predict(X_test_events)
        xgb_preds = final_regressor_xb.predict(X_test_events)
        ensemble_raw = 0.7 * lgb_preds + 0.3 * xgb_preds
        capacity_preds[event_indices] = ensemble_raw
    
    return {'flag_probs': flag_probs, 'capacity': capacity_preds}

# --- 3. Main Execution ---

# --- GLOBAL MODEL ---
print("\n" + "="*60 + "\n--- STEP 1: PROCESSING GLOBAL MODEL ---\n" + "="*60)
# Generate features on the full dataset
train_df_processed, test_df_processed, features_clf_global, features_reg_global = generate_and_select_features(train_df, test_df)
print(f"  - Global model using {len(features_clf_global)} clf features and {len(features_reg_global)} reg features.")

global_params = {
    "CLF" : {'colsample_bytree': 0.8, 'learning_rate': 0.026, 'min_child_samples': 20, 'n_estimators': 2500, 'num_leaves': 29, 'reg_alpha': 0.46, 'reg_lambda': 0.5},
    "REG" : {'colsample_bytree': 0.8, 'learning_rate': 0.0175, 'min_child_samples': 20, 'n_estimators': 10000, 'num_leaves': 49, 'reg_alpha': 0.3, 'reg_lambda': 0.2}
}

## Manual feature lists based on prior knowledge and importance analysis

selected_features = ['month', 'temp_roll_max_96', 'power_pct_change_4', 'temp_diff_24',
'power_lag_24', 'temp_lag_96', 'archetype_small_temp_interaction',
'power_roll_mean_24', 'power_lag_8', 'temp_roll_min_24', 'dayofweek',
'is_heating_likely', 'temp_x_hour_sin', 'is_morning_ramp_up', 'power_diff_1', 'is_peak_hours', 'state_changed',
'cooling_degree', 'power_diff_96', 'temp_ewm_span_24', 'temp_roll_std_8',
'power_ewm_span_8', 'temp_x_is_weekend', 'power_vs_roll_mean_96', 'temp_roll_std_4',
'temp_lag_144', 'is_business_hours', 'power_pct_change_96', 'temp_rad_interaction',
'archetype_large', 'power_vs_daily_max', 'is_weekend', 'is_summer', 'is_mild_temp_workday',
'archetype_small_power_interaction', 'Global_Horizontal_Radiation_W/m2', 'Building_Power_kW',
'power_roll_std_4', 'power_roll_mean_96', 'is_winter', 'archetype_large_power_interaction', 
'quarter', 'power_lag_48']

official = ['power_roll_mean_4', 'power_cumsum_4', 'is_lunch_hour', 'temp_bins_cold', 'time_of_day_minute', 
            'time_of_day_fraction', 'hour_sin', 'is_evening_ramp_down', 'is_last_work_hour', 'power_roll_max_4',
            'power_roll_max_12', 'power_roll_max_24', 'power_roll_max_48', 'power_diff_24', 
            'power_zscore_24', 'power_zscore_96', 'power_percentile_96']

forgetten = ['hour_cos', 'power_roll_median_24', "power_roll_min_48", "power_roll_median_48",
"power_roll_min_96", "power_roll_max_96", "power_roll_min_144", "power_roll_median_144",
"power_roll_skew_24", "power_roll_kurt_24", "power_roll_kurt_96", "power_pct_change_1",
"power_diff_4", "power_pct_change_24", "power_diff_672", "power_pct_change_672", "is_work_hours_lag", "time_since_last_transition",
"state_Weekend_Day", "state_Workday_Hours", "temp_x_is_work_hours", "rad_x_is_work_hours", "rad_x_is_weekend", 
"rad_x_hour_cos", "effective_temp_load", "hour_x_power_roll_mean_24", "power_lag_96_x_dayofweek", "rad_x_power_lag_96",
"temp_x_power_roll_mean_12", "rad_roll_sum_12", "temp_roll_sum_24", "rad_roll_sum_24", "power_roll_sum_48", 
"rad_roll_sum_48", "power_vs_daily_avg", "power_zscore_24", "power_zscore_96", "power_percentile_96"]

features_reg_global = list(set(selected_features + official + forgetten))


global_predictions = train_and_predict_pipeline(train_df_processed, test_df_processed, features_clf_global, features_reg_global, run_grid_search=RUN_GRID_SEARCH, params=global_params)

# --- SITE-SPECIFIC MODELS ---
print("\n" + "="*60 + "\n--- STEP 2: PROCESSING SITE-SPECIFIC MODELS ---\n" + "="*60)

archetype_params = {
    "CLF" : {'colsample_bytree': 0.8, 'learning_rate': 0.026, 'min_child_samples': 20, 'n_estimators': 2500, 'num_leaves': 29, 'reg_alpha': 0.46, 'reg_lambda': 0.5},
    "REG" : {'colsample_bytree': 0.8, 'learning_rate': 0.0175, 'min_child_samples': 20, 'n_estimators': 10000, 'num_leaves': 49, 'reg_alpha': 0.3, 'reg_lambda': 0.2}
}


all_archetype_predictions = []
# Loop over each unique archetype ('small', 'large')
print(SITE_ARCHETYPE_MAP)
for archetype in ['small', 'large']:
    print(f"\n--- Processing Archetype-Specific Model for: '{archetype}' ---")

    # Find all sites that belong to the current archetype
    sites_in_archetype = [site for site, arch in SITE_ARCHETYPE_MAP.items() if arch == archetype]
    
    # Filter the ORIGINAL raw dataframes for this group of sites
    train_arch_raw = train_df[train_df[COL_SITE].isin(sites_in_archetype)].copy()
    test_arch_raw = test_df[test_df[COL_SITE].isin(sites_in_archetype)].copy()

    if len(train_arch_raw) == 0 or len(test_arch_raw) == 0:
        print(f"  - WARNING: No data for archetype {archetype}. Skipping.")
        continue
        
    # Run feature engineering and selection specifically for this group of sites
    train_df_arch, test_df_arch, features_clf_arch, features_reg_arch = generate_and_select_features(train_arch_raw, test_arch_raw)
    
    print(f"  - Archetype model using {len(features_clf_arch)} clf features and {len(features_reg_arch)} reg features.")
    
    # Train a single model on all data for this archetype
    arch_predictions = train_and_predict_pipeline(
        train_df_arch,
        test_df_arch,
        features_clf_arch,
        features_reg_global,
        run_grid_search=RUN_GRID_SEARCH,
        params=archetype_params
    )
    
    # Store the predictions for all sites in this archetype
    arch_pred_df = pd.DataFrame({
        'Site': test_df_arch[COL_SITE],
        'Timestamp_Local': test_df_arch[COL_TIME],
        'flag_probs_arch': list(arch_predictions['flag_probs']),
        'capacity_arch': arch_predictions['capacity']
    })
    all_archetype_predictions.append(arch_pred_df)


archetype_predictions_df = pd.concat(all_archetype_predictions)

# --- 4. Ensembling & Final Submission ---
print("\n" + "="*60 + "\n--- STEP 3: ENSEMBLING & SAVING ---\n" + "="*60)
global_predictions_df = pd.DataFrame({
    'Site': test_df_processed[COL_SITE], 
    'Timestamp_Local': test_df_processed[COL_TIME], 
    'flag_probs_global': list(global_predictions['flag_probs']), 
    'capacity_global': global_predictions['capacity']
})

# Merge global predictions with the site-specific ones
ensembled_df = pd.merge(global_predictions_df, archetype_predictions_df, on=[COL_SITE, COL_TIME], how='left')

# Handle cases where a site-specific model might have failed
ensembled_df['flag_probs_arch'] = ensembled_df['flag_probs_arch'].fillna(ensembled_df['flag_probs_global'])
ensembled_df['capacity_arch'] = ensembled_df['capacity_arch'].fillna(ensembled_df['capacity_global'])

global_weight = 0.9
site_weight = 0.1
ensembled_df['flag_probs_ensembled'] = ensembled_df.apply(lambda r: (np.array(r['flag_probs_global']) * global_weight + np.array(r['flag_probs_arch']) * site_weight), axis=1)
ensembled_df['capacity_ensembled'] = (ensembled_df['capacity_global'] * global_weight) + (ensembled_df['capacity_arch'] * site_weight)

# Create final submission columns
ensembled_df['flag_mapped_ensembled'] = ensembled_df['flag_probs_ensembled'].apply(np.argmax)
reverse_flag_mapping = {0: -1, 1: 0, 2: 1}
ensembled_df[COL_FLAG] = ensembled_df['flag_mapped_ensembled'].map(reverse_flag_mapping)
ensembled_df[COL_CAP] = ensembled_df['capacity_ensembled']

# Post-processing: ensure capacity is 0 when flag is 0
ensembled_df.loc[ensembled_df[COL_FLAG] == 0, COL_CAP] = 0

# 1. Define the columns required for the submission
submission_columns = [COL_SITE, COL_TIME, COL_FLAG, COL_CAP]

# 2. Select only these columns and sort them to ensure correct order
final_submission = ensembled_df[submission_columns].sort_values(by=[COL_SITE, COL_TIME])

# 3. Save the final DataFrame to CSV
final_submission.to_csv('submission.csv', index=False)

print("\nEnsembled submission file 'submission.csv' created successfully.")