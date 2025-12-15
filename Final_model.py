import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_percentage_error
import xgboost as xgb
try:
    import lightgbm as lgb
    HAS_LGB = True
except:
    HAS_LGB = False
import warnings
warnings.filterwarnings('ignore')


def load_data():
    data = pd.read_csv("master_dataset_includes2023.csv")
    if 'Unnamed: 0' in data.columns:
        data = data.drop('Unnamed: 0', axis=1)
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values('Date').reset_index(drop=True)
    data = data[data['Date'] < '2023-10-15'].reset_index(drop=True)
    return data


def get_weekday_hour_stats(data, train_end_date, n_recent=12):
    """Get comprehensive weekday-hour statistics"""
    train_data = data[data['Date'] <= train_end_date].copy()
    train_data['weekday'] = train_data['Date'].dt.weekday
    train_data['hour'] = train_data['Date'].dt.hour
    train_data['weekday_hour'] = train_data['weekday'] * 24 + train_data['hour']

    recent_cutoff = train_end_date - pd.Timedelta(weeks=n_recent)
    recent_data = train_data[train_data['Date'] >= recent_cutoff]

    wh_stats = recent_data.groupby('weekday_hour')['slot'].agg([
        'mean', 'median', 'std', 'min', 'max',
        lambda x: x.quantile(0.25),
        lambda x: x.quantile(0.75),
        lambda x: x.quantile(0.1),
        lambda x: x.quantile(0.9)
    ])
    wh_stats.columns = ['wh_mean', 'wh_median', 'wh_std', 'wh_min', 'wh_max',
                        'wh_q25', 'wh_q75', 'wh_q10', 'wh_q90']
    mid_point = train_end_date - pd.Timedelta(weeks=n_recent//2)
    recent_half = recent_data[recent_data['Date'] >= mid_point]
    older_half = recent_data[recent_data['Date'] < mid_point]
    recent_means = recent_half.groupby('weekday_hour')['slot'].mean()
    older_means = older_half.groupby('weekday_hour')['slot'].mean()
    wh_stats['wh_trend'] = (recent_means - older_means).fillna(0)
    wh_stats['wh_trend_ratio'] = (recent_means / (older_means + 1)).fillna(1)
    recent_std = recent_half.groupby('weekday_hour')['slot'].std()
    older_std = older_half.groupby('weekday_hour')['slot'].std()
    wh_stats['wh_volatility_trend'] = (recent_std - older_std).fillna(0)

    return wh_stats.to_dict()


def find_similar_weeks_v5(data, train_end_date, n_similar=12):
    train_data = data[data['Date'] <= train_end_date].copy()
    last_week_start = train_end_date - pd.Timedelta(days=6)
    last_week = train_data[(train_data['Date'] >= last_week_start) & (train_data['Date'] <= train_end_date)]

    if len(last_week) == 0:
        return {}, 0

    last_week_mean = last_week['slot'].mean()
    last_week_std = last_week['slot'].std()
    week_of_year = train_end_date.isocalendar()[1]
    train_data['week_start'] = train_data['Date'] - pd.to_timedelta(train_data['Date'].dt.dayofweek, unit='D')
    weekly_stats = train_data.groupby('week_start')['slot'].agg(['mean', 'std', 'median']).reset_index()
    weekly_stats.columns = ['week_start', 'weekly_mean', 'weekly_std', 'weekly_median']
    weekly_stats['week_of_year'] = weekly_stats['week_start'].dt.isocalendar().week
    weekly_stats['year'] = weekly_stats['week_start'].dt.year
    recent_cutoff = train_end_date - pd.Timedelta(weeks=4)
    weekly_stats = weekly_stats[weekly_stats['week_start'] < recent_cutoff]

    if len(weekly_stats) == 0:
        return {}, 0


    week_diff = np.abs(weekly_stats['week_of_year'] - week_of_year)
    week_diff = np.minimum(week_diff, 52 - week_diff)
    weekly_stats['season_sim'] = 1 / (1 + week_diff)
    weekly_stats['level_sim'] = 1 / (1 + np.abs(weekly_stats['weekly_mean'] - last_week_mean) / (last_week_mean + 1))
    weekly_stats['var_sim'] = 1 / (1 + np.abs(weekly_stats['weekly_std'] - last_week_std) / (last_week_std + 1))
    weekly_stats['recency'] = (weekly_stats['year'] - 2012) / 11
    weekly_stats['is_post_covid'] = (weekly_stats['year'] >= 2021).astype(float)
    weekly_stats['is_recent'] = (weekly_stats['year'] >= 2022).astype(float)
    weekly_stats['similarity'] = (
        0.20 * weekly_stats['season_sim'] +
        0.30 * weekly_stats['level_sim'] +
        0.15 * weekly_stats['var_sim'] +
        0.15 * weekly_stats['recency'] +
        0.10 * weekly_stats['is_post_covid'] +
        0.10 * weekly_stats['is_recent']
    )

    top_weeks = weekly_stats.nlargest(n_similar, 'similarity')

    # Create index to weight mapping
    similarity_weights = {}
    for ws, sim in zip(top_weeks['week_start'].values, top_weeks['similarity'].values):
        week_end = ws + pd.Timedelta(days=6)
        mask = (train_data['Date'] >= ws) & (train_data['Date'] <= week_end)
        week_indices = train_data[mask].index.tolist()
        for idx in week_indices:
            similarity_weights[idx] = 1 + 0.5 * sim  # Slightly reduced weight boost

    return similarity_weights, last_week_mean


def create_features_v5(data):
    df = data.copy()
    df['hour'] = df['Date'].dt.hour
    df['day'] = df['Date'].dt.day
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year
    df['weekday'] = df['Date'].dt.weekday
    df['week_of_year'] = df['Date'].dt.isocalendar().week.astype(int)
    df['day_of_year'] = df['Date'].dt.dayofyear
    df['quarter'] = df['Date'].dt.quarter

    for period, name in [(24, 'hour'), (7, 'weekday'), (12, 'month'), (52, 'week'), (4, 'quarter')]:
        col = name if name not in ['week', 'quarter'] else ('week_of_year' if name == 'week' else 'quarter')
        df[f'{name}_sin'] = np.sin(2 * np.pi * df[col] / period)
        df[f'{name}_cos'] = np.cos(2 * np.pi * df[col] / period)
    df['hour_sin2'] = np.sin(4 * np.pi * df['hour'] / 24)
    df['hour_cos2'] = np.cos(4 * np.pi * df['hour'] / 24)
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)
    df['is_friday'] = (df['weekday'] == 4).astype(int)
    df['is_saturday'] = (df['weekday'] == 5).astype(int)
    df['is_sunday'] = (df['weekday'] == 6).astype(int)
    df['is_peak'] = df['hour'].isin([18, 19, 20, 21, 22, 23, 0, 1]).astype(int)
    df['is_late_night'] = df['hour'].isin([0, 1, 2, 3, 4, 5]).astype(int)
    df['is_evening'] = df['hour'].isin([17, 18, 19, 20, 21]).astype(int)
    df['holiday'] = (df['holiday'] != 0).astype(int) if 'holiday' in df.columns else 0
    df['category'] = (df['category'] != 0).astype(int) if 'category' in df.columns else 0
    df['weekday_hour'] = df['weekday'] * 24 + df['hour']

    for i in range(1, 9):
        df[f'slot_lag_{i}week'] = df['slot'].shift(168 * i)

    weights = np.array([0.30, 0.25, 0.20, 0.12, 0.08, 0.05])
    df['slot_exp_weighted'] = sum(df[f'slot_lag_{i}week'] * weights[i-1] for i in range(1, 7))
    lag_4 = ['slot_lag_1week', 'slot_lag_2week', 'slot_lag_3week', 'slot_lag_4week']
    lag_8 = [f'slot_lag_{i}week' for i in range(1, 9)]
    df['slot_mean_4week'] = df[lag_4].mean(axis=1)
    df['slot_mean_8week'] = df[lag_8].mean(axis=1)
    df['slot_std_4week'] = df[lag_4].std(axis=1)
    df['slot_std_8week'] = df[lag_8].std(axis=1)
    df['slot_median_4week'] = df[lag_4].median(axis=1)
    df['slot_min_4week'] = df[lag_4].min(axis=1)
    df['slot_max_4week'] = df[lag_4].max(axis=1)
    df['slot_range_4week'] = df['slot_max_4week'] - df['slot_min_4week']
    df['slot_q25_4week'] = df[lag_4].quantile(0.25, axis=1)
    df['slot_q75_4week'] = df[lag_4].quantile(0.75, axis=1)
    df['slot_iqr_4week'] = df['slot_q75_4week'] - df['slot_q25_4week']
    df['trend_1to2'] = df['slot_lag_1week'] - df['slot_lag_2week']
    df['trend_2to3'] = df['slot_lag_2week'] - df['slot_lag_3week']
    df['trend_3to4'] = df['slot_lag_3week'] - df['slot_lag_4week']
    df['trend_1to4'] = df['slot_lag_1week'] - df['slot_lag_4week']
    df['trend_4to8'] = df['slot_lag_4week'] - df['slot_lag_8week']
    df['trend_ratio_1to2'] = df['slot_lag_1week'] / (df['slot_lag_2week'] + 1)
    df['trend_ratio_1to4'] = df['slot_lag_1week'] / (df['slot_lag_4week'] + 1)
    df['rel_to_mean_4'] = df['slot_lag_1week'] / (df['slot_mean_4week'] + 1)
    df['rel_to_mean_8'] = df['slot_lag_1week'] / (df['slot_mean_8week'] + 1)
    df['rel_to_median'] = df['slot_lag_1week'] / (df['slot_median_4week'] + 1)
    df['zscore'] = (df['slot_lag_1week'] - df['slot_mean_4week']) / (df['slot_std_4week'] + 1)
    df['cv_4week'] = df['slot_std_4week'] / (df['slot_mean_4week'] + 1)


    df['momentum_1'] = df['trend_1to2'] - df['trend_2to3']
    df['momentum_2'] = df['trend_2to3'] - df['trend_3to4']
    df['acceleration'] = df['momentum_1'] - df['momentum_2']
    df['weekend_peak'] = df['is_weekend'] * df['is_peak']
    df['friday_peak'] = df['is_friday'] * df['is_peak']
    df['saturday_peak'] = df['is_saturday'] * df['is_peak']
    df['holiday_weekend'] = df['holiday'] * df['is_weekend']
    df['holiday_peak'] = df['holiday'] * df['is_peak']

    df = df.fillna(0)

    cols_to_drop = ['Date', 'Promo3', 'Promo', 'Promo2', 'Type',
                    'BSAP', 'FAP', 'HAP', 'BKAP', 'FAR', 'BSAR', 'BKAR', 'HAR',
                    'Cost', 'CoinIn', 'Games', 'Revenue',
                    'darryl_table', '4week_ma_table', 'cur_month_max_table', '2week_bf_table',
                    'darryl_slot', '4week_ma_slot', '2week_bf_slot', 'umcsent', 'monthly_gas_price']
    for col in cols_to_drop:
        if col in df.columns:
            df = df.drop(col, axis=1)

    return df


def add_wh_features(df, wh_stats_dict):
    for stat in ['wh_mean', 'wh_median', 'wh_std', 'wh_q25', 'wh_q75', 'wh_q10', 'wh_q90',
                 'wh_trend', 'wh_trend_ratio', 'wh_volatility_trend']:
        default = 0 if 'trend' in stat or 'volatility' in stat else df['slot_mean_4week']
        if stat == 'wh_trend_ratio':
            default = 1
        mapped = df['weekday_hour'].map(wh_stats_dict.get(stat, {}))
        df[stat] = mapped.fillna(default if isinstance(default, (int, float)) else default)

    df['rel_to_wh_mean'] = df['slot_lag_1week'] / (df['wh_mean'] + 1)
    df['rel_to_wh_median'] = df['slot_lag_1week'] / (df['wh_median'] + 1)
    df['diff_from_wh_mean'] = df['slot_lag_1week'] - df['wh_mean']
    df['wh_zscore'] = (df['slot_lag_1week'] - df['wh_mean']) / (df['wh_std'] + 1)
    df['wh_range'] = df['wh_q90'] - df['wh_q10']
    df['wh_position'] = (df['slot_lag_1week'] - df['wh_q10']) / (df['wh_range'] + 1)
    df['wh_baseline'] = df['wh_mean'] + df['wh_trend']
    df['diff_from_baseline'] = df['slot_lag_1week'] - df['wh_baseline']

    return df


class DiverseEnsemble:
    def __init__(self):
        self.models = [
            ('xgb_deep', xgb.XGBRegressor(
                n_estimators=600, max_depth=10, learning_rate=0.03,
                min_child_weight=8, subsample=0.85, colsample_bytree=0.85,
                reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1
            )),
            ('xgb_mid', xgb.XGBRegressor(
                n_estimators=500, max_depth=8, learning_rate=0.05,
                min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0, random_state=43, n_jobs=-1
            )),
            ('xgb_shallow', xgb.XGBRegressor(
                n_estimators=400, max_depth=6, learning_rate=0.06,
                min_child_weight=12, subsample=0.8, colsample_bytree=0.8,
                random_state=44, n_jobs=-1
            )),
            ('hgb', HistGradientBoostingRegressor(
                max_iter=600, max_depth=10, learning_rate=0.03,
                min_samples_leaf=10, l2_regularization=0.1, random_state=42
            )),
            ('rf', RandomForestRegressor(
                n_estimators=200, max_depth=15, min_samples_leaf=8,
                n_jobs=-1, random_state=42
            )),
        ]

        if HAS_LGB:
            self.models.append(('lgb', lgb.LGBMRegressor(
                n_estimators=500, max_depth=8, learning_rate=0.05,
                min_child_samples=15, subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1, verbose=-1
            )))
            self.weights = [0.25, 0.20, 0.12, 0.18, 0.10, 0.15]
        else:
            self.weights = [0.30, 0.25, 0.15, 0.20, 0.10]

    def fit(self, X, y, sample_weight=None):
        for name, model in self.models:
            if sample_weight is not None:
                try:
                    model.fit(X, y, sample_weight=sample_weight)
                except:
                    model.fit(X, y)
            else:
                model.fit(X, y)
        return self

    def predict(self, X):
        preds = [model.predict(X) for name, model in self.models]
        return np.average(preds, axis=0, weights=self.weights)

    def predict_all(self, X):
        return np.array([model.predict(X) for name, model in self.models])


def run_prediction_v5(data, weeks_to_predict=2):
    df = create_features_v5(data)
    all_predictions = []
    all_actuals = []
    current_end_train = pd.to_datetime('2022-12-31')
    current_start_test = pd.to_datetime('2023-01-02')
    days_to_predict = weeks_to_predict * 7 - 1
    current_end_test = current_start_test + pd.Timedelta(days=days_to_predict)
    max_date = data['Date'].max()
    period_count = 0

    while current_end_test <= max_date:
        period_count += 1

        train_mask = data['Date'] <= current_end_train
        test_mask = (data['Date'] >= current_start_test) & (data['Date'] <= current_end_test)

        train_idx = data[train_mask].index
        test_idx = data[test_mask].index
        train_idx = train_idx[train_idx >= 1344]

        if len(test_idx) == 0:
            break

        wh_stats = get_weekday_hour_stats(data, current_end_train, n_recent=12)
        df_period = add_wh_features(df.copy(), wh_stats)
        similarity_weights, _ = find_similar_weeks_v5(data, current_end_train, n_similar=12)
        train_idx_list = train_idx.tolist()
        sample_weight = np.ones(len(train_idx_list))
        for i, idx in enumerate(train_idx_list):
            if idx in similarity_weights:
                sample_weight[i] = similarity_weights[idx]

        train = df_period.loc[train_idx]
        test = df_period.loc[test_idx]

        X_train = train.drop('slot', axis=1)
        y_train = train['slot']
        X_test = test.drop('slot', axis=1)
        y_test = test['slot']
        model = DiverseEnsemble()
        model.fit(X_train, y_train, sample_weight=sample_weight)
        y_pred = model.predict(X_test)

        mape = mean_absolute_percentage_error(y_test, y_pred)
        print(f"Period {period_count}: MAPE = {mape*100:.2f}%")

        all_predictions.extend(y_pred)
        all_actuals.extend(y_test.values)

        current_end_train += pd.Timedelta(days=weeks_to_predict*7)
        current_start_test = current_end_test + pd.Timedelta(days=1)
        current_end_test = current_start_test + pd.Timedelta(days=days_to_predict)

    overall_mape = mean_absolute_percentage_error(all_actuals, all_predictions)
    return overall_mape


def main():
    print("="*70)
    print(f"\nLightGBM available: {HAS_LGB}")
    print("="*70)

    data = load_data()
    print(f"\nData: {data.shape}")
    print("\n--- 2-Week Prediction ---")
    mape_2w = run_prediction_v5(data, weeks_to_predict=2)
    print(f"\n2-Week MAPE: {mape_2w*100:.2f}%")
    print("\n" + "="*70)
    print("Summary:")
    print(f"  2-Week MAPE: {mape_2w*100:.2f}%")

    improvement = (0.1781 - mape_2w) / 0.1781 * 100
    print(f"  Improvement from original: {improvement:.1f}%")
    print("="*70)

    return mape_2w


if __name__ == "__main__":
    main()
