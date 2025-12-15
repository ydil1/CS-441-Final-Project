import pandas as pd
import numpy as np
from scipy import stats

data = pd.read_csv("master_dataset_includes2023.csv")
if 'Unnamed: 0' in data.columns:
    data = data.drop('Unnamed: 0', axis=1)
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date').reset_index(drop=True)


print("\n1. (Weekday, Hour) combination stability analysis")
print("-"*70)


data['weekday'] = data['Date'].dt.weekday
data['hour'] = data['Date'].dt.hour
data['weekday_hour'] = data['weekday'].astype(str) + '_' + data['hour'].astype(str)
wh_stats = data.groupby(['weekday', 'hour'])['slot'].agg(['mean', 'std', 'count'])
wh_stats['cv'] = wh_stats['std'] / wh_stats['mean'] 
print("\n each (weekday, hour)(CV) ")
print(wh_stats['cv'].describe())
print(f"\nMost stable 5 time slots (lowest CV):")
stable_slots = wh_stats['cv'].nsmallest(5)
for (wd, hr), cv in stable_slots.items():
    weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    print(f"  {weekday_names[wd]} {hr}:00 - CV={cv:.3f}, mean={wh_stats.loc[(wd, hr), 'mean']:.1f}")

print(f"\nMost unstable 5 time slots (highest CV):")
unstable_slots = wh_stats['cv'].nlargest(5)
for (wd, hr), cv in unstable_slots.items():
    weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    print(f"  {weekday_names[wd]} {hr}:00 - CV={cv:.3f}, mean={wh_stats.loc[(wd, hr), 'mean']:.1f}")

print("\n\n2. Autocorrelation of the same (Weekday, Hour) across consecutive weeks")
print("-"*70)
data['week_num'] = ((data['Date'] - data['Date'].min()).dt.days // 7)
correlations = []
for wd in range(7):
    for hr in range(24):
        subset = data[(data['weekday'] == wd) & (data['hour'] == hr)].sort_values('Date')
        if len(subset) > 10:
            values = subset['slot'].values
            if len(values) > 2:
                corr_1week = np.corrcoef(values[:-1], values[1:])[0, 1]
                corr_2week = np.corrcoef(values[:-2], values[2:])[0, 1] if len(values) > 3 else np.nan
                corr_4week = np.corrcoef(values[:-4], values[4:])[0, 1] if len(values) > 5 else np.nan
                correlations.append({
                    'weekday': wd,
                    'hour': hr,
                    'corr_1week': corr_1week,
                    'corr_2week': corr_2week,
                    'corr_4week': corr_4week
                })

corr_df = pd.DataFrame(correlations)
print("\nAverage autocorrelation of the same (weekday, hour) across different weeks:")
print(f"  1 week lag: {corr_df['corr_1week'].mean():.3f}")
print(f"  2 weeks lag: {corr_df['corr_2week'].mean():.3f}")
print(f"  4 weeks lag: {corr_df['corr_4week'].mean():.3f}")
print("\n\n3. Analysis of hourly patterns within the week")
print("-"*70)
daily_totals = data.groupby(data['Date'].dt.date)['slot'].sum()
data['date_only'] = data['Date'].dt.date
data['daily_total'] = data['date_only'].map(daily_totals)
data['hour_ratio'] = data['slot'] / data['daily_total']
hour_ratio_stats = data.groupby(['weekday', 'hour'])['hour_ratio'].agg(['mean', 'std'])
hour_ratio_stats['cv'] = hour_ratio_stats['std'] / hour_ratio_stats['mean']
print("\nStability of hourly ratios (proportion of daily total):")
print(f"  Average CV: {hour_ratio_stats['cv'].mean():.3f}")
print(f"  CV range: {hour_ratio_stats['cv'].min():.3f} - {hour_ratio_stats['cv'].max():.3f}")
print("\n\n4. 2022 vs 2023 Yearly Pattern Comparison")
print("-"*70)

data_2022 = data[data['Date'].dt.year == 2022]
data_2023 = data[data['Date'].dt.year == 2023]
wh_2022 = data_2022.groupby(['weekday', 'hour'])['slot'].mean()
wh_2023 = data_2023.groupby(['weekday', 'hour'])['slot'].mean()

ratio_2023_2022 = wh_2023 / wh_2022

print("\n2023/2022 Ratio Statistics:")
print(f"  Average Ratio: {ratio_2023_2022.mean():.3f}")
print(f"  Standard Deviation: {ratio_2023_2022.std():.3f}")
print(f"  Range: {ratio_2023_2022.min():.3f} - {ratio_2023_2022.max():.3f}")
hour_ratios = ratio_2023_2022.groupby(level='hour').mean()
print("\nAverage 2023/2022 Ratios by Hour:")
for hr in [0, 6, 12, 18, 21]:
    print(f"  {hr}:00 - {hour_ratios[hr]:.3f}")
print("\n\n5. Error analysis of predictions using recent N weeks data")
print("-"*70)
data_2023_sorted = data_2023.sort_values('Date').reset_index(drop=True)

def predict_with_recent_weeks(target_date, n_weeks, data_df):
    wd = target_date.weekday()
    hr = target_date.hour
    historical = data_df[(data_df['weekday'] == wd) &
                         (data_df['hour'] == hr) &
                         (data_df['Date'] < target_date)]

    if len(historical) < n_weeks:
        return np.nan

    recent_values = historical.tail(n_weeks)['slot'].values
    return np.mean(recent_values)

print("\nMAPE of predictions using different numbers of recent weeks:")
test_period = data_2023_sorted[data_2023_sorted['Date'] >= '2023-03-01']

for n_weeks in [1, 2, 4, 8]:
    predictions = []
    actuals = []

    for _, row in test_period.iterrows():
        pred = predict_with_recent_weeks(row['Date'], n_weeks, data)
        if not np.isnan(pred):
            predictions.append(pred)
            actuals.append(row['slot'])

    if predictions:
        mape = np.mean(np.abs((np.array(actuals) - np.array(predictions)) / np.array(actuals)))
        print(f"  {n_weeks} weeks average prediction: MAPE = {mape*100:.2f}%")

print("\n\n6. Weighted historical average prediction with decay")
print("-"*70)

def predict_with_decay(target_date, data_df, decay=0.9):
    wd = target_date.weekday()
    hr = target_date.hour

    historical = data_df[(data_df['weekday'] == wd) &
                         (data_df['hour'] == hr) &
                         (data_df['Date'] < target_date)].sort_values('Date')

    if len(historical) < 4:
        return np.nan
    values = historical['slot'].values[-52:]
    n = len(values)
    weights = np.array([decay ** i for i in range(n-1, -1, -1)])
    weights = weights / weights.sum()

    return np.sum(values * weights)

print("\nMAPE of predictions using different decay factors:")
for decay in [0.7, 0.8, 0.9, 0.95]:
    predictions = []
    actuals = []

    for _, row in test_period.iterrows():
        pred = predict_with_decay(row['Date'], data, decay)
        if not np.isnan(pred):
            predictions.append(pred)
            actuals.append(row['slot'])

    if predictions:
        mape = np.mean(np.abs((np.array(actuals) - np.array(predictions)) / np.array(actuals)))
        print(f"  Decay factor={decay}: MAPE = {mape*100:.2f}%")

