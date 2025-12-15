# Casino Headcount Prediction System

## Project Overview

This project develops a machine learning ensemble model to predict hourly slot machine headcount (staffing needs) for Casino. The system uses historical data from 2012-2023 combined with external factors (promotions, holidays, gas prices, sports events, customer sentiment) to forecast staffing requirements 1-2 weeks in advance.

**Key Achievement**: Average 2-week prediction MAPE of ~17.4% across 21 rolling evaluation periods.

---

## Project Structure

```
final_deliever/
│
├── Python Scripts
│   ├── merge_input_dataset.ipynb    # Data preprocessing & feature engineering
│   ├── Final_model.py              # Main ensemble model (training & evaluation)
│   └── analyze_weekly_patterns.py # Pattern analysis & exploration
│
├── Data Files
│   ├── Final.csv                      # Original training data (2012-2023)
│   ├── master_dataset_includes2023.csv      # Preprocessed merged dataset
│   ├── HourlyCoinInGames2023.csv            # Slot machine coin-in metrics
│   ├── SlotHeadcounts2023.csv               # Slot staff headcount data
│   ├── TableHeadcounts2023.csv              # Table staff headcount data
│   ├── APUS23B74716_gas_2023_Jan_til_oct.csv # Monthly gas prices
│   └── sca-table1-on-2023-Nov-27.csv        # Customer sentiment index
│
├── Excel Data Sources
│   ├── Promos_data.xlsx  # Marketing promotions
│   ├── test.xlsx                         # Test dataset structure
│   ├── Holiday list With Edits.xlsx             # Holiday calendar
│   └── 2023_sports_away_games.xlsx              # Sports event schedule
│
├── Model & Results
│   ├── best_ensemble_model.pkl      # Trained model (243 MB)
│   ├── best_model_results.csv       # 2-week MAPE evaluation results
│   ├── weekly_mape_results.csv      # 1-week MAPE results
│   └── final_two_week_mape.csv      # Final evaluation metrics
│
└── Documentation
    └── Headcount to Cost Conversion.docx
```

---

## File Descriptions

### Python Scripts

#### 1. `merge_input_dataset.ipynb`
**Purpose**: Data preprocessing and feature engineering pipeline

**Key Components**:
- Promotional data parsing with timeframe extraction (lines 24-56)
- Multi-source data merging - 10+ datasets (lines 142-218)
- Custom sklearn transformers for automated preprocessing (lines 232-320)
- Missing value handling and categorical encoding (lines 270-313)

**Input**: Multiple CSV and Excel files
**Output**: `master_dataset_includes2023.csv`

---

#### 2. `Final_model.py`
**Purpose**: Main ensemble model training and 2-week rolling evaluation

**Model Architecture** (Weighted Ensemble):
| Model | Parameters | Weight |
|-------|------------|--------|
| XGBoost (Deep) | 600 trees, depth=10, lr=0.03 | 25-30% |
| XGBoost (Mid) | 500 trees, depth=8, lr=0.05 | 20-25% |
| XGBoost (Shallow) | 400 trees, depth=6, lr=0.06 | 12-15% |
| HistGradientBoosting | 600 iterations, depth=10 | 18-20% |
| RandomForest | 200 trees, depth=15 | 10% |
| LightGBM (optional) | 500 trees, depth=8 | 15% |

**Input**: `master_dataset_includes2023.csv`
**Output**: `best_ensemble_model.pkl`, `best_model_results.csv`

---

#### 3. `analyze_weekly_patterns.py`
**Purpose**: Exploratory data analysis for pattern discovery

**Analysis Includes**:
- Coefficient of variation (CV) for each (weekday, hour) slot (lines 32-48)
- Auto-correlation across 1-week, 2-week, 4-week intervals (lines 50-80)
- Year-over-year comparison (2022 vs 2023)
- Historical average prediction tests

---

## Feature Engineering (Detailed)

### 1. Temporal Features
> **Location**: `Final_model.py` → `create_features_v5()` function (lines 113-139)

```python
# Basic temporal extraction (Final_model.py lines 115-122)
hour, day, month, year, weekday, week_of_year, day_of_year, quarter

# Sinusoidal encoding for cyclical patterns (Final_model.py lines 124-127)
hour_sin = sin(2π * hour / 24)      hour_cos = cos(2π * hour / 24)
weekday_sin = sin(2π * weekday / 7)  weekday_cos = cos(2π * weekday / 7)
month_sin = sin(2π * month / 12)     month_cos = cos(2π * month / 12)
week_sin = sin(2π * week / 52)       week_cos = cos(2π * week / 52)

# Second harmonic for hour (Final_model.py lines 128-129)
hour_sin2 = sin(4π * hour / 24)      hour_cos2 = cos(4π * hour / 24)

# Binary time indicators (Final_model.py lines 130-136)
is_weekend, is_friday, is_saturday, is_sunday
is_peak (18:00-01:00), is_late_night (00:00-05:00), is_evening (17:00-21:00)
```

### 2. Lag Features (Historical Patterns)
> **Location**: `Final_model.py` → `create_features_v5()` function (lines 141-158)

```python
# Weekly lag values - 1-8 weeks back, shift = 168 hours per week (Final_model.py lines 141-142)
slot_lag_1week, slot_lag_2week, ..., slot_lag_8week

# Exponentially weighted mean - recent weeks weighted higher (Final_model.py lines 144-145)
weights = [0.30, 0.25, 0.20, 0.12, 0.08, 0.05]
slot_exp_weighted = Σ(slot_lag_i * weight_i)

# Rolling statistics over 4-week and 8-week windows (Final_model.py lines 146-158)
slot_mean_4week, slot_mean_8week
slot_std_4week, slot_std_8week
slot_median_4week, slot_min_4week, slot_max_4week
slot_range_4week, slot_q25_4week, slot_q75_4week, slot_iqr_4week
```

### 3. Trend & Momentum Features
> **Location**: `Final_model.py` → `create_features_v5()` function (lines 159-175)

```python
# Week-to-week trends (Final_model.py lines 159-163)
trend_1to2 = slot_lag_1week - slot_lag_2week
trend_2to3 = slot_lag_2week - slot_lag_3week
trend_1to4 = slot_lag_1week - slot_lag_4week
trend_4to8 = slot_lag_4week - slot_lag_8week

# Trend ratios - relative change (Final_model.py lines 164-165)
trend_ratio_1to2 = slot_lag_1week / (slot_lag_2week + 1)
trend_ratio_1to4 = slot_lag_1week / (slot_lag_4week + 1)

# Position relative to historical average (Final_model.py lines 166-170)
rel_to_mean_4 = slot_lag_1week / (slot_mean_4week + 1)
rel_to_median = slot_lag_1week / (slot_median_4week + 1)
zscore = (slot_lag_1week - slot_mean_4week) / (slot_std_4week + 1)
cv_4week = slot_std_4week / (slot_mean_4week + 1)  # Coefficient of variation

# Momentum and acceleration - second-order trends (Final_model.py lines 173-175)
momentum_1 = trend_1to2 - trend_2to3
momentum_2 = trend_2to3 - trend_3to4
acceleration = momentum_1 - momentum_2
```

### 4. Weekday-Hour Statistics Features
> **Location**: `Final_model.py` → `get_weekday_hour_stats()` function (lines 26-56) and `add_wh_features()` function (lines 196-214)

```python
# Statistics for each (weekday, hour) combination from recent 12 weeks (Final_model.py lines 36-44)
wh_mean, wh_median, wh_std, wh_min, wh_max
wh_q10, wh_q25, wh_q75, wh_q90

# Trend analysis - recent 6 weeks vs older 6 weeks (Final_model.py lines 45-54)
wh_trend = recent_mean - older_mean
wh_trend_ratio = recent_mean / (older_mean + 1)
wh_volatility_trend = recent_std - older_std

# Relative position features (Final_model.py lines 205-212)
rel_to_wh_mean = slot_lag_1week / (wh_mean + 1)
diff_from_wh_mean = slot_lag_1week - wh_mean
wh_zscore = (slot_lag_1week - wh_mean) / (wh_std + 1)
wh_position = (slot_lag_1week - wh_q10) / (wh_range + 1)
wh_baseline = wh_mean + wh_trend
```

### 5. Interaction Features
> **Location**: `Final_model.py` → `create_features_v5()` function (lines 176-180)

```python
# Time-based interactions (Final_model.py lines 176-178)
weekend_peak = is_weekend * is_peak
friday_peak = is_friday * is_peak
saturday_peak = is_saturday * is_peak

# Holiday interactions (Final_model.py lines 179-180)
holiday_weekend = holiday * is_weekend
holiday_peak = holiday * is_peak
```

---

## Missing Value Handling (Detailed)

### Strategy Overview

We implement a **domain-aware imputation strategy** using custom sklearn transformers.

### 1. `missing_encoder` Class - Imputation Strategy
> **Location**: `merge_input_dataset.ipynb` → `missing_encoder` class

```python
class missing_encoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Strategy 1: Mean imputation for continuous operational metrics (line 275)
        self.num_mean_columns = ['CoinIn', 'Games', 'monthly_gas_price']

        # Strategy 2: Zero imputation for event-based features (line 276)
        self.num_const_column = ['Cost', 'Revenue', 'umcsent']

        # Strategy 3: "0" string for categorical missing values (line 277)
        self.cate_const_columns = ['holiday', 'category', 'Type']
```

| Column Type | Columns | Strategy | Rationale |
|-------------|---------|----------|-----------|
| Operational Metrics | CoinIn, Games, monthly_gas_price | **Mean Imputation** | These are continuous metrics that should have typical values when missing |
| Event-based Numerics | Cost, Revenue, umcsent | **Zero Imputation** | Missing = no promotion/event occurred, so 0 is semantically correct |
| Categorical | holiday, category, Type | **"0" String** | Missing = non-holiday/no category, encoded as a distinct category |

### 2. `lag_transformer` Class - Temporal Shifting
> **Location**: `merge_input_dataset.ipynb` → `lag_transformer` class

```python
class lag_transformer(BaseEstimator, TransformerMixin):
    def __init__(self, lag=24*21):  # 21 days = 504 hours
        self.lag = lag
```

**Key Innovation**: 21-day forward shift for `CoinIn` and `Games` (line 256)
- **Business Logic**: Staff schedules must be created 3 weeks in advance
- **Implementation**: `X[['CoinIn','Games']] = X[['CoinIn','Games']].shift(504)`
- **Result**: Model can only use information available at prediction time

### 3. Post-lag NaN Handling
> **Location**: `merge_input_dataset.ipynb` → `drop_nans()` function

After lag transformation, the first 504 rows (21 days) will have NaN values:
```python
# Drop rows with NaN from moving average calculation (merge_input_dataset.ipynb)
def drop_nans(X, y=None):
    df = pd.DataFrame(X)
    df.dropna(subset=['darryl_table'], inplace=True)
    return df
```

### 4. `value_encoder` Class - Categorical Encoding
> **Location**: `merge_input_dataset.ipynb` → `value_encoder` class

```python
class value_encoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Ordinal encoding with unknown value handling (line 300)
        self.encoder1 = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=5000)
        self.num_columns = ['holiday', 'category', 'Promo', 'Promo2', 'Promo3', 'Type']
```

### 5. Final NaN Cleanup in Model
> **Location**: `Final_model.py` → `create_features_v5()` function (line 182)

```python
# Any remaining NaN filled with 0 (Final_model.py line 182)
df = df.fillna(0)
```

### 6. Complete Preprocessing Pipeline
> **Location**: `merge_input_dataset.ipynb` (lines 315-320)

```python
pipe = Pipeline([
    ("missing_value_imputer", missing_encoder()),  # Step 1: Fill missing values
    ("shifter", lag_transformer()),                 # Step 2: Apply 21-day lag
    ("dropper", FunctionTransformer(drop_nans)),   # Step 3: Drop NaN rows
    ("ordinal_encoder", value_encoder()),          # Step 4: Encode categories
])
```

---

## Key Innovations

### 1. Similarity-Based Sample Weighting
> **Location**: `Final_model.py` → `find_similar_weeks_v5()` function (lines 59-110)

**Problem**: Not all historical data is equally relevant for predicting a specific future period.

**Solution**: We identify historically similar weeks and upweight them during training:

```python
# Multi-factor similarity scoring (Final_model.py lines 90-97)
similarity = (
    0.20 * season_sim +      # Same time of year (week of year)
    0.30 * level_sim +       # Similar headcount levels
    0.15 * var_sim +         # Similar variance patterns
    0.15 * recency +         # More recent = more relevant
    0.10 * is_post_covid +   # Post-COVID behavior shift
    0.10 * is_recent         # 2022+ data preferred
)
```

**Seasonality Similarity** (Final_model.py lines 82-84): Uses circular distance for week-of-year
```python
week_diff = min(|week1 - week2|, 52 - |week1 - week2|)
season_sim = 1 / (1 + week_diff)
```

**Sample Weight Application** (Final_model.py lines 301-305):
```python
sample_weight = np.ones(len(train_idx_list))
for i, idx in enumerate(train_idx_list):
    if idx in similarity_weights:
        sample_weight[i] = similarity_weights[idx]
```

### 2. Dynamic Weekday-Hour Statistics
> **Location**: `Final_model.py` → `get_weekday_hour_stats()` function (lines 26-56)

**Problem**: Same (weekday, hour) slots have different patterns that evolve over time.

**Solution**: Compute statistics dynamically for each prediction period using only recent data:

```python
def get_weekday_hour_stats(data, train_end_date, n_recent=12):
    # Use only last 12 weeks before prediction date (lines 33-34)
    recent_cutoff = train_end_date - pd.Timedelta(weeks=n_recent)
    recent_data = train_data[train_data['Date'] >= recent_cutoff]

    # Compute rich statistics per (weekday, hour) (lines 36-44)
    wh_stats = recent_data.groupby('weekday_hour')['slot'].agg([
        'mean', 'median', 'std', 'min', 'max',
        quantile(0.10), quantile(0.25), quantile(0.75), quantile(0.90)
    ])

    # Trend detection: compare recent vs older half (lines 45-54)
    wh_stats['wh_trend'] = recent_half_mean - older_half_mean
    wh_stats['wh_volatility_trend'] = recent_std - older_std
```

### 3. 21-Day Lag Transformation for Real-World Constraints
> **Location**: `merge_input_dataset.ipynb` → `lag_transformer` class

**Business Constraint**: Casino staff schedules must be finalized 3 weeks in advance.

**Implementation** (line 256): Shift operational metrics forward by 21 days:
```python
X[['CoinIn', 'Games']] = X[['CoinIn', 'Games']].shift(24 * 21)
```

### 4. Diverse Ensemble with Complementary Models
> **Location**: `Final_model.py` → `DiverseEnsemble` class (lines 217-271)

**Design Philosophy**: Combine models with different strengths:

| Model | Code Location | Strength | Role in Ensemble |
|-------|---------------|----------|------------------|
| XGBoost Deep (depth=10) | lines 220-224 | Captures complex interactions | Primary predictor |
| XGBoost Mid (depth=8) | lines 225-229 | Balanced complexity | Secondary predictor |
| XGBoost Shallow (depth=6) | lines 230-234 | Generalization | Reduces overfitting |
| HistGradientBoosting | lines 235-238 | Handles missing values natively | Robustness |
| RandomForest | lines 239-242 | Variance reduction | Stability |
| LightGBM | lines 246-250 | Speed + accuracy | Optional booster |

**Weighted Prediction** (Final_model.py lines 266-268):
```python
def predict(self, X):
    preds = [model.predict(X) for name, model in self.models]
    return np.average(preds, axis=0, weights=self.weights)
```

### 5. Rolling Window Validation
> **Location**: `Final_model.py` → `run_prediction_v5()` function (lines 274-329)

**Approach**: Mimic real-world deployment scenario
```python
# Train on all data up to date X, predict next 2 weeks (lines 285-326)
while current_end_test <= max_date:
    train_mask = data['Date'] <= current_end_train
    test_mask = (data['Date'] >= current_start_test) & (data['Date'] <= current_end_test)
    # ... train and evaluate ...
    current_end_train += pd.Timedelta(days=weeks_to_predict*7)  # line 324
```

---

## How to Run the Model

### Prerequisites

```bash
pip install pandas numpy scikit-learn xgboost scipy matplotlib seaborn
pip install lightgbm
```

### Step 1: Data Preprocessing

If you need to regenerate the merged dataset from raw sources:

**Option 1**: Run in Jupyter Notebook/Lab
```bash
jupyter notebook merge_input_dataset.ipynb
```

**Option 2**: Run via command line
```bash
jupyter nbconvert --to notebook --execute merge_input_dataset.ipynb
```

**Option 3**: Convert to Python script first
```bash
jupyter nbconvert --to script merge_input_dataset.ipynb
python merge_input_dataset.py
```

**Output**: `master_dataset_includes2023.csv`

---

### Step 2: Train the Model

```bash
python Final_model.py
```

This will:
1. Load the preprocessed dataset (`load_data()` function, lines 16-23)
2. Create 50+ temporal and statistical features (`create_features_v5()` function, lines 113-193)
3. Train diverse ensemble with rolling validation (`run_prediction_v5()` function, lines 274-329)
4. Evaluate on 21 two-week prediction periods
5. Print MAPE results for each period

**Output**:
- Console output with MAPE for each 2-week period
- Overall MAPE summary

---

### Step 3: Make Predictions (Using Trained Model)

```python
import pickle
import pandas as pd
import numpy as np

# Load the trained model
with open('best_ensemble_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load and prepare your data
data = pd.read_csv('master_dataset_includes2023.csv')
data['Date'] = pd.to_datetime(data['Date'])

# Make predictions
predictions = model.predict(X_test)
```

---

### Step 4: Run Pattern Analysis (Optional)

```bash
python analyze_weekly_patterns.py
```

This script analyzes:
- Weekday-hour stability (CV analysis) - lines 32-48
- Auto-correlation patterns - lines 50-80
- Year-over-year trends

---

## Model Performance

### 2-Week Rolling Prediction Results (MAPE)

| Period | MAPE | Period | MAPE |
|--------|------|--------|------|
| Jan 02-15 | 17.4% | Jun 05-18 | 13.0% |
| Jan 16-29 | 24.1% | Jun 19-Jul 02 | 28.0% |
| Jan 30-Feb 12 | 13.1% | Jul 03-16 | 22.5% |
| Feb 13-26 | 22.0% | Jul 17-30 | 13.6% |
| Feb 27-Mar 12 | 28.6% | Jul 31-Aug 13 | 14.1% |
| Mar 13-26 | 30.8% | Aug 14-27 | 14.4% |
| Mar 27-Apr 09 | 19.5% | Aug 28-Sep 10 | 13.9% |
| Apr 10-23 | 16.0% | Sep 11-24 | 12.8% |
| Apr 24-May 07 | **11.3%** | Sep 25-Oct 08 | 14.6% |
| May 08-21 | 14.9% | | |
| May 22-Jun 04 | 11.7% | **Average** | **~17.4%** |

**Best Performance**: April-May (~11-15% MAPE)
**Challenging Periods**: Late February-March, Late June (~28-31% MAPE)

---

## Technical Notes

1. **Data Cutoff**: Model excludes data after 2023-10-15 due to data quality issues (`Final_model.py` line 22)
2. **Lag Transformation**: CoinIn/Games are shifted 21 days ahead (`merge_input_dataset.ipynb`)
3. **Training Start**: First 1344 hours (8 weeks) excluded to ensure sufficient lag features (`Final_model.py` line 293)
4. **Model Size**: Trained ensemble is ~243 MB (pickle format)

---

## License

This project was developed for academic purposes as part of the University of Illinois CS 441 Final Project group.



