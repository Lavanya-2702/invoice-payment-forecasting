# ============================================================================
# Invoice Payment Forecasting: Delay Prediction and Aging Buckets
# ============================================================================
import os  # creates directory
import warnings  # avoids harmless warning messages
warnings.filterwarnings("ignore")
import datetime as dt
import pandas as pd  # loading csv, cleaning and handles dataframes
import numpy as np  # Fast math array of numbers
import matplotlib
matplotlib.use("Agg")  # save charts to files (no GUI needed)
import matplotlib.pyplot as plt  # For charts
import seaborn as sns  # For cleaner look of charts
sns.set_theme(style="whitegrid")
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb  # A fast gradient boosting library
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)
CAD_TO_USD = 0.75  # approx. 2019 average CAD to USD rate (Bank of Canada)

#Storing the dataset in Dataframe
df = pd.read_csv("dataset_project.csv")

#Analysing shape, information and count of null values
print(df.shape)
print(df.info())
print(df.columns)
print(df.describe())
print(df.isnull().sum())

# ----------------------------------------------------------------------------
# 1. DATA CLEANING
# ----------------------------------------------------------------------------
#Checking if there is any duplicate columns - id
print(df[["invoice_id", "doc_id"]])
check = np.where((df['invoice_id'] == df['doc_id']), "Identical", "Different")
check_1 = pd.DataFrame(check, columns=['Identical/Different'])
print(pd.DataFrame(check_1.value_counts()))
#Interpretation: 6 rows differ because invoice_id is empty there (the 6 'X2' documents)
#Remove invoice_id and retain doc_id (it is complete)

#Checking if there is any duplicate columns - date
print(df[['document_create_date', 'document_create_date.1', 'baseline_create_date']])
check = np.where((df['document_create_date.1'] == df['document_create_date']),
                 "Identical", "Different")
check_2 = pd.DataFrame(check, columns=['Identical/Different'])
check = np.where((df['document_create_date.1'] == df['baseline_create_date']),
                 "Identical", "Different")
check_3 = pd.DataFrame(check, columns=['Identical/Different'])
check = np.where(pd.to_datetime(df['document_create_date.1'].astype(str), format="%Y%m%d")
                 == pd.to_datetime(df['posting_date']), "Identical", "Different")
check_4 = pd.DataFrame(check, columns=['Identical/Different'])
print(pd.DataFrame(check_2.value_counts()), pd.DataFrame(check_3.value_counts()),
      pd.DataFrame(check_4.value_counts()))
#Interpretation: document_create_date.1 is a copy of posting_date (~100% identical)
#and is ~89% identical to baseline_create_date, so it adds nothing new.
#document_create_date is only a near-posting date (differs by a few days).
#Remove document_create_date, document_create_date.1 and retain baseline_create_date

print(df[["posting_id", "isOpen", "document type"]].nunique())
print(df["document type"].value_counts())
#Interpretation: posting_id is constant. isOpen only repeats "clear_date is empty"
#(the answer we predict), so it would leak the target. document type has just 6 'X2' rows.
#Keep standard invoices (RV) only, then remove the 3 columns and empty column area business
df = df[df["document type"] == "RV"]

cols: list[str] = ["area_business", "posting_id", "invoice_id", "document_create_date",
                   "isOpen", "document type", "document_create_date.1"]
df.drop(cols, axis=1, inplace=True)
print(df.columns)

#Removing duplicates
print(df.duplicated().sum())
df.drop_duplicates(keep='first', inplace=True)
total_duplicate = df.duplicated().sum()
print(f"total_duplicate: {total_duplicate}")  #it must be 0 right now

#Checking null
finding_null = df.isnull().sum()
print(finding_null)
total_null = df.isnull().sum().sum()
print(f"total_null : {total_null}")

# ----------------------------------------------------------------------------
# 2. DATA TYPE CONVERSION
# ----------------------------------------------------------------------------
data_type = df.dtypes
print(pd.DataFrame(data_type))


#lets convert data columns into date formats "%Y%m%d"
def date_format(date):
    x = str(date)  #Coverting date dtypes to str & slicing
    year = int(x[0:4])
    month = int(x[4:6])
    day = int(x[6:])
    return dt.datetime(year, month, day)


# doc_id is an identifier, not a measurement: keep it as a whole number
df['doc_id'] = df['doc_id'].astype('int64')

# Down-casting date columns from float to int
df['buisness_year'] = pd.to_numeric(df['buisness_year'], downcast='integer')
df['due_in_date'] = pd.to_numeric(df['due_in_date'], downcast='integer')
df['baseline_create_date'] = pd.to_numeric(df['baseline_create_date'], downcast='integer')

# Applying the date formatter
df['due_in_date'] = df['due_in_date'].apply(date_format)
df['baseline_create_date'] = df['baseline_create_date'].apply(date_format)

# Converting object type to datetime type
df['clear_date'] = pd.to_datetime(df['clear_date'])
df['posting_date'] = pd.to_datetime(df['posting_date'])

# Printing the head of the df
print(df.head())

#Checking data types
print(pd.DataFrame(df[["buisness_year", "clear_date",
                       "posting_date", "due_in_date",
                       "baseline_create_date"]].dtypes))

# ----------------------------------------------------------------------------
# 3. CURRENCY CONVERSION
# ----------------------------------------------------------------------------
print(pd.DataFrame(df["invoice_currency"].value_counts()))
print(df["total_open_amount"])

#converting CAD to USD and storing "converted_usd"
df['converted_usd'] = np.where(df['invoice_currency'] == 'CAD', df['total_open_amount'] * CAD_TO_USD,
                               df['total_open_amount']).round(2)
print(df['converted_usd'])

#dropping 'invoice_currency', 'total_open_amount' (business_code still identifies the CAD unit)
df.drop(['invoice_currency', 'total_open_amount'], axis=1, inplace=True)
print(len(df.columns))

#Checking data count - year wise
print(pd.DataFrame(df["buisness_year"].value_counts()))

# ----------------------------------------------------------------------------
# 4. CUSTOMER IDENTIFICATION (customer number)
# ----------------------------------------------------------------------------
#The same customer number can appear with and without leading zeros (0100000217 / 100000217)
df['cust_num_norm'] = df['cust_number'].astype(str).str.lstrip('0').replace('', '0')
print(df["cust_num_norm"].nunique())

#One readable, unique label per customer number = its most frequent name (suffix removed) + the number
name_counts = df.groupby(["cust_num_norm", "name_customer"]).size().reset_index(name="n")
print(name_counts)
top_name = (name_counts.sort_values(["cust_num_norm", "n"], ascending=[True, False])
            .drop_duplicates("cust_num_norm").set_index("cust_num_norm")["name_customer"]
            .str.replace(r"\s+(corporation|corp|in|us|trust|llc|associates|co|foundation|systems)$",
                         "", regex=True).str.strip())
print(top_name)
df["customer_label"] = df["cust_num_norm"].map(top_name) + " (" + df["cust_num_norm"] + ")"
label_lookup = df.drop_duplicates("cust_num_norm").set_index("cust_num_norm")["customer_label"]
print(label_lookup)
print(f"{df['customer_label'].nunique()} unique customer labels")

# ----------------------------------------------------------------------------
# 5. SPLITTING DATASET (paid vs open invoices)
# ----------------------------------------------------------------------------
no_of_null_columns = [i for i in df.isnull().any().keys() if df.isnull().any()[i] == True]
print(no_of_null_columns)

columns_null_sums = df.isnull().sum()
print(columns_null_sums)

null_data = df.loc[df['clear_date'].isnull()].copy()  # open invoices (to predict)
main_data = df.loc[~df['clear_date'].isnull()].copy()  # paid invoices (to learn from)

print(main_data.shape)
print(main_data.head())
print(null_data.shape)
print(null_data.head())

# ----------------------------------------------------------------------------
# 6. TARGET: PAYMENT DELAY
# ----------------------------------------------------------------------------
main_data["Delay"] = main_data["clear_date"] - main_data["due_in_date"]
main_data["delay_days"] = main_data["Delay"].dt.days  # negative = paid early
print(main_data["delay_days"].describe().round(2))
print(f"Paid late: {(main_data['delay_days'] > 0).mean():.1%} | "
      f"95th pct: {main_data['delay_days'].quantile(.95):.0f} days | "
      f"99th pct: {main_data['delay_days'].quantile(.99):.0f} days")

# ----------------------------------------------------------------------------
# 7. EXPLORATORY DATA ANALYSIS
# ----------------------------------------------------------------------------
cust_summary = (main_data.groupby("cust_num_norm")["delay_days"]
                .agg(avg_delay="mean", invoices="size").query("invoices >= 20")
                .sort_values("avg_delay", ascending=False))
print(cust_summary.head(10).round(1))
print(main_data.groupby("business_code")["delay_days"].agg(["mean", "median", "size"]).round(2))

fig, ax = plt.subplots(1, 3, figsize=(18, 4.5))
ax[0].hist(main_data["delay_days"].clip(-30, 60), bins=90, color="#2a6f97")
ax[0].axvline(0, color="k", ls="--", lw=1)
ax[0].set(title="Payment delay (days after due date, clipped -30..60)", xlabel="days", ylabel="invoices")
dow = main_data.groupby(main_data["due_in_date"].dt.dayofweek)["delay_days"].median()
ax[1].bar(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], dow.reindex(range(7)).values, color="#468faf")
ax[1].set(title="Median delay by due-date weekday", ylabel="median days late")
top10 = cust_summary.head(10).iloc[::-1]
ax[2].barh(label_lookup.reindex(top10.index).values, top10["avg_delay"], color="#61a5c2")
ax[2].set(title="10 slowest-paying customers (20+ invoices)", xlabel="avg days late")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/01_eda.png", dpi=120)
plt.close()

# ----------------------------------------------------------------------------
# 8. FEATURE ENGINEERING
# ----------------------------------------------------------------------------
# 8a. Invoice features, known when the invoice is posted (applied to paid AND open invoices)
all_cat = pd.concat([main_data, null_data])
for frame in (main_data, null_data):
    frame["log_usd"] = np.log1p(frame["converted_usd"])
    frame["term_days"] = (frame["due_in_date"] - frame["baseline_create_date"]).dt.days
    frame["days_to_due"] = (frame["due_in_date"] - frame["posting_date"]).dt.days
    frame["base_gap"] = (frame["posting_date"] - frame["baseline_create_date"]).dt.days
    frame["post_month"] = frame["posting_date"].dt.month
    frame["post_dow"] = frame["posting_date"].dt.dayofweek
    frame["post_dom"] = frame["posting_date"].dt.day
    frame["due_dow"] = frame["due_in_date"].dt.dayofweek
    frame["due_month_end"] = frame["due_in_date"].dt.is_month_end.astype(int)
    frame["business_code_enc"] = pd.Categorical(frame["business_code"], categories=all_cat["business_code"].unique()).codes
    frame["payment_terms_enc"] = pd.Categorical(frame["cust_payment_terms"],
                                                categories=all_cat["cust_payment_terms"].unique()).codes

# 8b. Customer payment history, LEAKAGE-SAFE.
# For every invoice, use only that customer's invoices already CLEARED strictly before
# this invoice was posted (what a finance team would actually know at that moment).
history = main_data[["cust_num_norm", "clear_date", "delay_days"]].sort_values("clear_date").reset_index(drop=True)
history["late"] = (history["delay_days"] > 0).astype(int)
history["cum_delay"] = history.groupby("cust_num_norm")["delay_days"].cumsum()
history["cum_late"] = history.groupby("cust_num_norm")["late"].cumsum()
history["cum_n"] = history.groupby("cust_num_norm").cumcount() + 1
history = history.drop(columns=["delay_days", "late"]).rename(columns={"clear_date": "history_date"})

main_data = pd.merge_asof(main_data.sort_values("posting_date"), history, left_on="posting_date",
                          right_on="history_date", by="cust_num_norm", allow_exact_matches=False)
null_data = pd.merge_asof(null_data.sort_values("posting_date"), history, left_on="posting_date",
                          right_on="history_date", by="cust_num_norm", allow_exact_matches=False)
for frame in (main_data, null_data):
    frame["cust_avg_delay"] = frame["cum_delay"] / frame["cum_n"]  # customer's average delay so far
    frame["cust_late_rate"] = frame["cum_late"] / frame["cum_n"]  # share of invoices paid late so far
    frame["cust_n"] = frame["cum_n"].fillna(0)  # prior paid invoices (0 = new customer)

# 8c. Leakage audit: recompute the history the slow, obvious way for 300 random invoices
audit_ok = True
for _, r in main_data.sample(300, random_state=42).iterrows():
    prior = main_data[(main_data["cust_num_norm"] == r["cust_num_norm"]) & (main_data["clear_date"] < r["posting_date"])]
    expected = prior["delay_days"].mean() if len(prior) else np.nan
    audit_ok &= (len(prior) == r["cust_n"]) and np.isclose(expected, r["cust_avg_delay"], equal_nan=True)
print(f"Leakage audit passed on 300 random invoices: {audit_ok}")
assert audit_ok, "Feature leakage detected!"

FEATURES = ["log_usd", "term_days", "days_to_due", "base_gap", "post_month", "post_dow", "post_dom",
            "due_dow", "due_month_end", "business_code_enc", "payment_terms_enc",
            "cust_avg_delay", "cust_late_rate", "cust_n"]

# ----------------------------------------------------------------------------
# 9. TRAIN / TEST SPLIT (time-ordered: train on the past, test on the future)
# ----------------------------------------------------------------------------
main_data = main_data.sort_values("posting_date").reset_index(drop=True)
train_all, test_df = train_test_split(main_data, test_size=0.2, shuffle=False)  # no shuffling
train_all, test_df = train_all.copy(), test_df.copy()

# New customers have no history yet: fill with the TRAINING averages
fill_avg = train_all["delay_days"].mean()
fill_late = (train_all["delay_days"] > 0).mean()
for frame in (train_all, test_df, null_data):
    frame["cust_avg_delay"] = frame["cust_avg_delay"].fillna(fill_avg)
    frame["cust_late_rate"] = frame["cust_late_rate"].fillna(fill_late)

# At the test start, outcomes of invoices paid AFTER it are not yet known: drop them from training
train_df = train_all[train_all["clear_date"] < test_df["posting_date"].min()]
print(f"train {len(train_df):,} ({train_df.posting_date.min().date()} to {train_df.posting_date.max().date()}) | "
      f"test {len(test_df):,} ({test_df.posting_date.min().date()} to {test_df.posting_date.max().date()})")

X_train, y_train = train_df[FEATURES], train_df["delay_days"]
X_test, y_test = test_df[FEATURES], test_df["delay_days"]

# ----------------------------------------------------------------------------
# 10. MODELLING: baselines vs ML models
# ----------------------------------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=8, min_samples_leaf=20, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=42),
    "XGBoost": xgb.XGBRegressor(objective="reg:absoluteerror", n_estimators=500, learning_rate=0.03,
                                max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42),
    "LightGBM": lgb.LGBMRegressor(objective="l1", n_estimators=500, learning_rate=0.03, num_leaves=31,
                                  min_child_samples=40, subsample=0.8, subsample_freq=1,
                                  colsample_bytree=0.8, random_state=42, verbose=-1),
}
predictions = {"Baseline: pays on due date": np.zeros(len(y_test)),
               "Baseline: customer's average delay": X_test["cust_avg_delay"].values}
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions[name] = model.predict(X_test)

results = []
for name, pred in predictions.items():
    results.append({"Model": name,
                    "MAE (days)": mean_absolute_error(y_test, pred),
                    "RMSE (days)": np.sqrt(mean_squared_error(y_test, pred)),
                    "R2": r2_score(y_test, pred),
                    "Within 3 days": (np.abs(y_test - pred) <= 3).mean()})
results = pd.DataFrame(results).sort_values("MAE (days)").reset_index(drop=True)
print(results.round(3).to_string(index=False))

best_name = results[~results["Model"].str.startswith("Baseline")].iloc[0]["Model"]
print(f"\nBest model: {best_name}")

plt.figure(figsize=(9, 4.5))
plt.barh(results["Model"][::-1], results["MAE (days)"][::-1],
         color=["#1b4965" if m == best_name else "#b0b0b0" for m in results["Model"][::-1]])
plt.xlabel("Mean absolute error on test set (days, lower is better)")
plt.title("Model comparison")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/02_model_comparison.png", dpi=120)
plt.close()

# ----------------------------------------------------------------------------
# 11. PREDICT OPEN INVOICES (final model refit on ALL paid invoices) + AGING BUCKET
# ----------------------------------------------------------------------------
final_model = models[best_name]
all_paid = pd.concat([train_all, test_df])
final_model.fit(all_paid[FEATURES], all_paid["delay_days"])

null_data["pred_delay_days"] = final_model.predict(null_data[FEATURES]).round(0).astype(int)
null_data["pred_clear_date"] = null_data["due_in_date"] + pd.to_timedelta(null_data["pred_delay_days"], unit="D")

importance = pd.Series(
    final_model.feature_importances_,
    index=FEATURES
).sort_values(ascending=False)

print(importance)

# Aging bucket = how many days after the due date the invoice is expected to be paid
bins = [-np.inf, 0, 15, 30, 45, 60, np.inf]
labels = ["On time / early", "1-15 days late", "16-30 days late", "31-45 days late", "46-60 days late", "60+ days late"]
null_data["aging_bucket"] = pd.cut(null_data["pred_delay_days"], bins=bins, labels=labels)
main_data["aging_bucket"] = pd.cut(main_data["delay_days"], bins=bins, labels=labels)  # actual, for comparison

aging = (null_data.groupby("aging_bucket", observed=False)
         .agg(open_invoices=("doc_id", "size"), open_amount_usd=("converted_usd", "sum")).round(0))
aging["actual_history_%"] = (main_data["aging_bucket"].value_counts(normalize=True)
                             .reindex(labels) * 100).round(1).values
print(aging)

# ----------------------------------------------------------------------------
# 12. DAYS SALES OUTSTANDING (DSO)
# ----------------------------------------------------------------------------
# DSO = receivables at month end / sales posted in the month * days in the month (all amounts in USD)
# Only months with complete data: the extract starts Dec 2018, and every open invoice is from Feb 2020 on
dso_rows = []
for month in pd.period_range("2019-03", "2020-01", freq="M"):
    month_end = month.end_time.normalize()
    sales = df.loc[
        (df["posting_date"] >= month.start_time) &
        (df["posting_date"] <= month_end),
        "converted_usd"
    ].sum()
    receivables = df.loc[
        (df["posting_date"] <= month_end)&
        (df["clear_date"].isna() | (df["clear_date"] > month_end)),
        "converted_usd"
    ].sum()
    dso_rows.append({"month": str(month), "sales_usd": sales, "receivables_usd": receivables,
                     "DSO_days": receivables / sales * month.days_in_month})
dso = pd.DataFrame(dso_rows)
print(dso.round(1).to_string(index=False))
days_to_pay = (main_data["clear_date"] - main_data["posting_date"]).dt.days
print(f"\nMedian monthly DSO: {dso['DSO_days'].median():.1f} days | mean: {dso['DSO_days'].mean():.1f} days")
print(f"Average days to pay (weighted by USD amount): {np.average(days_to_pay, weights=main_data['converted_usd']):.1f} days")

fig, ax = plt.subplots(1, 2, figsize=(15, 4.5))
aging_plot = aging[["open_invoices"]].copy()
aging_plot["open_invoices"] = aging_plot["open_invoices"] / aging_plot["open_invoices"].sum() * 100
aging_plot["history"] = aging["actual_history_%"]
aging_plot.rename(columns={"open_invoices": "Predicted (open invoices)", "history": "Actual (paid invoices)"}).plot.bar(
    ax=ax[0], color=["#1b4965", "#b0b0b0"])
ax[0].set(title="Aging bucket distribution (% of invoices)", xlabel="", ylabel="%")
ax[0].tick_params(axis="x", rotation=25)
ax[1].plot(dso["month"], dso["DSO_days"], marker="o", color="#2a6f97")
ax[1].axhline(dso["DSO_days"].median(), color="grey", ls="--", lw=1, label="median")
ax[1].set(title="Monthly DSO (USD)", ylabel="days")
ax[1].tick_params(axis="x", rotation=45)
ax[1].legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/03_aging_and_dso.png", dpi=120)
plt.close()

# ----------------------------------------------------------------------------
# 13. SAVE OUTPUTS
# ----------------------------------------------------------------------------
df.to_csv(f"{OUT_DIR}/cleaned_dataset.csv", index=False)
dso.to_csv(f"{OUT_DIR}/monthly_dso.csv", index=False)
results.to_csv(f"{OUT_DIR}/model_comparison.csv", index=False)
open_cols = ["doc_id", "cust_num_norm", "customer_label", "business_code", "converted_usd", "due_in_date",
             "pred_delay_days", "pred_clear_date", "aging_bucket"]
null_data[open_cols].sort_values("converted_usd", ascending=False).to_csv(
    f"{OUT_DIR}/open_invoice_predictions.csv", index=False)
print(f"\nSaved {len(null_data):,} open-invoice predictions and charts in ./{OUT_DIR}/")
