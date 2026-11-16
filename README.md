# Invoice Payment Forecasting: Delay Prediction and Aging Buckets

-Predicts when each customer invoice will actually be paid
-Groups open invoices into aging buckets, and 
-Tracks Days Sales Outstanding (DSO), using invoice-level accounts receivable data.

## Key findings

- The best model (XGBoost) predicts payment timing with an average error of **2.58 days**: **40% better** than 
  assuming customers pay on the due date, and **24% better** than using each customer's average delay.
- About **80%** of test invoices are predicted within ±3 days of the actual payment date.
- Of the 9,681 open invoices ($312.5M), the model expects about 65% to be paid on time or early and 
  about 34% to be 1 to 15 days late. Only 83 (0.9%) are expected to be later than that.
- Monthly DSO is steady at about **17.7 days** (median). December 2019 shows 3.2 days, 
  which is a year-end timing effect and not a collections' improvement.

## Business problem

Finance teams plan cash around invoice due dates, but customers rarely pay exactly on time. 
Knowing which invoices will be late, and by how much, improves cash forecasts and 
helps collectors decide where to focus.

## Data

Invoice-level accounts receivable data: 50,000 rows, Dec 2018 to May 2020, in USD and CAD. 
Invoices with no clear date (payment date) are treated as open.

After cleaning: **48,833 invoices**, of which **39,152 are paid** (used for learning) 
and **9,681 are open** (scored by the model). 41.9% of paid invoices were paid late; 
the median delay is 0 days, the 95th percentile is 10 days late and the 99th percentile is 50 days late.

Data Source: Kaggle

## Approach

1. **Cleaning.** Removed 1,161 duplicate rows and 6 non-standard documents (type `X2`). 
   Dropped empty, constant and duplicated columns, including `isOpen`, which only repeats "clear date is empty" and 
   would leak the answer. Parsed dates.
2. **Currency.** CAD amounts converted to USD at 0.75 (approximate 2019 average rate) so amounts are comparable.
3. **Customers.** Customer numbers are normalised by removing leading zeros 
   (`0100000217` and `100000217` are the same customer), reducing 1,424 numbers to 1,098 customers. 
   Each customer gets a readable label (most frequent name plus number).
4. **Target.** Payment delay in days = clear date minus due date (negative means paid early).
5. **Features (14), all known when the invoice is posted:**
   - Invoice: log USD amount, payment-term days, days from posting to due date, posting-to-baseline gap, 
     posting month / weekday / day of month, due-date weekday, due on month end, business code, payment terms.
   - Customer history: average delay, share of invoices paid late, and number of previously paid invoices.
6. **Leakage control.** Customer history for each invoice uses only that customer's invoices 
   *cleared strictly before it was posted*. 
   A built-in audit recomputes this slowly for 300 random invoices and stops if any differ.
7. **Time-ordered split.** `train_test_split(shuffle=False)` on invoices sorted by posting date, 
   so the model trains on the past and is tested on the future:
   - Train: 29,595 invoices (30 Dec 2018 to 15 Nov 2019)
   - Test: 7,831 invoices (19 Nov 2019 to 27 Feb 2020)
   - Training invoices paid after the test period starts are excluded, because their outcomes would not yet be known.
8. **Models.** Linear Regression, Decision Tree, Random Forest, XGBoost and LightGBM, compared against two baselines. 
   The best model by test MAE is refit on all paid invoices and used to score the open invoices.
9. **Aging buckets and DSO** (see below).

## Results

Test set (7,831 invoices):

| Model                              | MAE (days) | RMSE (days) | R²    | Within 3 days |
|------------------------------------|------------|-------------|-------|---------------|
| **XGBoost**                        | **2.576**  | 7.114       | 0.443 | 79.8%         |
| LightGBM                           | 2.585      | 7.148       | 0.438 | 79.6%         |
| Random Forest                      | 2.940      | 7.046       | 0.454 | 75.0%         |
| Decision Tree                      | 3.091      | 7.190       | 0.431 | 75.6%         |
| Baseline: customer's average delay | 3.371      | 7.726       | 0.343 | 71.5%         |
| Linear Regression                  | 3.626      | 7.355       | 0.405 | 64.3%         |
| Baseline: pays on due date         | 4.263      | 9.535       | 0.000 | 63.9%         |


MAE (mean absolute error) is the main metric. XGBoost and LightGBM optimise absolute error, 
while RMSE and R² are dominated by a few very late invoices, 
which is why Random Forest scores slightly better on those two.

[Model comparison](outputs/02_model_comparison.png)

### Aging buckets

Each open invoice is placed in a bucket by its predicted days after the due date. 
The last column shows the share of paid invoices that actually fell in each bucket.

| Bucket          | Open invoices | Open USD | Actual history |
|-----------------|---------------|----------|----------------|
| On time / early | 6,324         | $188.04M | 58.1%          |
| 1-15 days late  | 3,274         | $123.11M | 38.4%          |
| 16-30 days late | 23            | $0.5M    | 1.4%           |
| 31-45 days late | 31            | $0.6M    | 1.0%           |
| 46-60 days late | 29            | $0.3M    | 0.6%           |
| 60+ days late   | 0             | $0.0M    | 0.5%           |

### DSO

DSO = receivables at month end ÷ sales posted in the month × days in the month, in USD. 
Shown for March 2019 to January 2020, the months with complete data.

| Month   | Sales ($M)  | Receivables ($M) | DSO (days) |
|---------|-------------|------------------|------------|
| 2019-03 | 94.1        | 56.0             | 18.5       |
| 2019-04 | 91.6        | 56.7             | 18.6       |
| 2019-05 | 92.1        | 48.6             | 16.4       |
| 2019-06 | 92.4        | 59.2             | 19.2       |
| 2019-07 | 84.4        | 48.4             | 17.8       |
| 2019-08 | 98.5        | 56.1             | 17.6       |
| 2019-09 | 89.1        | 56.7             | 19.1       |
| 2019-10 | 98.2        | 56.6             | 17.9       |
| 2019-11 | 97.1        | 56.2             | 17.4       |
| 2019-12 | 60.4        | 6.3              | **3.2**    |
| 2020-01 | 80.6        | 43.1             | 16.6       |

Median monthly DSO is **17.8 days**; the amount-weighted average time from posting to payment is **17.4 days**.

**December 2019 is a timing effect.** Only 1,768 invoices were posted (about 3,000 in a typical month), 
and just 23% of them after the 15th (47% to 55% in other months). Collections that month were the highest of any ($110M), 
so receivables fell from about $54M on Dec 15 to $6.3M at year-end, then rebuilt to $43M by Jan 31. 
The data cannot show *why* (a holiday billing lull and a year-end payment push are plausible), 
so the median is used as the headline figure.

[Aging buckets and DSO](outputs/03_aging_and_dso.png)

## Limitations

- **Distribution shift.** The open invoices were posted Mar to May 2020, the start of COVID-19, 
  while the model learned mostly from pre-pandemic payment behaviour. Error already rises in the most recent test month.
- **The tail is under-predicted.** The model predicts the typical case, so it rarely predicts very late payments 
  (83 open invoices beyond 15 days, none beyond 60, against about 3.5% of paid invoices historically). 
  Aging buckets from point predictions should be read as the expected case, not worst case.
- **No separate validation set or tuning.** Hyperparameters are fixed, 
  and the best model is picked on the same test window it is reported on. 
  XGBoost and LightGBM differ by 0.009 days, so the choice between them is not meaningful.
- **CAD invoices** (about 8% of the data) behave differently: they were paid 6.1 days late on average versus 0.4 for USD. 
  A single fixed exchange rate is used.
- **"Open" means a blank clear date** in this extract, and invoice amounts are used as a proxy for credit sales in the DSO calculation.
- **New customers** with no payment history are predicted using training-set averages.

## How to run

```bash
pip install -r requirements.txt
python invoice_payment_forecasting.py
```
Place the dataset next to the script as `dataset_project.csv`. 
The script creates an `outputs/` folder with all charts and CSV files. 
The date-parsing lines must match the file's date layout (for example, `dayfirst=True` for day-first dates such as `30-12-2019`).

## Project structure

```
invoice-payment-forecasting/
├── README.md
├── invoice_payment_forecasting.py
├── requirements.txt
├── dataset_project.csv
└── outputs/
```

## Outputs

| File                                  | Contents                                                                       |
|---------------------------------------|--------------------------------------------------------------------------------|
| `outputs/01_eda.png`                  | Delay distribution, median delay by due-date weekday, slowest-paying customers |
| `outputs/02_model_comparison.png`     | Test MAE for every model and baseline                                          |
| `outputs/03_aging_and_dso.png`        | Aging bucket distribution and monthly DSO                                      |
| `outputs/open_invoice_predictions.csv` | Predicted delay, predicted payment date and aging bucket for each open invoice |
| `outputs/model_comparison.csv`        | Full model results table                                                       |
| `outputs/monthly_dso.csv`             | Monthly sales, receivables and DSO                                             |
| `outputs/cleaned_dataset.csv`         | Cleaned invoice data                                                           |

## Possible next steps

- Prediction intervals (for example 10th and 90th percentile models) so forecasts report a range.
- A cash flow forecast and collections prioritisation built on `open_invoice_predictions.csv`.
- A dashboard of customer risk, aging and DSO.
- Customer segmentation by payment behaviour.
- Time-series cross-validation and hyperparameter tuning with a separate validation window.
