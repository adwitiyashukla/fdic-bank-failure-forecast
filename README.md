# fdic-bank-failure-forecast

Which US banks will fail in the next 12 months? This project scores every FDIC insured bank at every quarter end from 1985 to June 2026, using only what that quarter's call report showed. Then it checks the scores against the banks the FDIC actually closed, in the 2008 crisis, in the quiet years after it, and in 2023.

## Data

Both sources are public and need no account or key.

| source | what I use |
|---|---|
| FDIC BankFind API | quarterly financials for every insured bank from 1984 Q1 to 2026 Q2 (1,682,616 bank quarters) and the failed bank list (4,117 records) |
| Federal Reserve 2026 stress test files | quarterly unemployment, house prices and commercial real estate prices since 1976, plus the final baseline and severely adverse scenarios |

A row is one bank at one quarter end. Its label is 1 if the FDIC closed that bank within the next four quarters. The last four quarters are not labeled yet because their year is still running. I also leave out 1,459 rows that come right before an FDIC assistance deal, since those banks did not fail but did not make it on their own either. From 1985 on that leaves 1,610,856 rows for 23,568 banks, with 2,550 closed banks matched to a report from the two years before they closed and 10,454 positive rows.

## Features

Each bank quarter gets 51 features about capital, bad loans, earnings, the loan mix, growth, size and age, funding, and how many banks in the same state failed over the past year. One of them is the Texas ratio, the old rule of thumb for bank trouble. It divides bad loans plus foreclosed property by tangible equity plus loan loss reserves. When that denominator is zero or negative the ratio is capped at 1000 instead of dropped, because those are the banks that matter most.

Growth features compare a bank with itself exactly four quarters earlier, matched on the quarter number and not on row position, so a missing filing cannot pair up the wrong quarters. A test rewrites every value after a cutoff date and checks that no feature before the cutoff changes.

## Models

| model | what it sees |
|---|---|
| Texas ratio | the ratio alone, no training |
| logistic regression | 11 standard measures, clipped at the training quantiles, missing values filled and flagged |
| gradient boosting | all 51 features, scikit-learn HistGradientBoostingClassifier |

The boosting model has monotone constraints, so more capital can never raise a bank's risk and more bad loans can never lower it. It picks its number of trees on the most recent years of its own training window and then refits on the whole window. I started with the built-in random validation split, but it never stopped early. Quarters from the same bank landed on both sides of the split, and the extra trees made the crisis test worse.

## Results

The crisis test trains on reports up to the end of 2005, so every label it learns from was known by the end of 2006. It then scores every report from the end of 2006 to the end of 2011, which is 172,179 bank quarters and 467 banks that failed.

| model | AUC | 95% interval | caught in top 1% | caught in top 5% |
|---|---|---|---|---|
| Texas ratio | 0.967 | 0.958 to 0.973 | 45.7% | 86.3% |
| logistic regression | 0.955 | 0.945 to 0.962 | 44.0% | 82.2% |
| gradient boosting | 0.981 | 0.975 to 0.985 | 50.2% | 91.4% |

Caught in top 5% means that each quarter the riskiest 5% of banks get flagged, and the number is the share of failure rows that were flagged. The intervals come from 200 bootstrap samples of whole banks.

The recent test trains up to the end of 2013 and scores every report from the end of 2014 to mid 2025. That is 229,388 bank quarters but only 40 failed banks, so the intervals are wide.

| model | AUC | 95% interval | caught in top 1% | caught in top 5% |
|---|---|---|---|---|
| Texas ratio | 0.841 | 0.740 to 0.933 | 67.2% | 73.0% |
| logistic regression | 0.919 | 0.856 to 0.971 | 69.3% | 74.5% |
| gradient boosting | 0.908 | 0.835 to 0.978 | 71.5% | 75.9% |

In the recent test the boosting model expected 136.1 failure rows and 137 happened. In the crisis test it expected about 890 and 1,801 happened, because nothing between 1985 and 2005 looked like 2008. At the Q3 2008 reports it expected 15 banks to fail within a year and 106 did. The level was off, but the ranking held up, which is why its top 5% still caught 91%.

The rolling test retrains once a year from 1990 to 2025 and scores each year's reports with a model that only knew the outcomes available at the start of that year. Boosting has a median AUC of 0.977 and beats the Texas ratio in 30 of the 34 years that had failures. Its worst year is 2022 at 0.41. The failures in that year's window are the 2023 ones, and for most of 2022 the model rated SVB, Signature and First Republic as safer than the typical bank.

For warning time, the crisis test flags the riskiest 5% every quarter. 98% of the failed banks were flagged at their last report, and 81% had been flagged without a break for at least four quarters before they failed. The median was six quarters.

## What the 2023 failures showed

At the end of 2022 the boosting model ranked SVB 3,792nd out of 4,773 banks. First Republic was 3,944th and Signature 1,222nd. All three had normal capital and almost no bad loans, which is what almost 40 years of failures taught the model to look for. A boosting model limited to credit features, the logistic model and the Texas ratio ranked them about the same.

Their problem was on the funding side. 94% of SVB's deposits were above the insurance limit, and the unrealized losses on its securities were 114% of its equity. A screen that multiplies the percentile ranks of just those two numbers put SVB 10th, Signature 208th and First Republic 440th. Republic Bank in Philadelphia, which failed in April 2024, was 5th. No failure before 2023 looked like this, so a model trained on the past could not learn it. That is why the screen sits next to the model instead of inside it.

| bank | closed | boosting | Texas ratio | run screen |
|---|---|---|---|---|
| Silicon Valley Bank | 2023-03-10 | 3,792 | 3,043 | 10 |
| Signature Bank | 2023-03-12 | 1,222 | 1,891 | 208 |
| First Republic Bank | 2023-05-01 | 3,944 | 3,279 | 440 |
| Heartland Tri-State Bank | 2023-07-28 | 824 | 434 | 583 |
| Citizens Bank, Sac City, IA | 2023-11-03 | 28 | 23 | 313 |
| Republic Bank, Philadelphia | 2024-04-26 | 244 | 407 | 5 |
| First National Bank of Lindsay | 2024-10-18 | 723 | 352 | 4,002 |

Ranks among the 4,773 banks at the end of 2022, where 1 is the riskiest.

## The 2026 failures

A model trained only on outcomes known by September 2025 scored every report filed after that. The four banks the FDIC closed in 2026 were in its top 22 at every report they filed before closing, out of about 4,400 banks each quarter.

| bank | closed | 2025 Q3 | 2025 Q4 | 2026 Q1 |
|---|---|---|---|---|
| Metropolitan Capital B&T | 2026-01-30 | 3 | 2 | |
| Community Bank and Trust, West Georgia | 2026-05-01 | 2 | 6 | 1 |
| Kentland FS&LA | 2026-07-10 | 1 | 3 | 2 |
| Small Business Bank | 2026-07-17 | 22 | 8 | 11 |

## Stress view

A second boosting model also sees how unemployment, house prices and commercial real estate prices move over the next eight quarters, and it predicts failures within those eight quarters. I ran the June 2026 banks through three paths. The Fed paths start from the end of 2025, where the Fed's history ends.

| path | unemployment | house prices | CRE prices | failures expected in 8 quarters | 90% range |
|---|---|---|---|---|---|
| Fed 2026 baseline | +0.1 points | +1.3% | +9.2% | 4.9 | 2 to 8 |
| Fed 2026 severely adverse | +5.5 points | -29.7% | -38.8% | 18.6 | 13 to 25 |
| What happened, 2008 Q3 to 2010 Q3 | +3.9 points | -9.6% | -25.3% | 15.8 | 11 to 22 |

The backtest says to read these as a floor. Trained on data up to 2004 and given the real 2007 to 2010 path, the stress model expected 88 failures within two years of the Q1 2009 reports. Without the path it expected 74, and 302 happened. The path helps, but a model that never saw a housing crash still misses most of one. Today's banks also start in better shape. Only 8 banks have a Texas ratio above 100% now, against 31 at the end of 2007 and 118 by Q3 2008.

## Latest quarter

With the June 2026 reports the boosting model expects 3.1 failures among 4,313 banks over the next year (90% range 1 to 6), and 15 banks are above 1%. The demo only shows totals for banks that are still open. The export step fails if any demo table names a bank that has not failed, and the full list stays in the local results folder.

## Running it

On Windows:

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python -m bankfail download
.venv\Scripts\python -m bankfail all
.venv\Scripts\python -m streamlit run app.py
```

The download pulls about 360 MB of quarter files from the FDIC API in about 17 minutes and skips quarters it already has. The all command runs panel, evaluate, rolling, case, current, stress and export in that order, and each step can also run on its own, like `python -m bankfail rolling`. The full run takes about an hour on my laptop, a 12 thread Intel Core 5 120U with 8 GB of RAM. Every date, horizon and model setting is in config.toml and gets checked when it loads.

`python -m bankfail synth` writes a synthetic dataset in the same format as the download. CI uses it to run the whole pipeline without calling the FDIC.

On Linux or Mac, use `.venv/bin/` instead of `.venv\Scripts\`.

Boosting is capped at four threads by `threads` in the gbm section of config.toml. Letting it use all twelve made the test suite about four times slower on my laptop.

scikit-learn is held below 1.9. Version 1.9 changed how the boosting model bins weighted rows, which shifts the numbers above and made the test suite about 8 times slower.

## Tests

71 pytest tests. They check the labels around failure dates, that no feature can see the future, the monotone constraints, features that have no data yet in older training windows, the time based validation split, paging and retries against a fake FDIC API, and config validation. One test runs the whole pipeline on synthetic data and then renders every tab of the Streamlit app.

## Layout

```
bankfail/download.py   FDIC API and Fed files with paging, retries and row checks
bankfail/panel.py      quarter files into one panel, failure labels
bankfail/features.py   the 51 features
bankfail/models.py     logistic regression and boosting with time based early stopping
bankfail/metrics.py    capture rates, warning time, bootstrap intervals
bankfail/evaluate.py   crisis, recent and rolling tests
bankfail/cases.py      the 2023 check, the run screen and the latest quarter
bankfail/stress.py     the eight quarter stress model and the Fed scenarios
bankfail/export.py     demo tables and the check that no open bank is named
bankfail/synth.py      synthetic data for tests and CI
bankfail/pipeline.py   runs the steps and writes results
app.py                 the Streamlit demo
config.toml            every setting, validated in bankfail/config.py
```

## License

MIT
