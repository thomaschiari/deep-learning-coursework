# Classification Project

The complete source code for this project is available [here](https://github.com/thomaschiari/deep-learning-classification-project).

## 1. Dataset Selection

**Name**: Bank Marketing Dataset

**Source**: UCI Machine Learning Repository [available here](https://archive.ics.uci.edu/dataset/222/bank%2Bmarketing)

**Size**: 41188 rows, with 20 features and 1 target variable

**Task**: Binary classification, predicting whether a client will subscribe to a term deposit

**Why this dataset?** 

We wanted a real-world, business-relevant classification problem with enough rows and feature diversity to justify an MLP. The Bank Marketing dataset provides 41k examples with both categorical and numeric attributes (client profile + campaign context), enabling preprocessing (one-hot + scaling) and meaningful model comparisons. The target is notably imbalanced (~89% “no”, ~11% “yes”), making it more complex than a balanced dataset. The dataset also has no missing values, and is a public academic dataset.

## 2. Dataset Explanation

The dataset represents records from telemarketing campaigns of a Portuguese bank. The goal is to predict whether the client will subscribe to a term deposit. The file `bank-additional-full.csv` contains 41,188 examples and 20 inputs. There are no missing values.

The target is column `y`. It is a binary variable, indicating whether the client will subscribe to a term deposit, and is imbalanced (~89% “no”, ~11% “yes”).

**Features**:

- `age`: numeric

![Hist Age](images/hist_age.png)

- `job`: categorical, indicating the type of job of the client

![Bar Job](images/bar_job.png)

- `marital`: categorical, indicating the marital status of the client

![Bar Marital](images/bar_marital.png)

- `education`: categorical, indicating the education level of the client

![Bar Education](images/bar_education.png)

- `default`: categorical, indicating if the client has credit in default

![Bar Default](images/bar_default.png)

- `housing`: categorical, indicating if the client has a housing loan

![Bar Housing](images/bar_housing.png)

- `loan`: categorical, indicating if the client has a personal loan

![Bar Loan](images/bar_loan.png)

- `contact`: categorical, indicating the communication type (e.g. telephone, cellular...)

![Bar Contact](images/bar_contact.png)

- `month`: categorical, indicating the last contact month of the year

![Bar Month](images/bar_month.png)

- `day_of_week`: categorical, indicating the last contact day of the week

![Bar Day of Week](images/bar_day_of_week.png)

- `duration`: numeric, indicating the last call duration in seconds (this feature is dropped because it leaks the outcome, e.g. `duration=0` → always `y="no"`)

- `campaign`: numeric, indicating the number of contacts in the current campaign

![Hist Campaign](images/hist_campaign.png)

- `pdays`: numeric, indicating the number of days that passed by after the client was last contacted (999 means not previously contacted)

![Hist Pdays](images/hist_pdays.png)

- `previous`: numeric, indicating the number of contacts before the current campaign

![Hist Previous](images/hist_previous.png)

- `poutcome`: categorical, indicating the outcome of the previous marketing campaign

![Bar Poutcome](images/bar_poutcome.png)

- `emp.var.rate`: numeric, indicating the employment variation rate

![Hist Emp Var Rate](images/hist_emp.var.rate.png)

- `cons.price.idx`: numeric, indicating the consumer price index

![Hist Cons Price Idx](images/hist_cons.price.idx.png)

- `cons.conf.idx`: numeric, indicating the consumer confidence index

![Hist Cons Conf Idx](images/hist_cons.conf.idx.png)

- `euribor3m`: numeric, indicating the euribor 3 month rate

![Hist Euribor3m](images/hist_euribor3m.png)

- `nr.employed`: numeric, indicating the number of employees

![Hist Nr Employed](images/hist_nr.employed.png)

The last 5 features are macroeconomic indicators, important for context.

**Potential issues to address**

- Imbalance: the target is imbalanced (~89% “no”, ~11% “yes”).

**Summary statistics and visuals**

Class distribution of the target:

![Class distribution](images/class_distribution.png)

Correlation matrix of the numeric features:

![Correlation matrix](images/correlation_matrix.png)

Summary statistics of the numeric features:

![Summary statistics](images/summary_statistics.png)

We additionally created a new feature `prev_contacted`, indicating if the client was previously contacted by treating the `pdays` feature as 1 if the client was previously contacted, and 0 otherwise.





---

*Note*: Artificial Intelligence was used in this exercise for code completion and review, as well as for text revision and refinement.


