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

## 3. Data Preprocessing

### Treating missing values and duplicates

The first step in order to clean and preprocess the data was to treat the variable `pdays` for when there was no previous contact. For that, we used the created `prev_contacted` feature to filter the rows where the client was not previously contacted, then imputed the median value of `pdays`. This is the only feature with "missing" values, and we chose to fill with the median value to avoid introducing new labels.

```python
df.loc[df["prev_contacted"] == 0, "pdays"] = np.nan

imputer = SimpleImputer(strategy='median')
df[numeric_features] = imputer.fit_transform(df[numeric_features])
```

The next step was to drop the duplicates. for that, we used the `drop_duplicates` function: 

```python
before_n = len(df)
df = df.drop_duplicates()
after_n = len(df)
dedup_removed = before_n - after_n
dedup_removed
```

With that, we removed 1784 rows that were exactly the same. 

### Scaling numerical features

The next step was to scale numerical features. We applied standardization (z-score), which centers features around 0 with standard deviation of 1. This was made in order to remove the effect of different scales between features, equalizing variances, making the data robust to outliers. Min-max scaling was also considered, but it is highly sensitive to outliers. A few extremes may squash most data into a narrow range. 

```python
for c in numeric_features:
    df[c] = pd.to_numeric(df[c], errors='coerce')

scaler = StandardScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])
```

With that, we keep the distribution of the data, but the values are normalized relative to other features. Here is a histogram of the `age` feature after scaling:

![Age after scaling](images/age_distribution_after_scaling.png)

### Encoding categorical features

We used one-hot encoding for all categorical features. We kept all information, avoiding the creation of a reference level. This is because neural networks are not sensitive to multicollinearity in the same way as linear models. We chose one-hot because it encodes each category as a binary feature, avoiding the creation of an order between the categories.

```python
X_cat = pd.get_dummies(df[categorical_features], drop_first=False, dtype=np.float32)

processed = pd.concat(
    [df[numeric_features].astype(np.float32), X_cat],
    axis=1,
)

processed.insert(0, "y", df["y"].astype(np.int8))
```

Finally, we saved the processed data into a CSV file in order to use it as input for the MLP in the next step.

## 4. MLP Implementation

Now, we are going to implement a multi-layer perceptron (MLP) from scratch using Numpy operations, trained with mini-batch SGD and cross-entropy. The model supports an arbitrary number of hidden layers selectable via CLI (argument parser), and supports `relu`, `tanh` and `sigmoid` activations. 

We train on the data that was the output of the previous step, where the first column is the target and all remaining columns are the treated features. We use a split of 70% for training, 15% for validation and 15% for testing in the training loop. 

How to run the MLP (in the source code repository):

```bash
python src/mlp_numpy.py --data data/clean/bank-additional-full-post-preprocessed.csv --hidden 64,64 --activation relu --epochs 30 --lr 0.05 --batch_size 256 --seed 42
```

### Hyperparameters

- Hidden Layers: selectable via CLI (size of each layer). Default is `64`.

- Activation Functions: selectable via CLI (either `relu`, `tanh` or `sigmoid`). Default is `relu`.

- Epochs: selectable via CLI (number of epochs). Default is `30`.

- Learning Rate: selectable via CLI (learning rate). Default is `0.05`.

- Batch Size: selectable via CLI (batch size). Default is `256`.

- Seed: selectable via CLI (random seed). Default is `42`.

### Architecture

The model is a feed-forward network: input -> hidden layers -> output logits -> softmax. Here is the initial implementation of the MLP class:

```py
class MLP:
    def __init__(self, input_dim, hidden_layers, num_classes, activation="relu", seed=42, l2=0.0):
        self.activation = activation
        self.l2 = float(l2)
        rng = np.random.default_rng(seed)

        sizes = [input_dim] + list(hidden_layers) + [num_classes]
        self.W = []
        self.b = []
        for i in range(len(sizes)-1):
            fan_in, fan_out = sizes[i], sizes[i+1]
            if activation == "relu":
                W = rng.normal(0.0, np.sqrt(2.0/fan_in), size=(fan_in, fan_out)).astype(np.float32)
            else:  
                W = rng.normal(0.0, np.sqrt(1.0/fan_in), size=(fan_in, fan_out)).astype(np.float32)
            b = np.zeros((1, fan_out), dtype=np.float32)
            self.W.append(W); self.b.append(b)
```

### Activation functions

Activation functions introduce non-linearity to the model, allowing it to learn non-linear boundaries. We support three activation functions: `relu`, `tanh` and `sigmoid`. Here is the implementation of the activation functions:

```py
# activation functions
def act_forward(z, kind):
    if kind == "relu":   
        return np.maximum(0, z)
    if kind == "tanh":   
        return np.tanh(z)
    if kind == "sigmoid":
        return 1.0 / (1.0 + np.exp(-z))
    raise ValueError("activation must be relu/tanh/sigmoid")

def act_backward(z, a, kind):
    if kind == "relu":    
        return (z > 0).astype(z.dtype)
    if kind == "tanh":    
        return 1.0 - a*a
    if kind == "sigmoid": 
        return a * (1.0 - a)
    raise ValueError("activation must be relu/tanh/sigmoid")
``` 

The forward pass per layer is implemented as follows:

```py
def forward(self, X):
    """Returns probs and caches for backprop"""
    A = X
    caches = []  
    L = len(self.W)
    for l in range(L):
        Z = A @ self.W[l] + self.b[l]   
        if l < L-1:
            A_next = act_forward(Z, self.activation)
        else:
            A_next = Z
        caches.append((A, Z, A_next))
        A = A_next
    probs = softmax(A)  
    return probs, caches
```

And the softmax function, for the output layer, is implemented as follows:

```py
def softmax(z):
    z = z - z.max(axis=1, keepdims=True) 
    ez = np.exp(z)
    return ez / (ez.sum(axis=1, keepdims=True) + 1e-12)
```

### Loss function

The loss function is the cross-entropy on the softmax probabilities, calculated as follows:

```py
tr_loss = -np.log(tr_probs[np.arange(Xtr.shape[0]), ytr] + 1e-12).mean()
va_loss = -np.log(va_probs[np.arange(Xva.shape[0]), yva] + 1e-12).mean()
```

### Optimizer

The optimizer is a mini-batch SGD: we shuffle the data, take batches, compute the gradients, backpropagate and update the weights. Here is the implementation of the complete training loop:

```py
# train loop
def train(model, Xtr, ytr, Xva, yva, epochs=30, lr=0.05, batch_size=256, seed=42):
    rng = np.random.default_rng(seed)
    n = Xtr.shape[0]
    for epoch in range(1, epochs+1):
        # mini-batch SGD
        idx = rng.permutation(n)
        for start in range(0, n, batch_size):
            b = idx[start:start+batch_size]
            probs, caches = model.forward(Xtr[b])
            dW, db = model.backward(probs, ytr[b], caches)
            model.step(dW, db, lr)
        # metrics
        tr_pred, tr_probs = model.predict(Xtr)
        va_pred, va_probs = model.predict(Xva)
        tr_acc = accuracy(ytr, tr_pred)
        va_acc = accuracy(yva, va_pred)
        tr_loss = -np.log(tr_probs[np.arange(Xtr.shape[0]), ytr] + 1e-12).mean()
        va_loss = -np.log(va_probs[np.arange(Xva.shape[0]), yva] + 1e-12).mean()
        print(f"epoch {epoch:03d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {va_loss:.4f} acc {va_acc:.4f}")
```

### Final model implementation

The final model includes a CLI to parse the arguments, load the processed CSV, make the stratification, build the model, train, and evaluate on test:

```py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# activation functions
def act_forward(z, kind):
    if kind == "relu":   
        return np.maximum(0, z)
    if kind == "tanh":   
        return np.tanh(z)
    if kind == "sigmoid":
        return 1.0 / (1.0 + np.exp(-z))
    raise ValueError("activation must be relu/tanh/sigmoid")

def act_backward(z, a, kind):
    if kind == "relu":    
        return (z > 0).astype(z.dtype)
    if kind == "tanh":    
        return 1.0 - a*a
    if kind == "sigmoid": 
        return a * (1.0 - a)
    raise ValueError("activation must be relu/tanh/sigmoid")

# helpers
def parse_hidden(s: str):
    return [int(x) for x in s.split(",")] if s else [64]

def softmax(z):
    z = z - z.max(axis=1, keepdims=True) 
    ez = np.exp(z)
    return ez / (ez.sum(axis=1, keepdims=True) + 1e-12)

def accuracy(y_true, y_pred):
    return float((y_true == y_pred).mean())

def stratified_split(X, y, train=0.70, val=0.15, seed=42):
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    idx_tr, idx_va, idx_te = [], [], []
    for c in classes:
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_tr = int(n * train)
        n_va = int(n * val)
        idx_tr.append(idx[:n_tr])
        idx_va.append(idx[n_tr:n_tr+n_va])
        idx_te.append(idx[n_tr+n_va:])
    idx_tr = np.concatenate(idx_tr); idx_va = np.concatenate(idx_va); idx_te = np.concatenate(idx_te)
    rng.shuffle(idx_tr); rng.shuffle(idx_va); rng.shuffle(idx_te)
    return (X[idx_tr], y[idx_tr]), (X[idx_va], y[idx_va]), (X[idx_te], y[idx_te])


# mlp class
class MLP:
    def __init__(self, input_dim, hidden_layers, num_classes, activation="relu", seed=42, l2=0.0):
        self.activation = activation
        self.l2 = float(l2)
        rng = np.random.default_rng(seed)

        sizes = [input_dim] + list(hidden_layers) + [num_classes]
        self.W = []
        self.b = []
        for i in range(len(sizes)-1):
            fan_in, fan_out = sizes[i], sizes[i+1]
            if activation == "relu":
                W = rng.normal(0.0, np.sqrt(2.0/fan_in), size=(fan_in, fan_out)).astype(np.float32)
            else:  
                W = rng.normal(0.0, np.sqrt(1.0/fan_in), size=(fan_in, fan_out)).astype(np.float32)
            b = np.zeros((1, fan_out), dtype=np.float32)
            self.W.append(W); self.b.append(b)

    def forward(self, X):
        """Returns probs and caches for backprop"""
        A = X
        caches = []  
        L = len(self.W)
        for l in range(L):
            Z = A @ self.W[l] + self.b[l]   
            if l < L-1:
                A_next = act_forward(Z, self.activation)
            else:
                A_next = Z
            caches.append((A, Z, A_next))
            A = A_next
        probs = softmax(A)  
        return probs, caches

    def backward(self, probs, y, caches):
        """Cross-entropy grads; returns dW, db lists"""
        N = y.shape[0]
        L = len(self.W)
        dZ = probs.copy()
        dZ[np.arange(N), y] -= 1.0
        dZ /= N

        dW_list, db_list = [None]*L, [None]*L
        for l in reversed(range(L)):
            A_prev, Z, A = caches[l]
            dW = A_prev.T @ dZ + self.l2 * self.W[l]
            db = dZ.sum(axis=0, keepdims=True)
            dW_list[l] = dW.astype(np.float32)
            db_list[l] = db.astype(np.float32)

            if l > 0:
                dA_prev = dZ @ self.W[l].T
                A_prev_prev, Z_prev, A_prev_post = caches[l-1]
                dZ = dA_prev * act_backward(Z_prev, A_prev_post, self.activation)
        return dW_list, db_list

    def step(self, dW_list, db_list, lr):
        for l in range(len(self.W)):
            self.W[l] -= lr * dW_list[l]
            self.b[l] -= lr * db_list[l]

    def predict(self, X):
        probs, _ = self.forward(X)
        return probs.argmax(axis=1), probs

# train loop
def train(model, Xtr, ytr, Xva, yva, epochs=30, lr=0.05, batch_size=256, seed=42):
    rng = np.random.default_rng(seed)
    n = Xtr.shape[0]
    for epoch in range(1, epochs+1):
        # mini-batch SGD
        idx = rng.permutation(n)
        for start in range(0, n, batch_size):
            b = idx[start:start+batch_size]
            probs, caches = model.forward(Xtr[b])
            dW, db = model.backward(probs, ytr[b], caches)
            model.step(dW, db, lr)
        # metrics
        tr_pred, tr_probs = model.predict(Xtr)
        va_pred, va_probs = model.predict(Xva)
        tr_acc = accuracy(ytr, tr_pred)
        va_acc = accuracy(yva, va_pred)
        tr_loss = -np.log(tr_probs[np.arange(Xtr.shape[0]), ytr] + 1e-12).mean()
        va_loss = -np.log(va_probs[np.arange(Xva.shape[0]), yva] + 1e-12).mean()
        print(f"epoch {epoch:03d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {va_loss:.4f} acc {va_acc:.4f}")

# main function with arg parse
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/processed/processed.csv")
    ap.add_argument("--hidden", type=str, default="64", help='e.g. "64" or "128,64,32"')
    ap.add_argument("--activation", type=str, default="relu", choices=["relu","tanh","sigmoid"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--test_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--l2", type=float, default=0.0)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    y = df.iloc[:, 0].to_numpy(dtype=np.int64)
    X = df.iloc[:, 1:].to_numpy(dtype=np.float32)

    uniq, y_mapped = np.unique(y, return_inverse=True)
    y = y_mapped.astype(np.int64)
    num_classes = len(uniq)

    tr, va, te = stratified_split(X, y, train=1.0-args.val_ratio-args.test_ratio, val=args.val_ratio, seed=args.seed)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = tr, va, te

    model = MLP(
        input_dim=Xtr.shape[1],
        hidden_layers=parse_hidden(args.hidden),
        num_classes=num_classes,
        activation=args.activation,
        seed=args.seed,
        l2=args.l2
    )

    # train
    train(model, Xtr, ytr, Xva, yva, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, seed=args.seed)

    # test
    yhat, _ = model.predict(Xte)
    test_acc = accuracy(yte, yhat)
    print(f"\nTest accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()
```

### Summary

In this step, the MLP was implemented using only Numpy in a flexible way, so that a user can select the hyperparameters of the model, the architecture, activation functions, etc. This completes the implementation of the core model, which will be evaluated in the next section. 


---

*Note*: Artificial Intelligence was used in this exercise for code completion and review, as well as for text revision and refinement.


## 5. Model Training

We train the NumPy MLP using **mini‑batch SGD** and cross‑entropy. The training loop implements: **forward propagation → loss calculation → backpropagation → parameter updates**. We initialize weights randomly (small variance) and optionally apply **L2 penalty** to reduce overfitting.

**Python (core idea):**
```python
from mlp_numpy import MLP, stratified_split
import numpy as np, pandas as pd

df = pd.read_csv("src/data/clean/bank-additional-full-post-preprocessed.csv")
y = df.iloc[:,0].to_numpy(dtype=np.int64)
X = df.iloc[:,1:].to_numpy(dtype=np.float32)

(Xtr,ytr), (Xva,yva), (Xte,yte) = stratified_split(X, y, train=0.70, val=0.15, seed=42)
model = MLP(input_dim=X.shape[1], hidden_layers=[128,64], num_classes=len(np.unique(y)),
            activation="relu", l2=1e-4, seed=42)
# training loop: forward -> loss -> backward -> step
```
**CLI (reproducible baseline):**
```bash
python src/step5_train_mlp.py --data src/data/clean/bank-additional-full-post-preprocessed.csv --hidden 128,64 --activation relu --epochs 60 --lr 0.03 --batch_size 512 --l2 1e-4 --seed 42
```
**Challenges & fixes.** To avoid saturation/vanishing gradients we prefer **ReLU** over `tanh/sigmoid` for hidden layers; we keep inputs standardized; and we use **early stopping** (below) plus mild **L2** to control overfitting.

---

## 6. Training & Testing Strategy

- **Split:** stratified **70/15/15** (train/validation/test) with seed **42** for reproducibility.  
- **Training mode:** **mini‑batch** (batch size 512) balances speed and stability.  
- **Early stopping:** on validation loss with `patience=10`, `min_delta=1e-4`.  
- **Rationale:** Validation guides hyperparameter tuning (hidden sizes, LR, patience, L2).

**CLI (final early‑stopped run):**
```bash
python src/step6_strategy.py --data src/data/clean/bank-additional-full-post-preprocessed.csv --hidden 256,128,64 --activation relu --epochs 80 --lr 0.025 --batch_size 512 --patience 10 --min_delta 1e-4 --l2 1e-4 --seed 123
```
**Curves (deep/wide run):**  
![Loss vs. Epochs](images/curves_step6_deep_loss.png)  
![Accuracy vs. Epochs](images/curves_step6_deep_acc.png)

---

## 7. Error Curves and Visualization

We **show four plots** to analyze convergence and generalization: training/validation **loss** and **accuracy** for a quick baseline and for the final deep/wide model.

**Notebook:** `src/step7_curves.ipynb`

**Quick baseline (32×32)**  
![Loss — quick](images/curves_step6_quick_loss.png)  
![Accuracy — quick](images/curves_step6_quick_acc.png)

**Deep/Wide (256×128×64)**  
![Loss — deep/wide](images/curves_step6_deep_loss.png)  
![Accuracy — deep/wide](images/curves_step6_deep_acc.png)

**Interpretation.** The training curves reveal several important patterns:

**Early Training Phase (Epochs 1-20):** Both training and validation metrics improve steadily, with validation accuracy actually exceeding training accuracy initially. This suggests the model is learning generalizable patterns effectively.

**Peak Performance (Epoch 21):** Validation accuracy reaches its maximum at **0.9024** (epoch 21), indicating the optimal point before overfitting begins. The validation loss also reaches its minimum around this time.

**Overfitting Transition (Epochs 30-50):** Around epoch 30, we observe a subtle but important shift:
- **Validation accuracy plateaus** and begins fluctuating around 0.900-0.902, while **training accuracy continues improving** (reaching 0.900+ by epoch 45)
- The **accuracy gap reverses**: Initially validation > training (negative gap), but by epoch 45-50, training > validation (positive gap)
- **Loss behavior mirrors this pattern**: Validation loss initially lower than training loss, but the gap narrows and eventually reverses

**Late Training Phase (Epochs 50+):** Clear overfitting emerges as training accuracy continues rising while validation accuracy stagnates or slightly declines. The model becomes increasingly specialized to the training set.

**Early Stopping Effectiveness:** The model stopped at epoch 65 with the best validation performance at epoch 55, demonstrating that early stopping successfully prevented severe overfitting. The final model maintains good generalization despite the overfitting trend in later epochs.

**Key Insights:**
- **Standardized inputs and L2 regularization** (λ=1e-4) help control overfitting but don't eliminate it entirely
- The **deep architecture** (256×128×64) has sufficient capacity to overfit, but early stopping provides effective regularization
- The **validation set effectively guides** hyperparameter selection and prevents overfitting


## 8. Evaluation Metrics

**Notebook:** `src/step8_metrics.ipynb`

We evaluate on the **held‑out test set** and compare to a **majority‑class baseline**. Because the dataset is **imbalanced**, we report **ROC‑AUC** and **PR‑AUC** (in addition to accuracy). We include a **confusion matrix heatmap** and the **per‑class precision/recall/F1** table.

### Summary metrics:

| Metric                       | Value                      | Interpretation                                                                                     |
| ---------------------------- | -------------------------- | -------------------------------------------------------------------------------------------------- |
| **Accuracy**                 | 0.897                      | Overall 89.7 % correct predictions; inflated by majority class                                     |
| **Baseline Accuracy**        | 0.883                      | Always predicting 0 (majority) would already get 88.3 % — so accuracy alone isn’t very informative |
| **ROC–AUC**                  | 0.790                      | Good discrimination ability across thresholds                                                      |
| **PR–AUC**                   | 0.458                      | Moderate; model can identify some positives better than random                                     |



### Confusion Matrix

![Confusion Matrix](images/metrics_step8_cm.png)  

| True Class ↓ / Predicted → | 0    | 1   |
| -------------------------- | ---- | --- |
| **0 (negative)**           | 5121 | 101 |
| **1 (positive)**           | 510  | 181 |

**Interpretation:**

* **True Negatives (TN):** 5121
  → The model correctly predicted class 0 (negative) for 5121 samples.
* **False Positives (FP):** 101
  → 101 samples were incorrectly classified as positive.
* **False Negatives (FN):** 510
  → 510 actual positives were missed (classified as 0).
* **True Positives (TP):** 181
  → 181 actual positives were correctly detected.

So while the overall accuracy is **high (0.897)**, that’s largely because class 0 dominates — the model is very good at recognizing negatives but struggles with positives.

---

### ROC Curve

![ROC Curve](images/metrics_step8_roc.png) 

The **Receiver Operating Characteristic** (ROC Curve (AUC = 0.790)) curve plots:

* **x-axis:** False Positive Rate (1 − Specificity)
* **y-axis:** True Positive Rate (Recall or Sensitivity)

The **AUC (Area Under Curve)** measures how well the model separates the two classes regardless of threshold.

**Interpretation:**

* **AUC = 0.5:** random guessing
* **AUC = 1.0:** perfect separation
* **Your AUC = 0.79:** good — the model can discriminate between classes better than random, but not perfectly.
  In practical terms, given one random positive and one random negative, the model assigns the positive a higher score ~79 % of the time.

The curve shape (steep rise near the origin, then tapering) suggests it achieves high recall with relatively few false positives early on, then saturates as threshold decreases.

---

### Precision–Recall Curve

![Precision–Recall Curve](images/metrics_step8_pr.png)

This plot is more **informative for imbalanced datasets** like yours.

* **Precision:** TP / (TP + FP) — how many predicted positives are correct.
* **Recall:** TP / (TP + FN) — how many actual positives are detected.
* **AP (Average Precision):** area under the Precision–Recall curve (similar to AUC).

**Interpretation:**

* Baseline (dotted line) represents the **positive class prevalence**.
  So the model’s precision is substantially above random across much of the curve.
* **AP = 0.458** indicates moderate ability to rank positive cases above negatives, but still many false positives when recall increases.
* The sharp initial peaks show that at stricter thresholds, the model can achieve **very high precision** (but only for a small number of cases).

---
### **Per‑class metrics:**  
| Class | Precision | Recall | F1-score | Support |
|:--|--:|--:|--:|--:|
| 0 | 0.909 | 0.981 | 0.944 | 5222 |
| 1 | 0.642 | 0.262 | 0.372 | 691 |
| macro avg | 0.776 | 0.621 | 0.658 | 5913 |
| weighted avg | 0.878 | 0.897 | 0.877 | 5913 |


### Interpretation

#### Class 0 (majority)

* **Precision = 0.909** → About 91 % of predicted 0’s are truly 0.
* **Recall = 0.981** → The model correctly finds almost all real 0’s (misses only ~2 %).
* **F1 = 0.944** → Excellent overall balance between precision and recall.
* **Interpretation:** The classifier is very reliable when identifying class 0.

#### Class 1 (minority)

* **Precision = 0.642** → Roughly two-thirds of predicted 1’s are correct.
* **Recall = 0.262** → Detects only about a quarter of the actual positives.
* **F1 = 0.372** → Weak combined performance, driven by low recall.
* **Interpretation:** The model struggles to capture true positives, producing many false negatives.

#### Macro averages (unweighted)

* **Precision = 0.776**, **Recall = 0.621**, **F1 = 0.658**
* Treats both classes equally.
* Shows moderate overall discrimination but clear imbalance between the two classes.

#### Weighted averages (by class frequency)

* **Precision = 0.878**, **Recall = 0.897**, **F1 = 0.877**
* Dominated by class 0 performance.
* Closely matches overall test accuracy (~0.897).


**Notes.** As typical in imbalanced settings, recall for the positive class is lower at default threshold 0.5; **threshold tuning**, **class weighting**, or **cost‑sensitive** optimization can raise recall for the positive class with acceptable precision trade‑offs.


## 9. Conclusion

* The model is **excellent at predicting the majority class** but **weak on minority detection**.
* Future improvements should target **raising recall for class 1** — e.g., by resampling, adjusting thresholds, or rebalancing the loss function.

**Findings.** A from‑scratch NumPy MLP with mini‑batch SGD and early stopping reaches strong ranking metrics on the Bank Marketing dataset, outperforming a majority baseline.

**Limitations.** Plain MLPs do not directly address class imbalance and may require threshold tuning; probability calibration is not ensured.

**Future work.** Class‑weighting or focal loss, wider architecture/regularization sweeps, feature engineering on contact history/time, and comparison with gradient‑boosted trees.

---

### Submission — GitHub Pages

This report (Steps 1–8, Conclusion, and References) is designed for **GitHub Pages** (course template compatible). All images live in `src/images/` and are referenced relatively.

### Academic Integrity & AI Collaboration

AI assistance was used for code scaffolding, documentation, and figure generation. The authors understand and can explain all parts of the solution; plagiarism policies were respected.

### References

1. Moro, S., Cortez, P., & Rita, P. (2014). *A Data‑Driven Approach to Predict the Success of Bank Telemarketing*. Decision Support Systems, 62, 22–31. (UCI Bank Marketing)  
2. Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). *Learning representations by back‑propagating errors*. Nature, 323, 533–536.  
3. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press (Chs. 6–7 for MLPs and optimization).
