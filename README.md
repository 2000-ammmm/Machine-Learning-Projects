# Python & Machine Learning Practice

A collection of my practice files while learning Python, data preprocessing, and machine learning. Each script is a small exercise to understand one concept at a time.

## Topics Covered

- Data loading and cleaning with **pandas**
- Handling missing values (mean, median, most frequent)
- Feature scaling and categorical encoding
- Building preprocessing workflows with `Pipeline` and `ColumnTransformer`
- Train/test splitting
- Training a **Logistic Regression** model
- Evaluating models: accuracy, classification report, confusion matrix

## Repository Structure

```
.
├── ML Models/
│   ├── Automatic the best method for fillling values.py
│   └── ...
├── Data Sets/
│   └── train.csv
└── README.md
```

> Update this tree to match the files in your repo.

## Featured Practice: Titanic Missing-Value Handling

**File:** `ML Models/Automatic the best method for fillling values.py`

This script compares ways of filling missing values on the Titanic dataset and builds a full preprocessing and modeling workflow.

**Steps:**

1. Load `train.csv` and drop unused columns (`PassengerId`, `Name`, `Ticket`, `Cabin`).
2. Split the data into training and test sets (80/20, `random_state=42`).
3. Preprocess with a `ColumnTransformer`:
   - **Numeric columns** (`Pclass`, `Age`, `SibSp`, `Parch`, `Fare`): `SimpleImputer` then `StandardScaler`
   - **Categorical columns** (`Sex`, `Embarked`): `SimpleImputer` (most frequent) then `OneHotEncoder`
4. Train a `LogisticRegression` model.
5. Evaluate on the test set.

**Result (median imputation + Logistic Regression):**

| Metric | Value |
|---|---|
| Accuracy | 0.81 |
| Precision (survived) | 0.79 |
| Recall (survived) | 0.74 |
| F1-score (survived) | 0.76 |

**Confusion matrix:**

|  | Predicted: Died | Predicted: Survived |
|---|---|---|
| **Actual: Died** | 90 | 15 |
| **Actual: Survived** | 19 | 55 |

## Requirements

- Python 3.9+
- pandas
- numpy
- scikit-learn

Install everything with:

```bash
pip install pandas numpy scikit-learn
```

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/2000-ammmm/<your-repo-name>.git
   cd <your-repo-name>
   ```
2. Open the script and update the dataset path to match your computer, for example:
   ```python
   data = pd.read_csv(r'Data Sets/train.csv')
   ```
3. Run it:
   ```bash
   python "ML Models/Automatic the best method for fillling values.py"
   ```

## Key Things I Learned

- `inplace=True` returns `None`, so never assign its result to a variable.
- `ColumnTransformer` runs transformers in parallel on the original columns; use a `Pipeline` when steps must run one after another.
- Fit preprocessing on the training data only (`fit_transform`), then use `transform` on the test data to avoid data leakage.
- Accuracy alone can hide weak spots, so check precision, recall, and the confusion matrix too.

## Author

**Shuvashis Biswas**
GitHub: [@2000-ammmm](https://github.com/2000-ammmm)

## Notes

This repository is for learning and practice. The code is not meant for production use.
