# AI-Engineering

A hands-on collection of beginner-to-intermediate **machine learning** notebooks built with Python, pandas, seaborn and scikit-learn. Each notebook explores a different dataset and a different core ML concept: from regression to classification and exploratory data analysis (EDA).

## 📂 Repository Contents

| Notebook | Topic | Dataset |
|---|---|---|
| `01_Linear_regression.ipynb` | Simple & multiple linear regression (predicting `sepal_length`), train/test split, MSE | `iris.csv` |
| `02_LinearRegression.ipynb` | Data cleaning, trend visualization and linear regression on time-series style data (confirmed cases in Maharashtra & Gujarat) | `covid_19_india.csv` |
| `03_LogisticRegression.ipynb` | Univariate/bivariate analysis, IQR outlier handling, encoding and Logistic Regression for credit default prediction | `Default.csv` |
| `04_DecisionTreeClassifier.ipynb` | Exploring Pokémon stats by type and classifying whether a Pokémon is Legendary | `Pokemon.csv` |
| `05_Heart_Disease_EDA.ipynb` | Full EDA: data quality, distributions, outliers, feature-vs-target analysis, correlations | `heart__1_.csv` |

## 📊 Datasets

| File | Rows | Description |
|---|---|---|
| `iris.csv` | 150 | Classic Iris flower measurements (sepal/petal length & width, species) |
| `covid_19_india.csv` | ~1,250 | State-wise daily COVID-19 cases (confirmed, cured, deaths) in India |
| `Default.csv` | 10,000 | Credit card customers: balance, income, student status, default (Yes/No) |
| `Pokemon.csv` | 800 | Pokémon stats (HP, Attack, Defense, Speed, etc.), types and Legendary flag |
| `heart__1_.csv` | 303 | Patient health records with 13 clinical features and a heart disease target (`output`) |

## ❤️ Highlight: Heart Disease EDA

`05_Heart_Disease_EDA.ipynb` analyses 303 patients (302 after removing 1 duplicate row).

**Key findings**
- No missing values; target is fairly balanced (≈54% `output=1`, 46% `output=0`).
- Strongest predictors: chest pain type (`cp`), max heart rate (`thalachh`), exercise-induced angina (`exng`), ST depression (`oldpeak`), number of major vessels (`caa`), `thall` and `slp`.
- Weak predictors: fasting blood sugar (`fbs`), cholesterol (`chol`) and resting blood pressure (`trtbps`).
- `chol` and `oldpeak` are right-skewed with a few outliers (kept, as they are plausible medical values).
- Patients with `output=1` have a noticeably higher max heart rate (~158 vs ~139 bpm) and lower `oldpeak` (~0.6 vs ~1.6).
- The dataset has about twice as many males as females, so sex-based comparisons should be read with care.

## 🛠️ Tech Stack

- Python 3
- pandas, NumPy
- matplotlib, seaborn, plotly
- scikit-learn
- Jupyter Notebook

## 🚀 Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/AI-engineering.git
cd AI-engineering

# 2. (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install pandas numpy matplotlib seaborn plotly scikit-learn jupyter

# 4. Launch Jupyter
jupyter notebook
```

Keep the CSV files in the same folder as the notebooks so the relative paths work.

## 🗺️ Roadmap

- [ ] Train Logistic Regression and Decision Tree models on the Heart dataset
- [ ] Add evaluation metrics (confusion matrix, precision/recall, ROC-AUC) and cross-validation
- [ ] Try Random Forest / KNN and compare models
- [ ] Add a `requirements.txt`

## 👩‍💻 Author

**Aamina Bibi**: Frontend Developer (MERN) & Computer Science student
Portfolio: [aaminabibi.github.io/Portfolio](https://aaminabibi.github.io/Portfolio)

## ⚠️ Disclaimer

These notebooks are for learning purposes only. The heart disease analysis is **not** medical advice.
