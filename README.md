# AI-Engineering

A hands-on learning repository covering **Machine Learning** and **Deep Learning**, from linear regression to CNNs, transfer learning and Hugging Face Transformers. Each notebook focuses on one core concept, built step by step with Python.

## 📁 Repository Structure

| Folder | What's inside |
|---|---|
| [`ML/`](./ML) | Linear & logistic regression, decision trees, cross-validation, feature engineering, hyperparameter search |
| [`DL/`](./DL) | Perceptron, ANN / MLP, CNN, transfer learning, BERT fine-tuning, Hugging Face pipelines |

## ⭐ Highlights

| Notebook | What it does | Result |
|---|---|---|
| `DL/Transfer_Learning_VGG16_FMNIST.ipynb` | Pretrained VGG16 (frozen conv layers + custom classifier) on Fashion-MNIST | 90.25% test accuracy |
| `DL/Fine_Tuning_BERT_IMDB.ipynb` | Fine-tunes `bert-base-cased` for IMDB sentiment classification with Hugging Face `Trainer` | Trained on 25k reviews |
| `DL/HuggingFace_Pipelines_Demo.ipynb` | Encoder, decoder and encoder-decoder models (BERT, DistilBERT, GPT-2, T5, BART) | Inference demos |
| `ML/03_LogisticRegression.ipynb` | EDA, IQR outlier handling, encoding and logistic regression for credit default prediction | — |

## 🧠 Topics Covered

**Machine Learning:** Linear Regression, Logistic Regression, Decision Trees, Cross-Validation, Feature Engineering, GridSearchCV & RandomizedSearchCV

**Deep Learning:** Perceptron & logic gates, ANN, Multi-Layer Perceptron (MLP), CNN, Transfer Learning, Fine-tuning, Transformers (encoder / decoder / encoder-decoder)

## 🛠️ Tech Stack

Python · PyTorch · torchvision · Hugging Face Transformers & Datasets · scikit-learn · pandas · NumPy · matplotlib · seaborn

## 🚀 Getting Started

```bash
git clone https://github.com/AAMINABIBI/AI-engineering.git
cd AI-engineering
pip install pandas numpy matplotlib seaborn scikit-learn torch torchvision transformers datasets jupyter
jupyter notebook
```

GPU-based notebooks were run on Google Colab. Datasets are either loaded inside the notebook or expected in the same folder as the notebook.

## 🗺️ Roadmap

- [ ] Add evaluation metrics (confusion matrix, precision/recall) to DL notebooks
- [ ] Use data augmentation to reduce overfitting in transfer learning
- [ ] Fine-tune a pretrained model on a custom dataset
- [ ] Add `requirements.txt`

## 👩‍💻 Author

**Aamina Bibi** · Frontend Developer (MERN) & Computer Science graduate
Portfolio: [aaminabibi.github.io/Portfolio](https://aaminabibi.github.io/Portfolio)

> These notebooks are for learning purposes.
