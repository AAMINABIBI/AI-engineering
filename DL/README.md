
# Deep Learning

Notebooks covering the deep learning journey: from a single perceptron to CNNs, transfer learning and Transformers.

## 1. Foundations: Perceptron, ANN & MLP

| Notebook | Topic |
|---|---|
| `Gate_perceptron.ipynb` | Perceptron and logic gates |
| `ANN.ipynb` | Artificial Neural Network on a larger dataset |
| `ann_pytorch.ipynb` | ANN implemented in PyTorch |
| `MLP_DL_2.ipynb` | Multi-Layer Perceptron |
| `MLP_House_Price_Prediction_CaseStudy.ipynb` | MLP case study: house price prediction |
| `MLP_PyTorch_Student_Case_Study.ipynb` | MLP case study in PyTorch: student data |

## 2. Convolutional Neural Networks (CNN)

| Notebook | Topic |
|---|---|
| `CNN_Practice.ipynb` | CNN practice |
| `CNN_Practice2_YT.ipynb` | CNN practice (follow-along) |
| `CNN_PyTorch_GPU.ipynb` | CNN in PyTorch, trained on GPU |
| `TRN_2026_001088_AaminaBibi_CNN.ipynb` | CNN assignment |

## 3. Transfer Learning & Transformers

### `Transfer_Learning_VGG16_FMNIST.ipynb`
**Feature extraction with a pretrained VGG16 on Fashion-MNIST.**

- Dataset: 6,000 Fashion-MNIST samples (28×28 grayscale, 10 classes), 80/20 train-test split
- Preprocessing: grayscale replicated to 3 channels, resized to 224×224, ImageNet mean/std normalization
- Model: VGG16 pretrained on ImageNet. Convolutional layers frozen, classifier replaced with `25088 → 1024 → 512 → 10` (ReLU + Dropout 0.5)
- Training: Adam (lr = 0.0001), CrossEntropyLoss, batch size 32, 10 epochs on GPU
- Result: loss fell from 0.19 to 0.009. **Training accuracy 100%, test accuracy 90.25%**
- Observation: the train-test gap shows overfitting on the small dataset (augmentation or early stopping could help)

### `Fine_Tuning_BERT_IMDB.ipynb`
**Full fine-tuning of `bert-base-cased` for sentiment classification.**

- Dataset: IMDB (25,000 train / 25,000 test reviews) via Hugging Face `datasets`
- Tokenization with `AutoTokenizer`, classification head added with `AutoModelForSequenceClassification` (2 labels)
- Training with Hugging Face `Trainer`: lr 2e-5, batch size 16, weight decay 0.01, 1 epoch (1,563 steps, training loss ≈ 0.26)

### `HuggingFace_Pipelines_Demo.ipynb`
**Hands-on demo of the three Transformer architecture types using Hugging Face `pipeline`.**

| Architecture | Model | Task |
|---|---|---|
| Encoder | BERT (`bert-base-uncased`) | Fill-mask |
| Encoder | DistilBERT | Sentiment analysis |
| Decoder | GPT-2 | Text generation (temperature, sampling) |
| Encoder-Decoder | T5-small | Translation |
| Encoder-Decoder | BART-large-cnn | Summarization |

Includes experiments on truncation, `max_length` and special tokens, plus observations on model limitations (hallucination in GPT-2, task limits in T5).

## Key Concepts

- **Feature extraction:** freeze pretrained layers, train only a new head
- **Fine-tuning:** update pretrained weights with a small learning rate
- **Overfitting:** high train accuracy with lower test accuracy

## Tools

PyTorch · torchvision · Hugging Face Transformers & Datasets · Google Colab (GPU)
