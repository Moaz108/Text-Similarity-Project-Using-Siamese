# Text-Similarity-Project-Using-Siamese
Text Similarity Project Using Siamese Architecture and LSTM


# Text Similarity Detection with Siamese LSTM Networks

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)

A Siamese LSTM-based model to measure semantic similarity between pairs of sentences. This project includes data augmentation, pre-trained embeddings, and evaluation metrics for NLP tasks.

![Siamese Architecture](https://miro.medium.com/max/1400/1*NqfWUxYZEmj3l4h_4n2Dgw.png)  
*(Example Siamese Network Architecture)*

---

## Table of Contents
- [Key Features](#key-features)
- [Installation](#installation)
- [Usage](#usage)
- [Data Preparation](#data-preparation)
- [Model Architecture](#model-architecture)
- [Evaluation Metrics](#evaluation-metrics)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Key Features
- **Siamese Architecture**: Twin LSTM networks to process sentence pairs.
- **Pre-trained Embeddings**: Supports GloVe or custom embeddings.
- **Data Augmentation**: Techniques like paraphrasing, noise injection, and back-translation.
- **Evaluation Metrics**: Accuracy, Precision, Recall, and F1-score.
- **Scalable**: Easily extendable to larger datasets.

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/text-similarity-siamese-lstm.git
   cd text-similarity-siamese-lstm
Install dependencies:

bash
Copy
pip install -r requirements.txt
Requirements:

Copy
tensorflow==2.10.0
numpy==1.22.4
pandas==1.5.3
scikit-learn==1.2.2
nltk==3.8.1
transformers==4.28.1
Usage
1. Data Preparation
Place your dataset in data/raw/ with columns sentence1, sentence2, and label.

Run data augmentation scripts:

python
Copy
python scripts/data_augmentation.py
2. Train the Model
python
Copy
python train.py \
  --epochs 10 \
  --batch_size 32 \
  --embedding_dim 100 \
  --max_sequence_length 50
3. Evaluate the Model
python
Copy
python evaluate.py \
  --model_path models/siamese_lstm.h5 \
  --test_data data/processed/test.csv
4. Predict Similarity
python
Copy
python predict.py \
  --sentence1 "I love coding" \
  --sentence2 "Programming is fun"
Data Preparation
The dataset is expanded using:

Paraphrasing: Back-translation with Google Translate.

Noise Injection: Word shuffling and typos.

Public Datasets: Integrated Quora Question Pairs and STS Benchmark data.

Example data format (data/raw/dataset.csv):

csv
Copy
sentence1,sentence2,label
"I love programming","Coding is fun",1
"The cat sat on the mat","The kitten is on the rug",1
...
Model Architecture
python
Copy
Siamese LSTM Model:
1. Input Layer (2 text inputs)
2. Embedding Layer (GloVe or custom)
3. Shared LSTM Layer (128 units)
4. Manhattan Distance Calculation
5. Dense Layers (64 ReLU → 1 Sigmoid)
Evaluation Metrics
Metric	Score
Accuracy	89.2%
Precision	88.5%
Recall	90.1%
F1-Score	89.3%
Contributing
Contributions are welcome!

Fork the repository.

Create a branch: git checkout -b feature/your-feature.

Commit changes: git commit -m "Add your feature".

Push to the branch: git push origin feature/your-feature.

Submit a pull request.
