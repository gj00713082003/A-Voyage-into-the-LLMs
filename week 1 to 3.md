
# ✅ Python Foundation & Deep Learning Libraries - Learning Log

This document records my completion and understanding of the foundational concepts and tools used in deep learning with Python. It includes Python basics, NumPy, Pandas, Matplotlib, and PyTorch.

---

## ✔️ Python Refresher

✅ **Completed**
- Variables, Loops, Functions, Data Structures
- List comprehensions
- File handling and exceptions

### 🧪 Code Example:
```python
def greet(name):
    return f"Hello, {name}!"

names = ["Alice", "Bob", "Charlie"]
for name in names:
    print(greet(name))
```

---

## ✔️ NumPy

✅ **Completed Topics**:
- Creating arrays, indexing, slicing, reshaping
- Joining, splitting, searching, filtering, ufuncs

### 🧪 Code Example:
```python
import numpy as np

a = np.array([1, 2, 3, 4])
b = np.array([[1, 2], [3, 4]])

# Reshaping
reshaped = b.reshape(4, 1)

# Ufunc example
sqrt_a = np.sqrt(a)
print("Reshaped:
", reshaped)
print("Sqrt of a:", sqrt_a)
```

---

## ✔️ Pandas

✅ **Completed Topics**:
- Series, DataFrames, reading/writing CSV
- Filtering, sorting, grouping, handling NaN values

### 🧪 Code Example:
```python
import pandas as pd

data = {'Name': ['Alice', 'Bob', 'Charlie'], 'Age': [25, 30, 35]}
df = pd.DataFrame(data)

# Filtering
filtered_df = df[df['Age'] > 28]
print(filtered_df)
```

---

## ✔️ Matplotlib

✅ **Completed Topics**:
- Line plots, bar charts, scatter plots
- Customizing titles, axes, legends

### 🧪 Code Example:
```python
import matplotlib.pyplot as plt

x = [1, 2, 3, 4]
y = [10, 20, 25, 30]

plt.plot(x, y, marker='o', color='purple')
plt.title("Line Plot Example")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.grid(True)
plt.show()
```

---

## ✔️ Google Colab / Jupyter Notebook

✅ **Completed Setup & Usage**
- Used Google Colab for all notebooks
- Understood and used multiple code cells
- Used Markdown + Python in cells

---

## ✔️ PyTorch (Beginner Level)

✅ **Completed Topics**:
- Tensors and basic operations
- Creating simple neural network using `nn.Module`
- Basic training loop
- Used Autograd for backpropagation

### 🧪 Code Example:
```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        return self.fc(x)

model = Net()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Dummy training loop
for epoch in range(10):
    inputs = torch.tensor([[1.0, 2.0]])
    target = torch.tensor([[1.0]])

    output = model(inputs)
    loss = criterion(output, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
```

---

## 💡 Summary

- ✅ Refreshed Python basics with functions and data structures.
- ✅ Mastered array operations with **NumPy**.
- ✅ Cleaned and manipulated data using **Pandas**.
- ✅ Visualized data effectively using **Matplotlib**.
- ✅ Built and trained basic models using **PyTorch**.
- ✅ All code and experiments were performed using **Google Colab**.

---

## 📁 Folder Structure Suggestion for GitHub

```
python-foundation-learning/
├── notebooks/
│   ├── numpy_practice.ipynb
│   ├── pandas_practice.ipynb
│   ├── matplotlib_examples.ipynb
│   └── pytorch_intro.ipynb
├── images/
│   └── example_plot.png
├── python-foundation-learning-log.md
└── README.md
```

---

## 🚀 Ready for More!

With these fundamentals in place, I’m ready to move forward into advanced deep learning topics like model optimization, CNNs, and more.

# Retry saving the elaborated NLP markdown content into a file now that the tool is available again

nlp_markdown = """
# 🧠 Natural Language Processing (NLP) - Learning Log

This markdown summarizes key concepts in NLP including Regular Expressions, Word Embeddings, Text Preprocessing, and Sentiment Analysis. It also explores different techniques used to convert textual data into numerical form — essential for building machine learning models.

---

## 🧾 Contents

- Regular Expressions in Python
- Word Embeddings (Detailed)
- Text Preprocessing with Regex
- Neural vs Frequency-based Embeddings
- Embedding Techniques Comparison

---

## 🔍 Python RegEx

Python's `re` module allows powerful text matching and manipulation using regular expressions. It is used to clean and structure text before modeling.

| Action | RegEx Pattern | Reason |
|-------|---------------|--------|
| Remove special characters | `[^a-zA-Z0-9\\s]` | Clean noisy text |
| Remove email addresses | `\\S+@\\S+` | Avoid unnecessary tokens |
| Remove URLs | `http[s]?://\\S+` | Standardize input |
| Remove numbers | `\\d+` | Eliminate noise |
| Lowercase | `.lower()` | Normalize case |
| Remove whitespace | `\\s+` | Prevent tokenization issues |
| Remove punctuation | `[^\\w\\s]` | Remove grammar artifacts |
| Tokenize words | `\\b\\w+\\b` | Word-level separation |
| Strip HTML | `<[^>]+>` | Clean web data |
| Emojis | Unicode regex | Text-normalization |
| Expand contractions | `"can't" → "can not"` | Improve clarity |

---

## 🧬 Word Embeddings

Word embeddings are vector representations of words, enabling machine learning models to interpret text numerically.

### ⚡ Why Use Word Embeddings?
- Capture semantic meaning
- Dimensionality reduction
- Represent word relationships
- Improve model generalization

### 🪙 Process Flow:
Words → Numeric vectors → Train/Test Model


### ✅ Pros
- Fast training compared to symbolic models like WordNet
- Used in almost all modern NLP architectures

### ❌ Cons
- Can be memory intensive
- Biased by training corpus
- Cannot distinguish homophones (e.g., cell/sell)

---

## 🧹 Preprocessing Best Practices

- Keep preprocessing consistent across training and deployment
- Use special tokens for unknown or rare words (e.g., `UNK`)
- Ensure consistent embedding dimensions

---

## 📊 Types of Embedding Techniques

![Techniques in Embeddings in NLP](Screenshot%202025-06-25%20223036.png)

---

## 📚 Frequency-Based Approaches

### 👜 Bag of Words (BoW)
Represents text by counting word occurrences. Ignores grammar and word order.

**Cons**:
- High dimensionality
- No semantic context

### 🧮 TF-IDF (Term Frequency-Inverse Document Frequency)
Assigns weight to words based on how unique and frequent they are across documents.

**Formula**:
TF-IDF(t, d, D) = TF(t, d) × IDF(t, D)

yaml


**Pros**:
- Highlights important terms

**Cons**:
- Doesn't capture context or semantics

---

## 🤖 Prediction-Based Embeddings

### 1️⃣ Word2Vec

- **CBOW**: Predicts a target word from context
- **Skip-Gram**: Predicts context from a target word

| Feature | CBOW | Skip-Gram |
|--------|------|-----------|
| Focus | Frequent words | Rare words |
| Speed | Faster | Slower |
| Accuracy | Lower | Higher |
| Use Case | Large data | Small/rare words |

### 2️⃣ GloVe (Global Vectors)

- Uses word co-occurrence matrix
- Static embedding

### 3️⃣ FastText

- Breaks words into n-grams
- Great for handling rare/misspelled words

---

## 🌐 Contextualized Embeddings

### 🧠 BERT

- Transformer-based
- Bi-directional context awareness
- High performance on NLP tasks

### 🧠 ELMo

- Uses LSTM
- Contextual vectors for each occurrence of a word

### 🤖 GPT

- Predicts next token (auto-regressive)
- Useful for text generation and completion

---

## 📈 Embedding Comparisons

| Feature | GloVe | FastText | BERT |
|--------|-------|----------|------|
| Type | Static | Static | Contextualized |
| OOV Handling | ❌ | ✅ | ✅ |
| Contextual | ❌ | ❌ | ✅ |
| Morphology | ⚠️ | ✅ | ✅ |
| Best Use | Similarity | Morph-rich text | Deep NLP |

---

## ✅ Conclusion

- Use **BoW/TF-IDF** for quick baselines.
- Choose **Word2Vec/FastText** for efficient semantic similarity.
- Opt for **BERT** when context and nuance matter.
"""

# Save markdown to file
markdown_path = "/mnt/data/nlp-word-embeddings-log.md"
with open(markdown_path, "w") as f:
    f.write(nlp_markdown)

markdown_path

