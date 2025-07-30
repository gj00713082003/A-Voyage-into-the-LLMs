#  Sentiment Recognition using LSTM (PyTorch)

This project implements a sentiment classification model using LSTM (Long Short-Term Memory) in PyTorch. The dataset consists of text reviews with associated binary sentiment labels (positive or negative).

---

##  Dataset

The dataset (`Sentiment_Recognition_data.csv`) contains two columns:

- **Review text**
- **Sentiment label** (positive/negative)

---

##  Exploratory Data Analysis (EDA)

- Sentiment Distribution is plotted using bar plots.
- Tokenization and frequency analysis is performed on the first review.
- Count and percentage of uppercase words in the entire dataset are computed.

---

##  Preprocessing Steps

###  1. Tokenization
- Used `nltk.word_tokenize()` to break text into words.

###  2. Removing Stopwords and Punctuation
- Removed using NLTK's `stopwords` and Python's `string.punctuation`.

###  3. Lowercasing and Uppercase Tagging
- Words are lowercased.
- Uppercase words (length > 1) are tagged with a prefix `__emph__`.

###  4. Stemming & Lemmatization
- Stemming via `PorterStemmer`
- Lemmatization via `WordNetLemmatizer`

###  5. Padding
- All sequences are padded/truncated to a uniform length using `pad_sequence()`.

---

##  Vocabulary & Encoding

Created vocabularies for:

- stemmed_lowercased
- stemmed_tagged
- lemmatized_lowercased
- lemmatized_tagged

Converted tokens to indices using a frequency-based vocabulary.

- `<PAD>` = 0  
- `<UNK>` = 1

---

##  Data Splitting

A dataset of 50,000 samples was split as follows:

- **Training:** 35,000  
- **Validation:** 7,500  
- **Test:** 7,500

Encoded sentiment labels as:

- **positive** → `1`
- **negative** → `0`

---

##  PyTorch Dataloaders

- Used `TensorDataset` and `DataLoader` to prepare training and validation sets.
- Supported batching and shuffling.

---

##  Model: LSTM Classifier

```python
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, output_dim=2, num_layers=2)
```

## Model Training Logs

The following logs summarize training and validation performance for each preprocessed variant of the sentiment recognition model (lemmatized/lowercased/tagged/stemmed):

###  Model: `lemm_lc`
| Epoch | Train Loss | Val Loss | Val Acc |
|-------|------------|----------|---------|
| 1     | 0.7021     | 0.7031   | 0.5036  |
| 2     | 0.6954     | 0.6934   | 0.4964  |
| 3     | 0.6943     | 0.6932   | 0.4964  |
| 4     | 0.6939     | 0.6936   | 0.4964  |
| 5     | 0.6936     | 0.6931   | 0.5036  |

---

###  Model: `lemm_tagged`
| Epoch | Train Loss | Val Loss | Val Acc |
|-------|------------|----------|---------|
| 1     | 0.7021     | 0.6931   | 0.5065  |
| 2     | 0.6947     | 0.6931   | 0.5065  |
| 3     | 0.6940     | 0.6955   | 0.5065  |
| 4     | 0.6938     | 0.6941   | 0.4935  |
| 5     | 0.6935     | 0.7375   | 0.4935  |

---

###  Model: `stem_lc`
| Epoch | Train Loss | Val Loss | Val Acc |
|-------|------------|----------|---------|
| 1     | 0.7022     | 0.6958   | 0.4969  |
| 2     | 0.6951     | 0.6959   | 0.5031  |
| 3     | 0.6941     | 0.6936   | 0.4969  |
| 4     | 0.6938     | 0.7612   | 0.4969  |
| 5     | 0.6936     | 0.6981   | 0.5031  |

---

###  Model: `stem_tagged`
| Epoch | Train Loss | Val Loss | Val Acc |
|-------|------------|----------|---------|
| 1     | 0.7018     | 0.6946   | 0.4998  |
| 2     | 0.6953     | 0.6941   | 0.4998  |
| 3     | 0.6941     | 0.6935   | 0.4998  |
| 4     | 0.6938     | 0.6956   | 0.4998  |
| 5     | 0.6936     | 0.7080   | 0.5002  |


### Validation Accuracy per Variant

- Shows how different preprocessing methods impact accuracy over epochs. Higher curves indicate better generalization.
<img width="1018" height="547" alt="download" src="https://github.com/user-attachments/assets/6b557ee8-4c34-4687-8638-5f71800eeb18" />

### Validation Loss per Variant
- Tracks how validation loss changes across epochs. Lower, stable loss suggests better training and less overfitting.
<img width="1010" height="547" alt="download" src="https://github.com/user-attachments/assets/1a7d5ea4-80c2-4f5c-9234-f9c037bdf342" />
    
