# Airline Sentiment Classification with a Recurrent Neural Network

Three-class sentiment classification (negative / neutral / positive) of US airline tweets, using a
stacked bidirectional LSTM with an attention mechanism.

## Dataset
- The Twitter US Airline Sentiment dataset, and the text preprocessing applied to it, come from the previous exercise.


## Results

- Best validation loss 0.5433, reached at epoch 4; training stopped at epoch 9
- Recall per class on the test set: negative 85.8%, neutral 56.8%, positive 60.2%

 

## Architecture

The model uses: bidirectionality, stacked recurrent
layers, and attention. It has 1,148,968 trainable parameters.

- Embedding layer, 100 dimensions per vocabulary entry, learned from scratch with the rest of the
  network. Index 0 is reserved for padding and its vector is held at zero.
- Stacked bidirectional LSTM, two layers of 128 units.
- Attention. A small two-layer network scores every position.
- Classifier: dropout, then a linear layer to the three classes.

## Input sequences

Tokens are mapped to integer indices and padded to a common length. Both the vocabulary and the
padding length are derived from the training split.
- Vocabulary of 5,009 entries, keeping words that appear at least twice, plus `<pad>` and `<unk>`
- Padding length 27, the 95th percentile of the training lengths.
- About 4% of tweets lose their tail, and 4.1% of test tokens fall outside
  the vocabulary and become `<unk>`

## Training

- Batches of 128
- Early stopping on validation loss with a patience of 5 epochs
- The weights of the best validation epoch are restored before evaluation.


## Project structure

- `data/raw/Tweets.csv` — the original dataset
- `src/config.py` — paths, random seed, column names and sequence settings
- `src/preprocessing.py` — loading, cleaning and tokenization, carried over unchanged
- `src/sequences.py` — label encoding, splitting, vocabulary and padded sequences
- `src/model.py` — the bidirectional LSTM with masked attention
- `src/train.py` — batching and the training loop
- `src/evaluate.py` — predictions, loss curves and the confusion matrix
- `notebooks/01_airline_sentiment_rnn.ipynb` — the analysis, with markdowns and results
- `reports/figures/` — generated plots

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Then open `notebooks/01_airline_sentiment_rnn.ipynb` and run all cells. Results are reproducible:
every random source is seeded from a single value in `src/config.py`.

## Author

Vasileios Loizidis
