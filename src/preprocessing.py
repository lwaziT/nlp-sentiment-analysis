import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    return ' '.join([word for word in text.split() if word not in STOPWORDS])

def preprocess_dataset(df, label_encoder=None):
    df['text'] = df['text'].apply(clean_text)
    if label_encoder is None:
        label_encoder = LabelEncoder()
        df['label'] = label_encoder.fit_transform(df['label'])
    else:
        df['label'] = label_encoder.transform(df['label'])
    return df, label_encoder

def tokenize_and_pad(texts, max_words=10000, max_len=100):
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    return tokenizer, padded
