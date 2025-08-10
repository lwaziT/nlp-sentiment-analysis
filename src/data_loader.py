import pandas as pd

def load_datasets():
    emotion_df = pd.read_csv("C:/Users/tobos/OneDrive/Desktop/Projects/NLP-Sentiment-Analysis-project/nlp-sentiment-analysis/Datasets/emotions.csv")
    violence_df = pd.read_csv("C:/Users/tobos/OneDrive/Desktop/Projects/NLP-Sentiment-Analysis-project/nlp-sentiment-analysis/Datasets/violence.csv")
    gbv_df = pd.read_csv("C:/Users/tobos/OneDrive/Desktop/Projects/NLP-Sentiment-Analysis-project/nlp-sentiment-analysis/Datasets/emotions.csv", delimiter=";")

    return emotion_df, violence_df, gbv_df
