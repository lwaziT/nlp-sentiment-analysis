import pandas as pd

def load_datasets():
    emotion_df = pd.read_csv("data/emotions.csv")
    violence_df = pd.read_csv("data/violence.csv")
    gbv_df = pd.read_csv("data/gbv_hate_speech.csv")

    # Clean and align structure
    emotion_df.drop(columns=['Unnamed: 0'], inplace=True)
    violence_df.drop(columns=['Tweet_ID'], inplace=True)
    gbv_df = gbv_df[['tweet', 'class']]

    # Rename columns
    violence_df.rename(columns={'tweet': 'text', 'type': 'label'}, inplace=True)
    gbv_df.rename(columns={'tweet': 'text', 'class': 'label'}, inplace=True)

    return emotion_df, violence_df, gbv_df
