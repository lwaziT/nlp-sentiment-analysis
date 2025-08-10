import pandas as pd

def load_datasets():
    emotion_df = pd.read_csv("data/emotions.csv")
    violence_df = pd.read_csv("data/violence.csv")
    gbv_hate_speech_df = pd.read_csv("data/gbv_hate_speech.csv")

    # Dropping unwanted columns
    emotion_df.drop(columns=['Unnamed: 0'], inplace=True)
    violence_df.drop(columns=['Tweet_ID'], inplace=True)
    gbv_hate_speech_df = gbv_hate_speech_df[['tweet', 'class']]

    # Rename columns
    violence_df.rename(columns={'tweet': 'text', 'type': 'label'}, inplace=True)
    gbv_hate_speech_df.rename(columns={'tweet': 'text', 'class': 'label'}, inplace=True)

    # Data sampling: extracting 12 thousand rows from each dataset.
    emtn_df = pd.DataFrame()
    for i in range(6):
        subset = emotion_df[emotion_df['label'] == i].sample(n=2000, random_state=42)
        emtn_df = pd.concat([emtn_df, subset])

    emotion_df = emtn_df.copy()
    emotion_df['label'].value_counts()

    # Because violence_df has uneven counts in the columns, we only sample from the highest(sexual_violence).
    sexual_violence = violence_df[violence_df['label'] == 'sexual_violence'].sample(n=4998, random_state=42)

    # Removing 'sexual_violence' column from violence_df
    violence_df = violence_df[violence_df['label'] != 'sexual_violence']

    # Re-adding sampled 'sexual_violence' column to violence_df
    violence_df = pd.concat([sexual_violence, violence_df], axis=0)

    # Because gbv_hate_speech_df has uneven counts in the columns, we only sample from the highest(offensive_speech) column.
    offensive_speech = gbv_hate_speech_df[gbv_hate_speech_df['label'] == 1].sample(n=6407, random_state=42)

    # Removing 'offensive_speech' from gbv_hate_speech_df
    gbv_hate_speech_df = gbv_hate_speech_df[gbv_hate_speech_df['label'] != 1]

    # Re-adding sampled 'offensive_speech' column to gbv_hate_speech_df
    gbv_hate_speech_df = pd.concat([offensive_speech, gbv_hate_speech_df], axis=0)

    # Resetting indexes of dataframes
    emotion_df.reset_index(drop=True, inplace=True)
    gbv_hate_speech_df.reset_index(drop=True, inplace=True)
    violence_df.reset_index(drop=True, inplace=True)

    return emotion_df, violence_df, gbv_hate_speech_df
