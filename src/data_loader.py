import pandas as pd

def load_emotion_data(path):
    return pd.read_csv(path)

def load_gbv_data(path):
    return pd.read_csv(path)

def load_violence_data(path):
    return pd.read_csv(path)
