from src.data_loader import load_datasets
from src.preprocessing import preprocess_dataset, tokenize_and_pad
from src.model import build_model
from src.train import split_data, train_model
from src.evaluate import plot_confusion_matrix, print_classification_report
import numpy as np
import logging

# Basic logging setup
logging.basicConfig(
    level=logging.DEBUG,  # Minimum level of logs to capture
    format="%(asctime)s [%(levelname)s] %(message)s"
)

if __name__ == "__main__":
    logging.info("Loading data sets...")
    emotion_df, violence_df, gbv_df = load_datasets()


    # Use one dataset for demonstration
    logging.info("Preprocessing data set...")
    df = emotion_df.copy()
    df, le = preprocess_dataset(df)
    tokenizer, padded = tokenize_and_pad(df['text'])
    labels = df['label'].values

    logging.info("training model...")
    X_train, X_val, y_train, y_val = split_data(padded, labels)

    model = build_model(vocab_size=10000, output_dim=len(np.unique(labels)))
    history = train_model(model, X_train, y_train, X_val, y_val)

    logging.info("Predicting model...")
    y_pred = model.predict(X_val).argmax(axis=1)

    plot_confusion_matrix(y_val, y_pred, le.classes_)
    print_classification_report(y_val, y_pred, le.classes_)
