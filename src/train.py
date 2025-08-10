from sklearn.model_selection import train_test_split

def split_data(padded, labels, test_size=0.2, random_state=42):
    return train_test_split(padded, labels, test_size=test_size, random_state=random_state)

def train_model(model, X_train, y_train, X_val, y_val, epochs=10):
    return model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs)
