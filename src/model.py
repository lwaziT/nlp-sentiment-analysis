from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense

def build_model(vocab_size, output_dim):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=64),
        GlobalAveragePooling1D(),
        Dense(32, activation='relu'),
        Dense(output_dim, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
