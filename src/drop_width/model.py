import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split

def build_lstm_model(train_data_sliced, train_label_sliced, slide_window):
    # Determine the number of features
    num_features = train_data_sliced.shape[-1]

    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(
        train_data_sliced, train_label_sliced, test_size=0.2, random_state=42
    )

    # Define input shape
    input_shape = (slide_window, num_features)

    # Input layer
    inputs = Input(shape=input_shape)

    # LSTM layer
    x = LSTM(units=48, activation='tanh',
             kernel_regularizer=l2(0.01),
             recurrent_regularizer=l2(0.01))(inputs)

    # Dropout
    x = Dropout(0.5)(x)

    # Output layer
    raw_output = Dense(1, activation='tanh')(x)

    # Rescale to label range
    max_label = np.max(train_label_sliced)
    min_label = np.min(train_label_sliced)
    outputs = (raw_output + 1) / 2 * (max_label - min_label) + min_label

    # Build and compile the model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    return model, X_train, X_val, y_train, y_val

