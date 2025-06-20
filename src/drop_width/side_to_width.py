import numpy as np

def width_estim(df, model, processor, train_data_notnormalized, slide_window):
    """
    Predicts drop width from side-view features using a trained model.

    Args:
        df (pd.DataFrame): DataFrame containing raw drop measurements.
        model (tf.keras.Model): Trained LSTM model for predicting width.
        processor (DropDataProcessor): Preprocessing utility (must contain slicing and normalize methods).
        train_data_notnormalized (pd.DataFrame): Training data used for normalization reference.
        slide_window (int): Sliding window size for temporal slicing.

    Returns:
        tuple: (predicted_widths, measured_widths) – both as NumPy arrays in µm.
    """
    # Convert measured drop width from cm to µm
    if "Drop width (cm)" in df.columns:
        measured_width = df["Drop width (cm)"].to_numpy() * 10000
        # The slicing removes (window_size - 1) entries total: half at start, half at end
        offset = slide_window // 2
        measured_width = measured_width[offset: -offset]
    else:
        measured_width=[0]*len(df)

    # Extract relevant side-view features
    features = ["Advancing (degree)", "Receding (degree)", "Drop length (cm)",
                "Drop height (cm)", "Velocity (cm/s)", "Middle line angle (degree)"]
    input_features = df[features]

    # Normalize the input features based on training data statistics
    _, normalized_features = processor.normalize(train_data_notnormalized, input_features)

    # Slice the normalized data using the specified sliding window
    sliced_features = processor.slicing(normalized_features, slide_window)

    # Predict drop widths using the trained model
    y_pred = model.predict(sliced_features, verbose=0).reshape(-1)

    # Align the measured widths to match the length of the sliced inputs


    return y_pred, measured_width

