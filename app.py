import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

st.title("Aplikasi Prediksi Saham CNN-LSTM")

# --- Bagian Sidebar untuk Input Pengguna ---
st.sidebar.header("Parameter Model & Data")
ticker = st.sidebar.text_input("Kode Saham (Ticker)", value="ANTM.JK")
start_date = st.sidebar.date_input("Tanggal Mulai", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("Tanggal Akhir", pd.to_datetime("2023-12-31"))
epochs_input = st.sidebar.number_input("Epochs", min_value=1, max_value=100, value=15)

# --- Definisi Fungsi Persis Seperti di Notebook ---
def create_sequences(feature_data, target_data, time_steps):
    X, y, target_indices = [], [], []
    for i in range(time_steps, len(feature_data)):
        X.append(feature_data[i-time_steps:i])
        y.append(target_data[i])
        target_indices.append(i)
    return np.array(X), np.array(y), np.array(target_indices)

def prepare_data(df, feature_cols, target_col="Adj Close", time_steps=30):
    data = df.copy()
    n = len(data)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    feature_scaler.fit(data.loc[:train_end-1, feature_cols])
    target_scaler.fit(data.loc[:train_end-1, [target_col]])

    features_scaled = feature_scaler.transform(data[feature_cols])
    target_scaled = target_scaler.transform(data[[target_col]])

    X, y, target_indices = create_sequences(features_scaled, target_scaled, time_steps)
    y = y.reshape(-1, 1)

    train_mask = target_indices < train_end
    val_mask = (target_indices >= train_end) & (target_indices < val_end)
    test_mask = target_indices >= val_end

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "target_scaler": target_scaler
    }

def build_cnn_lstm(time_steps, n_features):
    model = Sequential([
        Input(shape=(time_steps, n_features)),
        Conv1D(filters=64, kernel_size=1, activation="elu"),
        MaxPooling1D(pool_size=2),
        LSTM(units=50, activation="elu"),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae", "mape"])
    return model

def evaluate_model(model, data_dict):
    X_test = data_dict["X_test"]
    y_test = data_dict["y_test"]
    target_scaler = data_dict["target_scaler"]

    y_pred_scaled = model.predict(X_test, verbose=0)
    y_true = target_scaler.inverse_transform(y_test).ravel()
    y_pred = target_scaler.inverse_transform(y_pred_scaled).ravel()

    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred) * 100,
        "R2": r2_score(y_true, y_pred),
        "y_true": y_true,
        "y_pred": y_pred
    }

# --- Alur Eksekusi Utama ---
if st.button("Jalankan Prediksi"):
    with st.spinner("Mengunduh dan menyiapkan data..."):
        df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False, progress=False)
        df = df.reset_index()
        df = df[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
        df = df.dropna()
        df["HiLo"] = df["High"] - df["Low"]
        df["OpSe"] = df["Open"] - df["Close"]
        df = df.dropna().reset_index(drop=True)
        
        st.subheader("Data Preview")
        st.dataframe(df.tail())
        
        st.subheader("Grafik Harga Penyesuaian (Adj Close)")
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(df["Date"], df["Adj Close"])
        ax.set_xlabel("Date")
        ax.set_ylabel("Harga")
        ax.grid(True)
        st.pyplot(fig)
        
    with st.spinner("Melatih Model CNN-LSTM (Eksperimen 6 Fitur)... Ini mungkin butuh waktu sedikit."):
        six_features = ["High", "Low", "Volume", "Open", "HiLo", "OpSe"]
        data_dict = prepare_data(df, feature_cols=six_features, time_steps=30)
        
        model = build_cnn_lstm(time_steps=30, n_features=len(six_features))
        early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        
        history = model.fit(
            data_dict["X_train"], data_dict["y_train"],
            validation_data=(data_dict["X_val"], data_dict["y_val"]),
            epochs=epochs_input,
            batch_size=4,
            callbacks=[early_stop],
            verbose=0
        )
        
        eval_result = evaluate_model(model, data_dict)
        
        st.success("Training Selesai!")
        
        # Menampilkan metrik seperti di DataFrame Jupyter
        st.subheader("Hasil Evaluasi")
        col1, col2, col3 = st.columns(3)
        col1.metric("RMSE", f"{eval_result['RMSE']:.4f}")
        col2.metric("MAPE", f"{eval_result['MAPE']:.2f}%")
        col3.metric("R2 Score", f"{eval_result['R2']:.4f}")
        
        st.subheader("Visualisasi Prediksi vs Aktual")
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(eval_result["y_true"], label="Nilai Aktual", color="blue")
        ax2.plot(eval_result["y_pred"], label="Prediksi Model", color="orange", alpha=0.7)
        ax2.legend()
        st.pyplot(fig2)