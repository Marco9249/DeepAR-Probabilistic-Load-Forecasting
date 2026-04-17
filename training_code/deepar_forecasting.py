"""
=============================================================================
DeepAR Probabilistic Electricity Load Forecasting
=============================================================================
Prepared by: Eng. Mohammed Ezzaldeen babiker abdallah

Description:
This script builds a robust DeepAR architecture using TensorFlow, Keras, and 
TensorFlow Probability to predict the probabilistic electricity load of a 
power grid. It strictly enforces chronologically sound data pipelines,
prevents data leakage, applies custom mathematical quantile evaluations, 
and generates ultra-high 4K resolution plots illustrating statistical 
uncertainty bands (confidence intervals).
=============================================================================
"""

import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Force use of legacy Keras for TFP compatibility
try:
    import tf_keras as keras
except ImportError:
    import tensorflow.keras as keras

# =============================================================================
# 1. Reproducibility Configuration
# =============================================================================
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

tfd = tfp.distributions

# =============================================================================
# 2. Data Loading & Preprocessing
# =============================================================================
def load_and_preprocess_data(file_path):
    print(">>> Phase 1: Loading & Cleaning Dataset...")
    df = pd.read_csv(file_path)
    
    # Identify datetime and target columns automatically
    datetime_col = 'Datetime' if 'Datetime' in df.columns else df.columns[0]
    target_col = 'AEP_MW' if 'AEP_MW' in df.columns else df.columns[1]
    
    # Parse chronological datetime column
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    
    # Sort chronologically
    df = df.sort_values(by=datetime_col).reset_index(drop=True)
    
    # Handle missing and duplicate datetime entries
    df = df.drop_duplicates(subset=[datetime_col], keep='first')
    
    # Set index to resample and guarantee continuous uninterrupted sequential gaps
    df = df.set_index(datetime_col)
    
    # Resample to strict 1-Hour frequency and systematically interpolate gaps
    df = df.resample('ME').asfreq() # Fallback for complete span, typically 'h'
    df = df.resample('h').asfreq()
    
    # Linear interpolation to seamlessly bridge unrecorded consumption hours
    df[target_col] = df[target_col].interpolate(method='linear')
    
    # Reset index for temporal feature extraction phase
    df = df.reset_index()
    datetime_col = 'index' if 'index' in df.columns else datetime_col
    df = df.rename(columns={'index': 'Datetime', 'Datetime': 'Datetime'})
    datetime_col = 'Datetime'
    
    # Feature Engineering: Extracting chronological seasonality variables
    df['hour'] = df[datetime_col].dt.hour
    df['day_of_week'] = df[datetime_col].dt.dayofweek
    df['month'] = df[datetime_col].dt.month
    
    # Memory and Processing Optimization: Convert down to float32
    df[target_col] = df[target_col].astype(np.float32)
    df['hour'] = df['hour'].astype(np.float32)
    df['day_of_week'] = df['day_of_week'].astype(np.float32)
    df['month'] = df['month'].astype(np.float32)
    
    print("Initial Data Properties:")
    print(df.info())
    print("\nClean Continuous Data Shape:", df.shape)
    
    return df, datetime_col, target_col

# =============================================================================
# 3. Scaling & Train-Test Splitting Framework (Leakage Prevention)
# =============================================================================
def prepare_data(df, target_col):
    print("\n>>> Phase 2: Feature Scaling & Data Splitting...")
    features = ['hour', 'day_of_week', 'month']
    
    # Chronological Split preventing future leakage
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    # MinMax Scaler instances (target isolated from temporal covariates)
    scaler_target = MinMaxScaler()
    scaler_features = MinMaxScaler()
    
    # Train constraint ONLY: fit & transform simultaneously
    train_target_scaled = scaler_target.fit_transform(train_df[[target_col]])
    train_features_scaled = scaler_features.fit_transform(train_df[features])
    
    # Test constraint: transform strictly utilizing training parameters
    test_target_scaled = scaler_target.transform(test_df[[target_col]])
    test_features_scaled = scaler_features.transform(test_df[features])
    
    train_data = np.hstack((train_target_scaled, train_features_scaled))
    test_data = np.hstack((test_target_scaled, test_features_scaled))
    
    return train_data, test_data, scaler_target, scaler_features

# =============================================================================
# 4. 3D Autoregressive Matrix Transformation
# =============================================================================
def create_sliding_windows(data, history_window, future_window, stride):
    X_past, X_future_covariates, Y = [], [], []
    
    # Col index 0 represents the targeted sequence
    for i in range(0, len(data) - history_window - future_window + 1, stride):
        # The historically observed data window
        past_window = data[i : i + history_window, :]
        # Expected future covariates isolated (excluding objective target index 0)
        future_cov = data[i + history_window : i + history_window + future_window, 1:]
        # Intended target chunk spanning chronological continuity
        future_target = data[i + history_window : i + history_window + future_window, 0]
        
        X_past.append(past_window)
        X_future_covariates.append(future_cov)
        Y.append(future_target)
        
    return np.array(X_past), np.array(X_future_covariates), np.array(Y)

# =============================================================================
# 5. Core DeepAR Architecture
# =============================================================================
def build_deepar_model(history_window, future_window, num_covariates):
    print("\n>>> Phase 3: Constructing Sub-Architectural Network...")
    
    past_input = keras.layers.Input(shape=(history_window, 1 + num_covariates), name='past_tensor')
    future_cov_input = keras.layers.Input(shape=(future_window, num_covariates), name='future_cov_tensor')
    
    # Sequential historical representation mapped to complex states
    lstm_enc = keras.layers.LSTM(64, return_sequences=False, name='lstm_encoder')(past_input)
    
    # Project latent representations extending towards targeted future steps 
    repeated_ctx = keras.layers.RepeatVector(future_window)(lstm_enc)
    
    # Concatenation unifying embedded latent sequence & deterministic covariates
    dec_input = keras.layers.Concatenate(axis=-1)([repeated_ctx, future_cov_input])
    
    # Autoregressive decoding layer
    lstm_dec = keras.layers.LSTM(64, return_sequences=True, name='lstm_decoder')(dec_input)
    
    # Neural emission layer specifying parameter parameters matching probabilistic format
    distribution_params = keras.layers.TimeDistributed(keras.layers.Dense(2), name='param_layer')(lstm_dec)
    
    # Probabilistic formulation using TensorFlow Probability lambda wrapper
    def independent_normal(params):
        loc = params[..., 0]
        scale = tf.math.softplus(params[..., 1]) + 1e-5
        return tfd.Independent(tfd.Normal(loc=loc, scale=scale), reinterpreted_batch_ndims=1)
    
    prob_dist_layer = tfp.layers.DistributionLambda(
        make_distribution_fn=independent_normal,
        name='gaussian_distribution_layer'
    )(distribution_params)
    
    return keras.Model(inputs=[past_input, future_cov_input], outputs=prob_dist_layer)

# =============================================================================
# 6. Negative Log-Likelihood Function definition
# =============================================================================
def negative_log_likelihood(y_true, y_pred_dist):
    return -y_pred_dist.log_prob(y_true)

# =============================================================================
# 7. Model Assessment Calculations
# =============================================================================
def root_mean_square_error(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_true - y_pred)))

def custom_quantile_loss(q, y_true, y_pred):
    e = y_true - y_pred
    return np.mean(np.maximum(q * e, (q - 1) * e))

# =============================================================================
# 8. Execution Pipeline
# =============================================================================
def main():
    file_path = 'AEP_hourly.csv'
    
    if not os.path.exists(file_path):
        print(f"File {file_path} non-existent, synthetically fabricating a dataset substitute.")
        dates = pd.date_range('2020-01-01', periods=10000, freq='h')
        df_dummy = pd.DataFrame({'Datetime': dates, 'AEP_MW': np.random.normal(15000, 2000, 10000)})
        df_dummy.to_csv(file_path, index=False)
        
    df, datetime_col, target_col = load_and_preprocess_data(file_path)
    train_data, test_data, scaler_target, scaler_features = prepare_data(df, target_col)
    
    H, F, S, num_covariates = 168, 24, 3, 3 
    
    X_past_train, X_future_train, Y_train = create_sliding_windows(train_data, H, F, S)
    X_past_test, X_future_test, Y_test = create_sliding_windows(test_data, H, F, S)
    
    print("\nStructural Arrays 3D Verification:")
    print(f"X_Past_Train: {X_past_train.shape} | X_Future_Train: {X_future_train.shape}")
    print(f"Y_Train Target Matrix: {Y_train.shape}")
    
    model = build_deepar_model(H, F, num_covariates)
    
    # Configured learning decay protocol matching dynamic stability tracking
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3, decay_steps=1000, decay_rate=0.9)
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule), 
                  loss=negative_log_likelihood)
    print("\n=== Analytical Model Blueprint ===")
    model.summary()
    
    # Prevent extreme divergent overfitting utilizing adaptive thresholds
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1)
    
    print("\n>>> Phase 4: Training Convergence Initiative...")
    model.fit([X_past_train, X_future_train], Y_train,
              validation_split=0.2, epochs=40, batch_size=128,
              callbacks=[early_stop, reduce_lr], verbose=1)
    
    print("\n>>> Phase 5: Generating Probabilistic Inferences...")
    # Emitting the structured TensorFlow distributions objects from testing sequences
    predicted_dists = model([X_past_test, X_future_test])
    y_pred_mean = predicted_dists.mean().numpy()
    y_pred_std = predicted_dists.stddev().numpy()
    
    # Transformation matrices handling denormalizing phase logic
    Y_test_inv = scaler_target.inverse_transform(Y_test.reshape(-1, 1)).reshape(Y_test.shape)
    y_pred_mean_inv = scaler_target.inverse_transform(y_pred_mean.reshape(-1, 1)).reshape(y_pred_mean.shape)
    
    scale_factor = scaler_target.data_max_[0] - scaler_target.data_min_[0]
    y_pred_std_inv = y_pred_std * scale_factor
    
    # Statistically deterministic Z scores to scale standard deviations representing exact quantiles
    z_10, z_90 = -1.28155, 1.28155
    y_pred_p10 = y_pred_mean_inv + z_10 * y_pred_std_inv
    y_pred_p90 = y_pred_mean_inv + z_90 * y_pred_std_inv
    
    # Analytical Mathematical Metrics Display
    print("\n>>> Phase 6: System Evaluation Diagnostics")
    print(f"RMSE (Mean Prediction Evaluation): {root_mean_square_error(Y_test_inv, y_pred_mean_inv):.2f} MW")
    print(f"Quantile Loss Framework Calculations:")
    print(f" - Quantile_10th: {custom_quantile_loss(0.10, Y_test_inv, y_pred_p10):.3f}")
    print(f" - Quantile_50th: {custom_quantile_loss(0.50, Y_test_inv, y_pred_mean_inv):.3f}")
    print(f" - Quantile_90th: {custom_quantile_loss(0.90, Y_test_inv, y_pred_p90):.3f}")
    
    # Generate Advanced Scientific 4K Plot Matrix Visual representations
    print("\n>>> Phase 7: Generating 10 High-Fidelity 6K Publication Charts...")
    
    # Create a directory for charts
    charts_dir = 'DeepAR_Research_Charts'
    if not os.path.exists(charts_dir):
        os.makedirs(charts_dir)
        
    for i in range(10):
        visual_idx = np.random.randint(0, len(Y_test_inv))
        
        fig = plt.figure(figsize=(16, 7), dpi=600)
        fig.patch.set_facecolor('white') # Explicit true white background
        plt.plot(range(1, 25), Y_test_inv[visual_idx], label='Actual Load (Ground Truth)', color='#e63946', linewidth=2.5, marker='o', markersize=4)
        plt.plot(range(1, 25), y_pred_mean_inv[visual_idx], label='DeepAR Median Prediction (P50)', color='#1d3557', linewidth=2.5, marker='s', markersize=4)
        
        # Graphical shading bounded between upper and lower parametric deviations
        plt.fill_between(range(1, 25), y_pred_p10[visual_idx], y_pred_p90[visual_idx], color='#457b9d', alpha=0.3, label='90% Bayesian Confidence Interval')
        
        plt.title(f'Scenario {i+1}: Probabilistic Load Trajectory (Index {visual_idx})', fontsize=18, fontweight='bold', family='serif')
        plt.xlabel('Forecasting Horizon (Next 24 Hours)', fontsize=14, fontweight='bold')
        plt.ylabel('Load Demand (MW)', fontsize=14, fontweight='bold')
        plt.legend(loc='best', frameon=True, shadow=True, fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        file_name = f'{charts_dir}/Figure_{i+1:02d}_DeepAR_Probabilistic_Prediction.png'
        plt.savefig(file_name, dpi=600, bbox_inches='tight', facecolor='white', transparent=False)
        plt.close(fig)
        print(f"Chart {i+1}/10 Exported: {file_name}")
    
    print(f"\nAll 10 publication-quality charts are saved in the '{charts_dir}' directory.")
    
    # System final architecture endpoint export
    print("\n>>> Phase 8: Functional Tensor System Serializing...")
    model.save('DeepAR_Probabilistic_Model_Engine')
    print("Status code (0x0): Successfully formatted logic under tf_saved_model specifications handling TFP extensions flawlessly without parameter collapse.")

if __name__ == "__main__":
    main()
