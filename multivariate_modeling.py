"""
multivariate_modeling.py - Multivariate LSTM for ENSO Forecasting

This script implements a Multivariate Long Short-Term Memory (LSTM) neural network
to forecast Indonesian SST anomalies using both local history AND the Niño 3.4 Index
as an exogenous predictor.

Oceanographic Context:
- The Niño 3.4 Index is the primary ENSO indicator, measured in the Central Pacific
- El Niño events (Niño 3.4 > 1.0°C) typically cause cooling in Indonesian waters
- La Niña events (Niño 3.4 < -1.0°C) typically cause warming in Indonesian waters
- This teleconnection makes Niño 3.4 a powerful predictor for Indonesian SST

Model Design:
- Multivariate input: [Indonesian SST Anomaly, Niño 3.4 Index]
- Uses 12-month lookback to capture seasonal cycles and ENSO persistence
- Predicts only Indonesian SST Anomaly (univariate output)

Author: Data Science Portfolio Project
Date: December 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Ensure reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# ============================================================================
# CONFIGURATION
# ============================================================================
# Data files
SST_INDO_FILE = "data/processed/sst_indo_clean.csv"
NINO34_FILE = "data/raw/nina34.anom.data.txt"

# Data parameters
LOOKBACK = 12  # 12 months lookback (captures full seasonal cycle)
TRAIN_RATIO = 0.8  # 80% train, 20% test

# Model hyperparameters (tuned for multivariate input)
INPUT_SIZE = 2        # 2 input features: Indonesian SST + Niño 3.4
HIDDEN_SIZE = 32      # Slightly larger for multivariate data
NUM_LAYERS = 1        # Single LSTM layer to prevent overfitting
OUTPUT_SIZE = 1       # Predict only Indonesian SST anomaly
DROPOUT = 0.0         # No dropout for small dataset

# Training parameters
EPOCHS = 150          # Number of training epochs
BATCH_SIZE = 4        # Small batch size for small dataset
LEARNING_RATE = 0.005 # Moderate learning rate

# El Niño threshold for visualization
ELNINO_THRESHOLD = 1.0  # Niño 3.4 > 1.0°C indicates El Niño

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# DATA LOADING AND PARSING
# ============================================================================

def load_indonesian_sst(filepath: str) -> pd.DataFrame:
    """
    Load the preprocessed Indonesian SST anomaly data.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame with date index and SST anomaly column
    """
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    
    print(f"✓ Loaded Indonesian SST data: {len(df)} records")
    print(f"  Date range: {df.index[0].strftime('%Y-%m')} to {df.index[-1].strftime('%Y-%m')}")
    
    return df


def load_nino34_data(filepath: str) -> pd.DataFrame:
    """
    Parse the NOAA Niño 3.4 anomaly data file.
    
    The file format is:
    - Line 1: Header with year range (skip)
    - Lines 2-79: Year followed by 12 monthly values (Jan-Dec)
    - Lines 80+: Footer metadata (skip)
    - Missing values are encoded as -99.99
    
    Args:
        filepath: Path to the nina34.anom.data.txt file
        
    Returns:
        DataFrame with date index and nino34 column
    """
    records = []
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Skip header (line 0) and process data lines
    for line in lines[1:]:
        parts = line.split()
        
        # Check if this is a valid data line (starts with a 4-digit year)
        if len(parts) < 13:
            continue
        
        try:
            year = int(parts[0])
            if year < 1900 or year > 2100:
                continue
        except ValueError:
            continue
        
        # Extract 12 monthly values
        for month_idx, value_str in enumerate(parts[1:13]):
            try:
                value = float(value_str)
                
                # Skip missing values (-99.99)
                if value < -90:
                    continue
                
                # Create date for this month (1st of month)
                date = pd.Timestamp(year=year, month=month_idx + 1, day=1)
                records.append({'date': date, 'nino34': value})
                
            except ValueError:
                continue
    
    # Create DataFrame
    df = pd.DataFrame(records)
    df = df.set_index('date')
    df = df.sort_index()
    
    print(f"✓ Loaded Niño 3.4 data: {len(df)} records")
    print(f"  Date range: {df.index[0].strftime('%Y-%m')} to {df.index[-1].strftime('%Y-%m')}")
    
    return df


def merge_datasets(sst_df: pd.DataFrame, nino_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge Indonesian SST and Niño 3.4 datasets on date index.
    
    Args:
        sst_df: Indonesian SST DataFrame
        nino_df: Niño 3.4 DataFrame
        
    Returns:
        Merged DataFrame with both features
    """
    # Inner join on date index to keep only overlapping periods
    merged = sst_df.join(nino_df, how='inner')
    
    # Keep only the columns we need
    merged = merged[['sst_anomaly', 'nino34']]
    
    # Handle any remaining missing values
    initial_len = len(merged)
    merged = merged.dropna()
    
    if len(merged) < initial_len:
        print(f"  ⚠ Dropped {initial_len - len(merged)} rows with missing values")
    
    print(f"\n✓ Merged dataset: {len(merged)} records")
    print(f"  Date range: {merged.index[0].strftime('%Y-%m')} to {merged.index[-1].strftime('%Y-%m')}")
    print(f"  Features: {list(merged.columns)}")
    
    return merged


# ============================================================================
# PREPROCESSING FOR MULTIVARIATE LSTM
# ============================================================================

def create_multivariate_sequences(data: np.ndarray, lookback: int) -> tuple:
    """
    Create sliding window sequences for multivariate time series forecasting.
    
    For each window of 'lookback' months, we use ALL features as input
    but predict only the first feature (Indonesian SST anomaly) for next month.
    
    Example with lookback=12 and 2 features:
        Input X: [[sst_1, nino_1], [sst_2, nino_2], ..., [sst_12, nino_12]]
        Target y: sst_13 (only Indonesian SST)
    
    Args:
        data: Normalized feature matrix (samples × n_features)
        lookback: Number of past time steps to use as input
        
    Returns:
        Tuple of (X, y) where:
        - X has shape [samples, lookback, n_features]
        - y has shape [samples, 1] (only the target variable)
    """
    X, y = [], []
    
    for i in range(len(data) - lookback):
        # Input: lookback months of ALL features
        X.append(data[i:(i + lookback), :])
        # Target: next month's Indonesian SST anomaly (column 0 only)
        y.append(data[i + lookback, 0])
    
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    
    return X, y


def prepare_dataloaders(X: np.ndarray, y: np.ndarray, 
                        train_ratio: float, batch_size: int) -> tuple:
    """
    Split data into train/test sets and create PyTorch DataLoaders.
    
    Note: For time series, we use a simple temporal split (not shuffled)
    to maintain temporal ordering and avoid lookahead bias.
    
    Args:
        X: Input sequences
        y: Target values
        train_ratio: Proportion of data for training
        batch_size: Batch size for DataLoader
        
    Returns:
        Tuple of (train_loader, test_loader, train_size, X_test_t, y_test_t)
    """
    train_size = int(len(X) * train_ratio)
    
    # Temporal split (no shuffling to preserve time order)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"\nData Split:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Testing samples:  {len(X_test)}")
    
    # Handle edge case: if test set is too small
    if len(X_test) < 2:
        raise ValueError(f"Test set too small ({len(X_test)} samples). "
                        f"Need more data or lower train ratio.")
    
    # Convert to PyTorch tensors
    X_train_t = torch.FloatTensor(X_train).to(DEVICE)
    y_train_t = torch.FloatTensor(y_train).to(DEVICE)
    X_test_t = torch.FloatTensor(X_test).to(DEVICE)
    y_test_t = torch.FloatTensor(y_test).to(DEVICE)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    
    # Adjust batch size if larger than dataset
    effective_batch_size = min(batch_size, len(train_dataset))
    
    # Create DataLoaders (shuffle only training data)
    train_loader = DataLoader(train_dataset, batch_size=effective_batch_size, 
                              shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), 
                             shuffle=False)
    
    return train_loader, test_loader, train_size, X_test_t, y_test_t


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class MultivariateLSTM(nn.Module):
    """
    Multivariate LSTM for time series forecasting with exogenous features.
    
    Architecture:
    - Input: [batch, seq_len, n_features] where n_features = 2
    - Single LSTM layer (prevents overfitting on small data)
    - Linear output layer for single-variable regression
    
    The model learns to combine information from both:
    1. Indonesian SST history (local memory)
    2. Niño 3.4 history (ENSO teleconnection signal)
    """
    
    def __init__(self, input_size: int = 2, hidden_size: int = 32, 
                 num_layers: int = 1, output_size: int = 1):
        """
        Initialize the Multivariate LSTM model.
        
        Args:
            input_size: Number of input features (2: SST Indo + Niño 3.4)
            hidden_size: Number of LSTM hidden units
            num_layers: Number of stacked LSTM layers
            output_size: Number of output features (1: Indonesian SST only)
        """
        super(MultivariateLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM Layer
        # batch_first=True means input shape is (batch, seq, features)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0 if num_layers == 1 else DROPOUT
        )
        
        # Linear layer to map LSTM output to prediction
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        LSTM returns:
        - output: (batch, seq_len, hidden_size) - output at each time step
        - (h_n, c_n): final hidden and cell states
        
        We only use the output from the last time step for prediction.
        """
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Take only the last time step's output
        # lstm_out shape: (batch, seq_len, hidden_size)
        # We want: (batch, hidden_size)
        last_output = lstm_out[:, -1, :]
        
        # Pass through fully connected layer
        prediction = self.fc(last_output)
        
        return prediction


# ============================================================================
# TRAINING
# ============================================================================

def train_model(model: nn.Module, train_loader: DataLoader, 
                epochs: int, learning_rate: float) -> list:
    """
    Train the Multivariate LSTM model.
    
    Args:
        model: LSTM model instance
        train_loader: DataLoader for training data
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        
    Returns:
        List of training losses per epoch
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler (reduce LR when loss plateaus)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, verbose=False
    )
    
    train_losses = []
    
    print("\n" + "=" * 50)
    print("TRAINING STARTED")
    print("=" * 50)
    print(f"Device: {DEVICE}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {learning_rate}")
    print("-" * 50)
    
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for X_batch, y_batch in train_loader:
            # Forward pass
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        # Average loss for the epoch
        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)
        
        # Update learning rate scheduler
        scheduler.step(avg_loss)
        
        # Print progress every 25 epochs
        if (epoch + 1) % 25 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1:3d}/{epochs}] | Loss: {avg_loss:.6f} | LR: {current_lr:.6f}")
    
    print("-" * 50)
    print("TRAINING COMPLETE")
    
    return train_losses


# ============================================================================
# EVALUATION AND VISUALIZATION
# ============================================================================

def evaluate_model(model: nn.Module, X_test: torch.Tensor, y_test: torch.Tensor,
                   scaler: MinMaxScaler, train_size: int, 
                   dates: pd.DatetimeIndex, merged_df: pd.DataFrame) -> tuple:
    """
    Evaluate the model on test data and calculate metrics.
    
    Args:
        model: Trained LSTM model
        X_test: Test input sequences
        y_test: Test target values
        scaler: Fitted MinMaxScaler for inverse transform
        train_size: Number of training samples (for date indexing)
        dates: Original date index
        merged_df: Original merged DataFrame (for Niño 3.4 values)
        
    Returns:
        Tuple of (actual values, predicted values, test dates, test nino34 values)
    """
    model.eval()
    
    with torch.no_grad():
        predictions_scaled = model(X_test).cpu().numpy()
    
    # Get actual values
    actual_scaled = y_test.cpu().numpy()
    
    # Inverse transform - need to handle multivariate scaler
    # Create dummy array with same shape as original features
    n_features = scaler.n_features_in_
    
    # For predictions (only first column is the target)
    pred_full = np.zeros((len(predictions_scaled), n_features))
    pred_full[:, 0] = predictions_scaled.flatten()
    predictions_original = scaler.inverse_transform(pred_full)[:, 0].reshape(-1, 1)
    
    # For actual values
    actual_full = np.zeros((len(actual_scaled), n_features))
    actual_full[:, 0] = actual_scaled.flatten()
    actual_original = scaler.inverse_transform(actual_full)[:, 0].reshape(-1, 1)
    
    # Calculate metrics
    mse = np.mean((predictions_original - actual_original) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions_original - actual_original))
    
    # Correlation coefficient
    correlation = np.corrcoef(actual_original.flatten(), 
                               predictions_original.flatten())[0, 1]
    
    print("\n" + "=" * 50)
    print("EVALUATION METRICS (Test Set)")
    print("=" * 50)
    print(f"MSE:         {mse:.4f} °C²")
    print(f"RMSE:        {rmse:.4f} °C")
    print(f"MAE:         {mae:.4f} °C")
    print(f"Correlation: {correlation:.4f}")
    
    # Get corresponding dates and Niño 3.4 values for test set
    test_start_idx = train_size + LOOKBACK
    test_dates = dates[test_start_idx:test_start_idx + len(actual_original)]
    test_nino34 = merged_df['nino34'].iloc[test_start_idx:test_start_idx + len(actual_original)].values
    
    return actual_original, predictions_original, test_dates, test_nino34


def plot_results(actual: np.ndarray, predicted: np.ndarray, 
                 dates: pd.DatetimeIndex, nino34: np.ndarray,
                 train_losses: list):
    """
    Create comprehensive visualization with El Niño period highlighting.
    
    Args:
        actual: Actual SST anomaly values
        predicted: Model predictions
        dates: Corresponding dates
        nino34: Niño 3.4 values for highlighting El Niño periods
        train_losses: List of training losses
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), dpi=100)
    
    # -------------------------------------------------------------------------
    # Plot 1: Actual vs Predicted with El Niño Highlighting
    # -------------------------------------------------------------------------
    ax1 = axes[0]
    
    # Highlight El Niño periods (Niño 3.4 > 1.0)
    elnino_mask = nino34 > ELNINO_THRESHOLD
    
    # Find continuous El Niño periods
    if np.any(elnino_mask):
        in_elnino = False
        start_idx = None
        for i, is_elnino in enumerate(elnino_mask):
            if is_elnino and not in_elnino:
                # Start of El Niño period
                start_idx = i
                in_elnino = True
            elif not is_elnino and in_elnino:
                # End of El Niño period
                ax1.axvspan(dates[start_idx], dates[i-1], 
                           alpha=0.3, color='coral', label='El Niño (Niño 3.4 > 1.0)' if start_idx == np.where(elnino_mask)[0][0] else '')
                in_elnino = False
        # Handle case where El Niño continues to end
        if in_elnino:
            ax1.axvspan(dates[start_idx], dates[-1], 
                       alpha=0.3, color='coral', label='El Niño (Niño 3.4 > 1.0)' if start_idx == np.where(elnino_mask)[0][0] else '')
    
    # Plot actual and predicted
    ax1.plot(dates, actual, 'b-', linewidth=2, 
             label='Actual Indonesian SST', marker='o', markersize=4)
    ax1.plot(dates, predicted, 'r--', linewidth=2, 
             label='Predicted (Multivariate LSTM)', marker='s', markersize=4)
    
    # Add zero reference line
    ax1.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
    
    # Styling
    ax1.set_xlabel('Date', fontsize=11)
    ax1.set_ylabel('SST Anomaly (°C)', fontsize=11)
    ax1.set_title('Multivariate LSTM Forecast: Indonesian SST Anomaly\n'
                  '(Using Indonesian SST + Niño 3.4 as Predictors)', 
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Rotate x-axis labels
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # -------------------------------------------------------------------------
    # Plot 2: Training Loss Curve
    # -------------------------------------------------------------------------
    ax2 = axes[1]
    
    epochs_range = range(1, len(train_losses) + 1)
    ax2.plot(epochs_range, train_losses, 'g-', linewidth=2)
    
    # Mark minimum loss
    min_loss = min(train_losses)
    min_epoch = train_losses.index(min_loss) + 1
    ax2.scatter([min_epoch], [min_loss], color='red', s=100, zorder=5,
                label=f'Min Loss: {min_loss:.4f} (Epoch {min_epoch})')
    
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('MSE Loss', fontsize=11)
    ax2.set_title('Training Loss Curve', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1, len(train_losses))
    
    plt.tight_layout()
    plt.savefig('output/figures/multivariate_lstm_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n✓ Results plot saved to: output/figures/multivariate_lstm_results.png")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """
    Main function to run the complete Multivariate LSTM forecasting pipeline.
    
    Pipeline:
    1. Load and parse both data sources
    2. Merge datasets on date index
    3. Normalize all features to [-1, 1] range
    4. Create multivariate sliding window sequences
    5. Build and train Multivariate LSTM model
    6. Evaluate on test set
    7. Visualize results with El Niño highlighting
    """
    print("=" * 70)
    print("MULTIVARIATE LSTM SST ANOMALY FORECASTING")
    print("Indonesian Maritime Region + Niño 3.4 Predictor")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # Step 1: Load Data Sources
    # -------------------------------------------------------------------------
    print("\n[Step 1/6] Loading data sources...")
    
    sst_df = load_indonesian_sst(SST_INDO_FILE)
    nino_df = load_nino34_data(NINO34_FILE)
    
    # -------------------------------------------------------------------------
    # Step 2: Merge Datasets
    # -------------------------------------------------------------------------
    print("\n[Step 2/6] Merging datasets...")
    merged_df = merge_datasets(sst_df, nino_df)
    
    # Store original dates for later
    dates = merged_df.index
    
    # Get feature matrix
    feature_data = merged_df[['sst_anomaly', 'nino34']].values
    
    print(f"\n  Feature matrix shape: {feature_data.shape}")
    print(f"  SST Anomaly range: {feature_data[:, 0].min():.2f}°C to {feature_data[:, 0].max():.2f}°C")
    print(f"  Niño 3.4 range:    {feature_data[:, 1].min():.2f}°C to {feature_data[:, 1].max():.2f}°C")
    
    # -------------------------------------------------------------------------
    # Step 3: Normalize Data
    # -------------------------------------------------------------------------
    print("\n[Step 3/6] Normalizing features to [-1, 1] range...")
    
    scaler = MinMaxScaler(feature_range=(-1, 1))
    feature_scaled = scaler.fit_transform(feature_data)
    
    print(f"  Scaled SST range:    {feature_scaled[:, 0].min():.2f} to {feature_scaled[:, 0].max():.2f}")
    print(f"  Scaled Niño 3.4 range: {feature_scaled[:, 1].min():.2f} to {feature_scaled[:, 1].max():.2f}")
    
    # -------------------------------------------------------------------------
    # Step 4: Create Multivariate Sequences
    # -------------------------------------------------------------------------
    print(f"\n[Step 4/6] Creating multivariate sequences with {LOOKBACK}-month lookback...")
    
    X, y = create_multivariate_sequences(feature_scaled, LOOKBACK)
    
    print(f"  Total sequences created: {len(X)}")
    print(f"  Input shape (X): {X.shape} [samples, lookback, n_features]")
    print(f"  Target shape (y): {y.shape} [samples, 1]")
    
    # Check if we have enough data
    if len(X) < 5:
        raise ValueError(f"Not enough data! Only {len(X)} sequences created. "
                        f"Need at least 5 for meaningful train/test split.")
    
    # Create DataLoaders
    train_loader, test_loader, train_size, X_test, y_test = prepare_dataloaders(
        X, y, TRAIN_RATIO, BATCH_SIZE
    )
    
    # -------------------------------------------------------------------------
    # Step 5: Build and Train Model
    # -------------------------------------------------------------------------
    print(f"\n[Step 5/6] Building Multivariate LSTM model...")
    
    model = MultivariateLSTM(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=OUTPUT_SIZE
    ).to(DEVICE)
    
    # Print model architecture
    print(f"\n  Model Architecture:")
    print(f"  {'-' * 40}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Input size:     {INPUT_SIZE} (Indonesian SST + Niño 3.4)")
    print(f"  Hidden size:    {HIDDEN_SIZE}")
    print(f"  LSTM layers:    {NUM_LAYERS}")
    print(f"  Output size:    {OUTPUT_SIZE} (Indonesian SST only)")
    print(f"  Total params:   {total_params:,}")
    print(f"  {'-' * 40}")
    
    # Train the model
    train_losses = train_model(model, train_loader, EPOCHS, LEARNING_RATE)
    
    # -------------------------------------------------------------------------
    # Step 6: Evaluate and Visualize
    # -------------------------------------------------------------------------
    print("\n[Step 6/6] Evaluating model on test set...")
    
    actual, predicted, test_dates, test_nino34 = evaluate_model(
        model, X_test, y_test, scaler, train_size, dates, merged_df
    )
    
    # Generate plots with El Niño highlighting
    plot_results(actual, predicted, test_dates, test_nino34, train_losses)
    
    # -------------------------------------------------------------------------
    # Summary and Oceanographic Interpretation
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\nFiles Generated:")
    print(f"  • multivariate_lstm_results.png - Predictions with El Niño highlighting")
    print(f"\nModel trained on {train_size} samples, tested on {len(actual)} samples")
    
    # Check for El Niño periods in test set
    elnino_periods = test_nino34[test_nino34 > ELNINO_THRESHOLD]
    print(f"\n  El Niño periods in test set: {len(elnino_periods)} months with Niño 3.4 > {ELNINO_THRESHOLD}°C")
    
    print("\n" + "-" * 70)
    print("OCEANOGRAPHIC INTERPRETATION")
    print("-" * 70)
    print("""
The Multivariate LSTM model leverages the teleconnection between:
• Indonesian SST and the Niño 3.4 Index (Central Pacific)

Key patterns the model learns:
• During El Niño (Niño 3.4 > 1.0): Indonesian waters typically cool
• During La Niña (Niño 3.4 < -1.0): Indonesian waters typically warm
• The Niño 3.4 signal often leads Indonesian SST by 1-3 months

The shaded coral regions on the plot indicate El Niño periods.
Compare predicted vs actual during these periods to assess if the
model successfully captures the ENSO teleconnection effect.
""")
    
    return model, scaler


if __name__ == "__main__":
    model, scaler = main()
