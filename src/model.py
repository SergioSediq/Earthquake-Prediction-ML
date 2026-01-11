import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Concatenate
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os


def create_advanced_model(input_dim, learning_rate=0.001):
    """
    Multi-branch neural network with separate processing paths
    
    Architecture:
    - Shared layers: 256 -> 128 (with BatchNorm + Dropout)
    - Magnitude branch: 64 -> 32 -> 1
    - Depth branch: 64 -> 32 -> 1
    """
    inputs = Input(shape=(input_dim,))
    
    # Shared feature extraction
    x = Dense(256, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Magnitude prediction branch
    mag_branch = Dense(64, activation='relu', name='mag_branch_1')(x)
    mag_branch = Dropout(0.2)(mag_branch)
    mag_branch = Dense(32, activation='relu', name='mag_branch_2')(mag_branch)
    mag_branch = Dropout(0.2)(mag_branch)
    magnitude_output = Dense(1, activation='linear', name='magnitude')(mag_branch)
    
    # Depth prediction branch
    depth_branch = Dense(64, activation='relu', name='depth_branch_1')(x)
    depth_branch = Dropout(0.2)(depth_branch)
    depth_branch = Dense(32, activation='relu', name='depth_branch_2')(depth_branch)
    depth_branch = Dropout(0.2)(depth_branch)
    depth_output = Dense(1, activation='linear', name='depth')(depth_branch)
    
    outputs = Concatenate()([magnitude_output, depth_output])
    
    model = Model(inputs=inputs, outputs=outputs)
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model


def create_deep_model(input_dim, learning_rate=0.001):
    """Single-path deep network"""
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.4),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(32, activation='relu'),
        Dropout(0.2),
        
        Dense(16, activation='relu'),
        Dense(2, activation='linear')
    ])
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model


def train_regression_model(X_train, y_train, X_test, y_test, 
                          model_type='advanced', epochs=150, batch_size=64):
    """Train model with callbacks for optimization"""
    
    print("\n" + "="*50)
    print("TRAINING ADVANCED REGRESSION MODEL")
    print("="*50)
    
    input_dim = X_train.shape[1]
    
    if model_type == 'advanced':
        print("\nUsing advanced multi-branch architecture:")
        print("  Shared layers: 256 → 128")
        print("  Magnitude branch: 64 → 32 → 1")
        print("  Depth branch: 64 → 32 → 1")
        print("  Regularization: BatchNorm + Dropout")
        model = create_advanced_model(input_dim=input_dim)
    else:
        print("\nUsing deep single-path architecture:")
        print("  Layers: 256 → 128 → 64 → 32 → 16 → 2")
        print("  Regularization: BatchNorm + Dropout")
        model = create_deep_model(input_dim=input_dim)
    
    print(f"  Input features: {input_dim}")
    print(f"  Optimizer: Adam")
    print(f"  Loss: MSE")
    print(f"  Metrics: MAE")
    print(f"  Batch size: {batch_size}")
    print(f"  Max epochs: {epochs}")
    
    # Training callbacks
    os.makedirs('results', exist_ok=True)
    
    callbacks = [
        ModelCheckpoint(
            'results/best_model.keras',
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=0
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    print("\nCallbacks enabled:")
    print("  - ModelCheckpoint: Save best model")
    print("  - EarlyStopping: Stop if no improvement (patience=20)")
    print("  - ReduceLROnPlateau: Reduce LR when stuck (patience=7)")
    
    print("\nTraining model...\n")
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(X_test, y_test),
        callbacks=callbacks
    )
    
    # Evaluation
    print("\n" + "-"*50)
    print("MODEL EVALUATION")
    print("-"*50)
    
    y_train_pred = model.predict(X_train, verbose=0)
    y_test_pred = model.predict(X_test, verbose=0)
    
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    print(f"Training MSE: {train_mse:.4f}")
    print(f"Training MAE: {train_mae:.4f}")
    print(f"Test MSE: {test_mse:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    
    if train_mse < test_mse * 0.7:
        print("\nWarning: Possible overfitting detected")
    elif abs(train_mse - test_mse) < 100:
        print("\nGood generalization: Train and test MSE are close")
    
    metrics = {
        'train_mse': train_mse,
        'train_mae': train_mae,
        'test_mse': test_mse,
        'test_mae': test_mae
    }
    
    return model, history, metrics


def evaluate_regression_model(model, X_test, y_test):
    """Calculate comprehensive regression metrics"""
    
    print("\n" + "-"*50)
    print("DETAILED REGRESSION METRICS")
    print("-"*50)
    
    y_pred = model.predict(X_test, verbose=0)
    
    targets = ['Magnitude', 'Depth']
    overall_metrics = {}
    
    for i, target in enumerate(targets):
        y_true = y_test.iloc[:, i].values if hasattr(y_test, 'iloc') else y_test[:, i]
        y_predicted = y_pred[:, i]
        
        mae = mean_absolute_error(y_true, y_predicted)
        mse = mean_squared_error(y_true, y_predicted)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_predicted)
        
        mean_true = np.mean(y_true)
        mape = (mae / mean_true) * 100 if mean_true != 0 else float('inf')
        
        median_error = np.median(np.abs(y_true - y_predicted))
        percentile_90_error = np.percentile(np.abs(y_true - y_predicted), 90)
        
        print(f"\n{target}:")
        print(f"  MAE: {mae:.4f}")
        print(f"  Median Error: {median_error:.4f}")
        print(f"  90th Percentile Error: {percentile_90_error:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R² Score: {r2:.4f}")
        print(f"  Mean Actual Value: {mean_true:.4f}")
        print(f"  Mean Percentage Error: {mape:.2f}%")
        
        overall_metrics[target] = {
            'mae': mae,
            'median_error': median_error,
            'percentile_90_error': percentile_90_error,
            'rmse': rmse,
            'r2': r2,
            'mape': mape
        }
    
    return overall_metrics


def save_model(model, filepath='results/earthquake_model.keras'):
    """Save trained model"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    model.save(filepath)
    print(f"\nModel saved to {filepath}")


def save_detailed_metrics(metrics, detailed_metrics, output_path='results/model_metrics.txt'):
    """Export comprehensive metrics to text file"""
    
    print("\nSaving comprehensive metrics report...")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("EARTHQUAKE PREDICTION - ADVANCED MODEL METRICS\n")
        f.write("="*60 + "\n\n")
        
        f.write("OVERALL PERFORMANCE:\n")
        f.write("-"*60 + "\n")
        f.write(f"Training MSE: {metrics['train_mse']:.4f}\n")
        f.write(f"Training MAE: {metrics['train_mae']:.4f}\n")
        f.write(f"Test MSE: {metrics['test_mse']:.4f}\n")
        f.write(f"Test MAE: {metrics['test_mae']:.4f}\n\n")
        
        f.write("DETAILED METRICS BY TARGET:\n")
        f.write("-"*60 + "\n")
        
        for target, target_metrics in detailed_metrics.items():
            f.write(f"\n{target}:\n")
            f.write(f"  MAE: {target_metrics['mae']:.4f}\n")
            f.write(f"  Median Error: {target_metrics['median_error']:.4f}\n")
            f.write(f"  90th Percentile Error: {target_metrics['percentile_90_error']:.4f}\n")
            f.write(f"  RMSE: {target_metrics['rmse']:.4f}\n")
            f.write(f"  R² Score: {target_metrics['r2']:.4f}\n")
            f.write(f"  Mean Percentage Error: {target_metrics['mape']:.2f}%\n")
        
        f.write("\n" + "="*60 + "\n")
    
    print(f"Metrics report saved to {output_path}")