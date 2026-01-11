import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing import preprocess_pipeline
from model import (
    train_regression_model, 
    save_model, 
    evaluate_regression_model,
    save_detailed_metrics
)
from visualization import (
    plot_earthquake_map_cartopy, 
    plot_training_history
)
from sklearn.preprocessing import StandardScaler
import joblib


def main():
    """Execute full training and evaluation pipeline"""
    
    print("\n" + "="*60)
    print("EARTHQUAKE PREDICTION - ADVANCED MODEL v2.0")
    print("="*60)
    
    # Configuration
    DATA_PATH = 'data/database.csv'
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    MODEL_TYPE = 'advanced'
    EPOCHS = 150
    BATCH_SIZE = 64
    USE_FEATURE_ENGINEERING = True
    
    if not os.path.exists(DATA_PATH):
        print(f"\nERROR: Dataset not found at {DATA_PATH}")
        return
    
    # Preprocessing and feature engineering
    print("\n[STEP 1/5] Enhanced Data Preprocessing + Feature Engineering")
    X_train, X_test, y_train, y_test, data = preprocess_pipeline(
        filepath=DATA_PATH, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE,
        use_feature_engineering=USE_FEATURE_ENGINEERING
    )
    
    print(f"\nFeature engineering summary:")
    print(f"  Original features: 3")
    print(f"  Engineered features: {X_train.shape[1]}")
    
    # Normalize features
    print("\n[STEP 2/5] Feature Scaling")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    os.makedirs('results', exist_ok=True)
    joblib.dump(scaler, 'results/scaler.pkl')
    print("Scaler saved to results/scaler.pkl")
    
    # Generate visualizations
    print("\n[STEP 3/5] Generating Earthquake Distribution Map")
    plot_earthquake_map_cartopy(data)
    
    # Train model
    print("\n[STEP 4/5] Advanced Neural Network Training")
    model, history, metrics = train_regression_model(
        X_train_scaled, y_train, 
        X_test_scaled, y_test,
        model_type=MODEL_TYPE,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )
    
    # Evaluate and save results
    print("\n[STEP 5/5] Evaluation and Results")
    detailed_metrics = evaluate_regression_model(model, X_test_scaled, y_test)
    
    save_model(model, 'results/earthquake_model.keras')
    plot_training_history(history)
    save_detailed_metrics(metrics, detailed_metrics)
    
    # Summary
    print("\n" + "="*60)
    print("ADVANCED PIPELINE EXECUTION COMPLETE")
    print("="*60)
    print("\nGenerated outputs in results/ folder:")
    print("  - earthquake_model.keras")
    print("  - best_model.keras")
    print("  - scaler.pkl")
    print("  - earthquake_map.png")
    print("  - training_history.png")
    print("  - model_metrics.txt")
    
    print(f"\nOverall Performance:")
    print(f"  Test MSE: {metrics['test_mse']:.4f}")
    print(f"  Test MAE: {metrics['test_mae']:.4f}")
    
    print(f"\nR² Scores:")
    for target, target_metrics in detailed_metrics.items():
        r2 = target_metrics['r2']
        print(f"  {target}: {r2:.4f}")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()