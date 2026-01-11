import matplotlib.pyplot as plt
import numpy as np
import os


def plot_earthquake_map_cartopy(data, output_path='results/earthquake_map.png'):
    """Generate global earthquake distribution map using Cartopy"""
    
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        
        print("\nGenerating earthquake world map...")
        
        plt.figure(figsize=(15, 10))
        ax = plt.axes(projection=ccrs.PlateCarree())
        
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.LAND, facecolor='coral')
        ax.add_feature(cfeature.OCEAN, facecolor='aqua')
        
        longitudes = data["Longitude"].tolist()
        latitudes = data["Latitude"].tolist()
        
        ax.scatter(longitudes, latitudes, s=2, c='blue', alpha=0.6, 
                   transform=ccrs.PlateCarree())
        
        ax.set_global()
        ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        
        plt.title("Global Earthquake Distribution", fontsize=16, fontweight='bold')
        
        os.makedirs('results', exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Map saved to {output_path}")
        return True
        
    except ImportError:
        print("Cartopy not available, skipping map visualization")
        return False


def plot_training_history(history, output_path='results/training_history.png'):
    """Plot training and validation metrics over epochs"""
    
    print("\nGenerating training history plots...")
    
    _, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    if 'mae' in history.history:
        # Regression metrics
        axes[0].plot(history.history['mae'], label='Training MAE', color='blue', linewidth=2)
        axes[0].plot(history.history['val_mae'], label='Validation MAE', color='red', linewidth=2)
        axes[0].set_title('Mean Absolute Error', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('MAE')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(history.history['loss'], label='Training MSE', color='blue', linewidth=2)
        axes[1].plot(history.history['val_loss'], label='Validation MSE', color='red', linewidth=2)
        axes[1].set_title('Mean Squared Error', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MSE')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        # Classification metrics fallback
        axes[0].plot(history.history['accuracy'], label='Training Accuracy', color='blue', linewidth=2)
        axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', color='red', linewidth=2)
        axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(history.history['loss'], label='Training Loss', color='blue', linewidth=2)
        axes[1].plot(history.history['val_loss'], label='Validation Loss', color='red', linewidth=2)
        axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training history saved to {output_path}")


def save_metrics_report(metrics, output_path='results/model_metrics.txt'):
    """Export model performance metrics to text file"""
    
    print("\nSaving metrics report...")
    
    os.makedirs('results', exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("="*50 + "\n")
        f.write("EARTHQUAKE PREDICTION MODEL - PERFORMANCE METRICS\n")
        f.write("="*50 + "\n\n")
        
        if 'train_mse' in metrics:
            f.write("TRAINING SET:\n")
            f.write(f"  MSE: {metrics['train_mse']:.4f}\n")
            f.write(f"  MAE: {metrics['train_mae']:.4f}\n\n")
            
            f.write("TEST SET:\n")
            f.write(f"  MSE: {metrics['test_mse']:.4f}\n")
            f.write(f"  MAE: {metrics['test_mae']:.4f}\n\n")
        else:
            f.write("TRAINING SET:\n")
            f.write(f"  Accuracy: {metrics['train_acc']:.4f}\n")
            f.write(f"  Loss: {metrics['train_loss']:.4f}\n\n")
            
            f.write("TEST SET:\n")
            f.write(f"  Accuracy: {metrics['test_acc']:.4f}\n")
            f.write(f"  Loss: {metrics['test_loss']:.4f}\n\n")
        
        f.write("="*50 + "\n")
    
    print(f"Metrics report saved to {output_path}")