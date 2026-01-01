import matplotlib.pyplot as plt
import os

def plot_losses(train_losses, test_losses, output_path=None):
    """
    Plots training and test losses over epochs.

    Args:
        train_losses (list): List of training losses.
        test_losses (list): List of test losses.
        output_path (str, optional): Path to save the plot. If None, the plot is shown but not saved.
    """
    plt.figure(figsize=(10, 6))
    epochs_range = range(1, len(test_losses)+1)
    plt.plot(epochs_range, train_losses, 'b-o', label='Training Loss', linewidth=2, markersize=6)
    plt.plot(epochs_range, test_losses, 'r-s', label='Test Loss', linewidth=2, markersize=6)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training vs Test Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved: {output_path}")
    else:
        plt.show()
    
    plt.close()