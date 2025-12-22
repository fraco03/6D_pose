# import matplotlib.pyplot as plt

# # Create plots directory
# # plots_dir = "plots"
# plots_dir = checkpoint_dir
# os.makedirs(plots_dir, exist_ok=True)

# # Plot 1: Training vs Test Loss
# plt.figure(figsize=(10, 6))
# epochs_range = range(1, len(test_losses)+1)
# plt.plot(range(1, len(train_losses)+1), train_losses, 'b-o', label='Training Loss', linewidth=2, markersize=6)
# plt.plot(range(1, len(test_losses)+1), test_losses, 'r-s', label='Test Loss', linewidth=2, markersize=6)
# plt.xlabel('Epoch', fontsize=12)
# plt.ylabel('Loss', fontsize=12)
# plt.title('Training vs Test Loss', fontsize=14, fontweight='bold')
# plt.legend(fontsize=11)
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# loss_plot_path = os.path.join(plots_dir, "loss_comparison.png")
# plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
# print(f"✅ Plot saved: {loss_plot_path}")
# plt.show()

# # Plot 2: Only Training Loss
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, len(train_losses)+1), train_losses, 'b-o', linewidth=2, markersize=6)
# plt.xlabel('Epoch', fontsize=12)
# plt.ylabel('Training Loss', fontsize=12)
# plt.title('Training Loss Over Epochs', fontsize=14, fontweight='bold')
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# train_loss_path = os.path.join(plots_dir, "training_loss.png")
# plt.savefig(train_loss_path, dpi=300, bbox_inches='tight')
# print(f"✅ Plot saved: {train_loss_path}")
# plt.show()

# # Plot 3: Only Test Loss
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, len(test_losses)+1), test_losses, 'r-s', linewidth=2, markersize=6)
# plt.xlabel('Epoch', fontsize=12)
# plt.ylabel('Test Loss', fontsize=12)
# plt.title('Test Loss Over Epochs', fontsize=14, fontweight='bold')
# plt.grid(True, alpha=0.3)
# plt.axhline(y=best_test_loss, color='g', linestyle='--', label=f'Best: {best_test_loss:.4f}', linewidth=2)
# plt.legend(fontsize=11)
# plt.tight_layout()
# test_loss_path = os.path.join(plots_dir, "test_loss.png")
# plt.savefig(test_loss_path, dpi=300, bbox_inches='tight')
# print(f"✅ Plot saved: {test_loss_path}")
# plt.show()

# print(f"\n✅ All plots saved in '{plots_dir}' directory!")

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