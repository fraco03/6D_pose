import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os


# Set chart style
sns.set_theme(style="whitegrid")

SYMMETRIC_OBJECTS = [10, 11]

def load_data_as_dataframe(files_dict):
    """Load multiple CSV files into a single pandas DataFrame with a model identifier.

    Args:
        files_dict (dict): A dictionary where keys are model names and values are file paths to CSV files.

    Returns:
        pd.DataFrame: A concatenated DataFrame containing data from all CSV files with an additional 'Model' column.
    """
    dataframes = []
    for model_name, file_path in files_dict.items():
        df = pd.read_csv(file_path)
        df['Model'] = model_name
        dataframes.append(df)
    full_df = pd.concat(dataframes, ignore_index=True)
    return full_df

"""
Index(['Object ID', 'Object Name', 'Diameter (mm)', 'Mean ADD (mm)',
       'Mean ADD-S (mm)', 'Mean ADD-Rot (mm)', 'Mean ADD-S-Rot (mm)',
       'ADD-0.1d Accuracy (%)', 'ADD-S-0.1d Accuracy (%)', 'Model'],
      dtype='object')
"""

def preprocess_dataframe(df):
    """Preprocess the DataFrame by providing ACCURACY and mean-ADD
    based on object class.
    """
    # Example preprocessing: Calculate mean accuracy per model
    # ADD-01d Accuracy (%) if not in SYMMETRIC_OBJECTS else ADD-S Accuracy (%)
    # Mean ADD (mm) if not in SYMMETRIC_OBJECTS else Mean ADD-S (mm)

    new_df = df.copy()

    new_df['ACCURACY'] = new_df.apply(
        lambda row: row['ADD-S-0.1d Accuracy (%)'] if row['Object ID'] in SYMMETRIC_OBJECTS else row['ADD-0.1d Accuracy (%)'],
        axis=1
    )

    new_df['mean-ADD'] = new_df.apply(
        lambda row: row['Mean ADD-S (mm)'] if row['Object ID'] in SYMMETRIC_OBJECTS else row['Mean ADD (mm)'],
        axis=1
    )

    new_df.drop(columns=[
        'Mean ADD (mm)', 'Mean ADD-S (mm)',
        'ADD-0.1d Accuracy (%)', 'ADD-S-0.1d Accuracy (%)'
    ], inplace=True)

    new_df.rename(columns={
        'ACCURACY': 'ADD-0.1d Accuracy (%)',
        'mean-ADD': 'Mean ADD (mm)'
    }, inplace=True)

    new_df = new_df[['Object ID', 'Object Name', 'Mean ADD (mm)',
                        'ADD-0.1d Accuracy (%)', 'Model']]

    return new_df


"""
Index(['Object ID', 'Object Name', 'Mean ADD (mm)', 'ADD-0.1d Accuracy (%)',
       'Model'],
      dtype='object')
"""


# Set global style
sns.set_theme(style="whitegrid")

def plot_leaderboard(df, metric, ascending=False, title=None, save_path=None):
    """
    Plots a leaderboard comparing models based on the average of a specific metric.
    
    Args:
        df (pd.DataFrame): The preprocessed DataFrame containing 'Model' and the metric column.
        metric (str): The column name to use for ranking (e.g., 'ADD-0.1d Accuracy (%)' or 'Mean ADD (mm)').
        ascending (bool): If True, lower is better (e.g., for Mean ADD). 
                          If False, higher is better (e.g., for Accuracy).
        title (str, optional): Custom title for the chart.
        save_path (str, optional): File path to save the image.
    """
    # 1. Group by Model and calculate the mean of the chosen metric
    leaderboard = df.groupby('Model')[metric].mean().reset_index()
    
    # 2. Sort the values based on the 'ascending' parameter
    leaderboard = leaderboard.sort_values(by=metric, ascending=ascending)
    
    # 3. Create the plot
    plt.figure(figsize=(8, 6))
    
    # Use a color palette (viridis maps well to ordered data)
    ax = sns.barplot(
        data=leaderboard, 
        x='Model', 
        y=metric, 
        hue='Model', 
        palette='viridis', 
        legend=False
    )
    
    # 4. Add labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3, fontsize=10)
        
    # 5. Styling
    final_title = title if title else f"Model Leaderboard: {metric}"
    plt.title(final_title, fontsize=14, fontweight='bold')
    plt.ylabel(metric, fontsize=12)
    plt.xlabel("Model", fontsize=12)
    
    # If plotting accuracy, fix the Y-axis to 0-100 for clarity
    if 'Accuracy' in metric:
        plt.ylim(0, 105)
        
    plt.tight_layout()
    
    # 6. Save or Show
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Chart saved to: {save_path}")

    plt.show()
    plt.close()


def plot_per_object_comparison(df, metric, title=None, save_path=None):
    """
    Plots a detailed comparison per object for a specific metric.
    
    Args:
        df (pd.DataFrame): The preprocessed DataFrame.
        metric (str): The column name to visualize.
        title (str, optional): Custom title.
        save_path (str, optional): File path to save the image.
    """
    # 1. Sort by Object ID to ensure consistent ordering on the X-axis (Ape -> Can -> Cat...)
    if 'Object ID' in df.columns:
        df_sorted = df.sort_values(by='Object ID')
    else:
        df_sorted = df.sort_values(by='Object Name')

    # 2. Create the plot
    plt.figure(figsize=(14, 7))
    
    sns.barplot(
        data=df_sorted, 
        x='Object Name', 
        y=metric, 
        hue='Model', 
        palette='magma', 
        edgecolor='black',
        alpha=0.9
    )
    
    # 3. Styling
    final_title = title if title else f"Per-Object Comparison: {metric}"
    plt.title(final_title, fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, fontsize=11)
    plt.ylabel(metric, fontsize=12)
    
    # Move legend outside the plot area
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', title='Model', fontsize=10)
    
    # Formatting for Accuracy charts
    if 'Accuracy' in metric:
        plt.ylim(0, 110)
        plt.axhline(100, color='gray', linestyle='--', alpha=0.5) # Reference line at 100%

    plt.tight_layout()
    
    # 4. Save or Show
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Chart saved to: {save_path}")

    plt.show()
    plt.close()

def latex_leaderboard_table(df, metric, ascending=False):
    """
    Generates a LaTeX formatted table for the leaderboard based on a specific metric.
    
    Args:
        df (pd.DataFrame): The preprocessed DataFrame.
        metric (str): The column name to rank models by.
        ascending (bool): If True, lower is better (e.g., for Mean ADD). 
                          If False, higher is better (e.g., for Accuracy).
    Returns:
        str: LaTeX formatted table as a string.
    # 1. Group by Model and calculate the mean of the chosen metric
    """

    leaderboard = df.groupby('Model')[metric].mean().reset_index()
    
    # 2. Sort the values based on the 'ascending' parameter
    leaderboard = leaderboard.sort_values(by=metric, ascending=ascending)
    
    # 3. Generate LaTeX table
    latex_table = leaderboard.to_latex(index=False, float_format="%.2f")
    
    return latex_table


def combine_gt_and_yolo_dfs(gt_df, yolo_df):
    """
    Merges the Ground Truth and YOLO DataFrames into a single structure for comparison.
    
    Args:
        gt_df (pd.DataFrame): DataFrame containing results using GT Bounding Boxes.
        yolo_df (pd.DataFrame): DataFrame containing results using YOLO Bounding Boxes.
        
    Returns:
        pd.DataFrame: A combined DataFrame with a new 'Source' column ('GT' or 'YOLO').
    """
    # Work on copies to avoid modifying original variables
    gt_copy = gt_df.copy()
    gt_copy['Source'] = 'GT'
    
    yolo_copy = yolo_df.copy()
    yolo_copy['Source'] = 'YOLO'
    
    # Concatenate
    combined_df = pd.concat([gt_copy, yolo_copy], ignore_index=True)
    return combined_df

def plot_gt_vs_yolo_summary(combined_df, metric='ADD-0.1d Accuracy (%)', save_path=None):
    """
    Plots a grouped bar chart comparing the overall average performance 
    of GT vs. YOLO for each model.
    
    Args:
        combined_df (pd.DataFrame): The merged dataframe output from combine_gt_and_yolo_dfs.
        metric (str): The metric to compare (default: Accuracy).
        save_path (str, optional): Path to save the figure.
    """
    # 1. Calculate the mean per Model and Source
    summary = combined_df.groupby(['Model', 'Source'])[metric].mean().reset_index()
    
    plt.figure(figsize=(10, 6))
    
    # 2. Create the Bar Plot
    # We use a custom palette: Green for GT (Ideal), Red/Orange for YOLO (Real)
    ax = sns.barplot(
        data=summary, 
        x='Model', 
        y=metric, 
        hue='Source', 
        palette={'GT': '#2ecc71', 'YOLO': '#e74c3c'},
        alpha=0.9,
        edgecolor='black'
    )
    
    # 3. Add labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', padding=3, fontsize=10, fontweight='bold')
        
    # 4. Styling
    plt.title(f"Impact of Detection Quality: GT vs YOLO ({metric})", fontsize=14, fontweight='bold')
    plt.xlabel("Model", fontsize=12)
    plt.ylabel(metric, fontsize=12)
    plt.legend(title='BBox Source')
    
    if 'Accuracy' in metric:
        plt.ylim(0, 115) # Extra space for labels
        
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Summary plot saved to: {save_path}")
    plt.show()
    plt.close()

def plot_yolo_drop_heatmap(combined_df, metric='ADD-0.1d Accuracy (%)', save_path=None):
    """
    Generates a Heatmap showing the PERFORMANCE DROP (GT - YOLO) for each object and model.
    Positive values indicate how much accuracy was lost due to YOLO errors.
    
    Args:
        combined_df (pd.DataFrame): The merged dataframe.
        metric (str): The metric to calculate the drop for.
        save_path (str, optional): Path to save the figure.
    """
    # 1. Group by Model, Object, and Source to ensure uniqueness
    # Then unstack 'Source' to compare columns side-by-side
    pivot_df = combined_df.pivot_table(
        index=['Model', 'Object Name'], 
        columns='Source', 
        values=metric
    ).reset_index()
    
    # 2. Calculate the Drop (GT - YOLO)
    # Positive value = Loss of accuracy
    # Negative value = YOLO somehow performed better (rare, but possible due to noise)
    pivot_df['Drop'] = pivot_df['GT'] - pivot_df['YOLO']
    
    # 3. Pivot again for the Heatmap format: 
    # Rows = Model, Columns = Object Name, Values = Drop
    heatmap_data = pivot_df.pivot(index='Model', columns='Object Name', values='Drop')
    
    # 4. Create Heatmap
    plt.figure(figsize=(12, 6))
    
    sns.heatmap(
        heatmap_data, 
        annot=True,       # Show numbers
        fmt=".1f",        # 1 decimal place
        cmap="Reds",      # Red indicates "bad" (high drop)
        linewidths=.5,    # Grid lines
        cbar_kws={'label': f'Accuracy Drop (GT - YOLO) %'}
    )
    
    plt.title(f"Sensitivity Analysis: Performance Drop using YOLO vs GT", fontsize=14, fontweight='bold')
    plt.xlabel("Object Class", fontsize=12)
    plt.ylabel("Model", fontsize=12)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Heatmap saved to: {save_path}")
    
    plt.show()
    plt.close()


def generate_latex_tables(df, caption_suffix=""):
    """
    Takes a preprocessed DataFrame and prints LaTeX code for two tables:
    1. Accuracy (%)
    2. Mean ADD (mm)
    
    Args:
        df: The preprocessed DataFrame (with columns 'Model', 'Object Name', etc.)
        caption_suffix: Optional string to append to the caption (e.g., "using YOLO detections")
    """
    
    # Define the specific order for the columns
    target_order = ['RGB', 'RGBD', 'PointNet', 'DenseFusion']
    
    # --- 1. DATA PREPARATION ---
    # Create pivot tables: Rows=Objects, Columns=Models
    
    # --- A. Accuracy Table ---
    acc_pivot = df.pivot_table(
        index='Object Name', 
        columns='Model', 
        values='ADD-0.1d Accuracy (%)'
    )
    
    # Reorder columns based on target_order
    # (We filter to ensure we only include models that actually exist in the DF)
    existing_order = [m for m in target_order if m in acc_pivot.columns]
    # Append any other models that might be in the DF but not in our target list
    remaining_cols = [c for c in acc_pivot.columns if c not in existing_order]
    acc_pivot = acc_pivot[existing_order + remaining_cols]
    
    # Add Average row
    acc_pivot.loc['Average'] = acc_pivot.mean()
    
    # --- B. Mean ADD Table ---
    add_pivot = df.pivot_table(
        index='Object Name', 
        columns='Model', 
        values='Mean ADD (mm)'
    )
    
    # Reorder columns (Apply same logic as above)
    existing_order_add = [m for m in target_order if m in add_pivot.columns]
    remaining_cols_add = [c for c in add_pivot.columns if c not in existing_order_add]
    add_pivot = add_pivot[existing_order_add + remaining_cols_add]
    
    # Add Average row
    add_pivot.loc['Average'] = add_pivot.mean()

    # --- 2. LATEX GENERATION ---
    
    print("-" * 30)
    print(f"📄 LATEX TABLE 1: ACCURACY (%) {caption_suffix}")
    print("-" * 30)
    
    latex_acc = acc_pivot.round(2).to_latex(
        caption=f"Per-object Accuracy Comparison (ADD-0.1d/ADD-S) {caption_suffix}. Higher is better.",
        label="tab:accuracy_results",
        column_format="l" + "c" * len(acc_pivot.columns), # Left align names, center numbers
        float_format="%.2f"
    )
    print(latex_acc)
    
    print("\n" + "-" * 30)
    print(f"📄 LATEX TABLE 2: MEAN ADD ERROR (mm) {caption_suffix}")
    print("-" * 30)
    
    latex_add = add_pivot.round(2).to_latex(
        caption=f"Per-object Mean ADD Error (mm) {caption_suffix}. Lower is better.",
        label="tab:add_error_results",
        column_format="l" + "c" * len(add_pivot.columns),
        float_format="%.2f"
    )
    print(latex_add)