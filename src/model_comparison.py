import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set chart style
sns.set_theme(style="whitegrid")

# --- CONFIGURAZIONE ---
# IDs degli oggetti simmetrici in LineMOD (Eggbox, Glue)
SYMMETRIC_IDS = [10, 11] 

def compute_mixed_metrics(df):
    """
    Crea una colonna 'LineMOD Accuracy (%)' che seleziona:
    - ADD-S-0.1d Accuracy per oggetti simmetrici (ID 10, 11)
    - ADD-0.1d Accuracy per oggetti non simmetrici
    """
    # Verifica che le colonne esistano
    if 'ADD-0.1d Accuracy (%)' not in df.columns or 'ADD-S-0.1d Accuracy (%)' not in df.columns:
        print("Warning: Colonne ADD/ADD-S mancanti. Impossibile calcolare metrica mista.")
        return df

    # Logica vettoriale con numpy where
    df['LineMOD Accuracy (%)'] = np.where(
        df['Object ID'].isin(SYMMETRIC_IDS),
        df['ADD-S-0.1d Accuracy (%)'],  # Valore se True (Simmetrico)
        df['ADD-0.1d Accuracy (%)']     # Valore se False (Non simmetrico)
    )
    return df

def load_and_compare_models(model_files: dict):
    """
    Loads CSV files, computes the correct hybrid metric and creates a single DataFrame.
    """
    df_list = []
    
    for model_name, file_path in model_files.items():
        if not os.path.exists(file_path):
            print(f"Warning: File not found for {model_name} -> {file_path}")
            continue
            
        df = pd.read_csv(file_path)
        df['Model'] = model_name
        
        # Calcola la metrica corretta subito dopo il caricamento
        df = compute_mixed_metrics(df)
        
        df_list.append(df)
    
    if not df_list:
        raise ValueError("No files loaded successfully.")
        
    full_df = pd.concat(df_list, ignore_index=True)
    return full_df

def save_plot(fig, save_path):
    """ Helper per salvare il plot se il path è definito """
    if save_path:
        # Crea la cartella se non esiste
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Plot salvato in: {save_path}")

def plot_leaderboard(df, metric='LineMOD Accuracy (%)', ascending=False, save_path=None):
    """
    Summary plot: average of the chosen metric across all objects.
    Defaults to the new 'LineMOD Accuracy (%)'.
    """
    # Calculate mean for each model
    leaderboard = df.groupby('Model')[metric].mean().reset_index()
    leaderboard = leaderboard.sort_values(by=metric, ascending=not ascending) 
    
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=leaderboard, x='Model', y=metric, hue='Model', palette='viridis', legend=False)
    
    # Add values on top of bars
    for i in ax.containers:
        ax.bar_label(i, fmt='%.2f', padding=3)
        
    plt.title(f"Global Comparison: {metric} (Mean across all objects)")
    plt.ylabel("Mean Value")
    if 'Accuracy' in metric:
        plt.ylim(0, 105)
        
    plt.tight_layout()
    
    # Gestione salvataggio
    if save_path:
        save_plot(plt.gcf(), save_path)
    else:
        plt.show()
    plt.close() # Chiude la figura per liberare memoria
    
    return leaderboard

def plot_per_object_comparison(df, metric='LineMOD Accuracy (%)', save_path=None):
    """
    Detailed plot: comparison for each individual object using the mixed metric.
    """
    plt.figure(figsize=(14, 6))
    
    sns.barplot(
        data=df, 
        x='Object Name', 
        y=metric, 
        hue='Model', 
        palette='magma',
        edgecolor='black'
    )
    
    plt.title(f"Per-Object Detail: {metric} (ADD-S for Sym, ADD for others)")
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Model')
    
    if 'Accuracy' in metric:
        plt.ylim(0, 105)
        plt.axhline(100, color='gray', linestyle='--', alpha=0.5)
        
    plt.tight_layout()
    
    if save_path:
        save_plot(plt.gcf(), save_path)
    else:
        plt.show()
    plt.close()

# --- FUNZIONI GT vs YOLO ---

def load_paired_data(gt_files: dict, yolo_files: dict):
    df_list = []
    
    def _load(files_dict, source_type):
        for model_name, file_path in files_dict.items():
            if not os.path.exists(file_path):
                print(f"Warning: File not found for {model_name} ({source_type})")
                continue
            df = pd.read_csv(file_path)
            df['Model'] = model_name
            df['Source'] = source_type
            
            # Calcolo la metrica mista anche qui!
            df = compute_mixed_metrics(df)
            
            df_list.append(df)

    _load(gt_files, 'GT')
    _load(yolo_files, 'YOLO')
    
    if not df_list:
        raise ValueError("No files loaded.")
        
    return pd.concat(df_list, ignore_index=True)

def plot_gt_vs_yolo_summary(df, metric='LineMOD Accuracy (%)', save_path=None):
    summary = df.groupby(['Model', 'Source'])[metric].mean().reset_index()
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=summary, 
        x='Model', 
        y=metric, 
        hue='Source', 
        palette={'GT': '#2ecc71', 'YOLO': '#e74c3c'},
        alpha=0.9
    )
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', padding=3, fontsize=9)
        
    plt.title(f"Impact of Detection Quality: GT vs YOLO ({metric})")
    if 'Accuracy' in metric:
        plt.ylim(0, 110)
        
    plt.legend(title='BBox Source')
    plt.tight_layout()
    
    if save_path:
        save_plot(plt.gcf(), save_path)
    else:
        plt.show()
    plt.close()

def plot_yolo_drop_heatmap(df, metric='LineMOD Accuracy (%)', save_path=None):
    grouped = df.groupby(['Model', 'Object Name', 'Source'])[metric].mean().reset_index()
    pivoted = grouped.pivot(index=['Model', 'Object Name'], columns='Source', values=metric).reset_index()
    
    pivoted['Drop'] = pivoted['GT'] - pivoted['YOLO']
    heatmap_data = pivoted.pivot(index='Model', columns='Object Name', values='Drop')
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(
        heatmap_data, 
        annot=True, 
        fmt=".1f", 
        cmap="Reds", 
        cbar_kws={'label': f'Accuracy Drop (GT - YOLO) %'},
        linewidths=.5
    )
    plt.title(f"YOLO Sensitivity Analysis: Which objects suffer most?")
    plt.tight_layout()
    
    if save_path:
        save_plot(plt.gcf(), save_path)
    else:
        plt.show()
    plt.close()
    
    return heatmap_data

def get_gt_vs_yolo_table(df, metric='LineMOD Accuracy (%)'):
    """
    Restituisce una tabella comparativa usando la metrica corretta.
    """
    summary = df.groupby(['Model', 'Source'])[metric].mean().unstack()
    
    if 'GT' in summary.columns and 'YOLO' in summary.columns:
        summary['Drop (%)'] = summary['GT'] - summary['YOLO']
        # Evita divisione per zero
        summary['Retained Perf (%)'] = np.where(
            summary['GT'] > 0, 
            (summary['YOLO'] / summary['GT']) * 100, 
            0
        )
        summary = summary.sort_values(by='YOLO', ascending=False)
    
    return summary.round(2)