# figures.py
import matplotlib.pyplot as plt
import numpy as np

def plot_single_scan(df, title, filename):
    plt.figure(figsize=(8, 6))
    plt.plot(df['Angle'], df['Intensity_cps'], color='blue', linewidth=1)
    plt.title(title)
    plt.xlabel('Angle (deg)')
    plt.ylabel('Intensity (cps)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Zapisano wykres: {filename}")

def plot_combined_scans(data_list, title, filename):
    """
    data_list: lista krotek (label, df)
    """
    plt.figure(figsize=(10, 6))
    for label, df in data_list:
        plt.plot(df['Angle'], df['Intensity_cps'], label=label, linewidth=1)
    
    plt.title(title)
    plt.xlabel('Angle (deg)')
    plt.ylabel('Intensity (cps)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Zapisano wykres: {filename}")

def plot_fit(df, popt, func, title, filename, param_text=""):
    """
    Rysuje dane i dopasowaną krzywą Gaussa.
    """
    plt.figure(figsize=(8, 6))
    
    # Dane pomiarowe
    plt.plot(df['Angle'], df['Intensity_cps'], 'b.', label='Dane pomiarowe')
    
    # Krzywa dopasowania
    x_fit = np.linspace(df['Angle'].min(), df['Angle'].max(), 1000)
    y_fit = func(x_fit, *popt)
    plt.plot(x_fit, y_fit, 'r-', linewidth=2, label='Dopasowanie Gaussa')
    
    plt.title(title)
    plt.xlabel('2Theta (deg)')
    plt.ylabel('Intensity (cps)')
    
    # Dodanie tekstu z wynikami na wykresie
    if param_text:
        plt.annotate(param_text, xy=(0.05, 0.95), xycoords='axes fraction',
                     verticalalignment='top', bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Zapisano wykres dopasowania: {filename}")
