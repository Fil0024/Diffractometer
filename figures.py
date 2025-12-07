# figures.py
import matplotlib.pyplot as plt
import numpy as np
import CONFIG

def _apply_style(ax, title, filename):
    """Pomocnicza funkcja aplikująca style z CONFIG."""
    plot_key = filename.split('/')[-1].replace('.pdf', '').replace('.png', '')
    final_title = CONFIG.CUSTOM_TITLES.get(plot_key, title)

    if CONFIG.SHOW_TITLES:
        ax.set_title(final_title)
    
    if CONFIG.SHOW_LEGENDS:
        ax.legend()
        
    ax.grid(True, linestyle='--', alpha=0.6)

def plot_single_scan(df, title, filename):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(df['Angle'], df['Intensity_cps'], color='blue', linewidth=1, label='Dane')
    ax.set_xlabel('Angle (deg)')
    ax.set_ylabel('Intensity (cps)')
    
    _apply_style(ax, title, filename)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Zapisano wykres: {filename}")

def plot_combined_scans(data_list, title, filename):
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, df in data_list:
        ax.plot(df['Angle'], df['Intensity_cps'], label=label, linewidth=1)
    
    ax.set_xlabel('Angle (deg)')
    ax.set_ylabel('Intensity (cps)')
    
    _apply_style(ax, title, filename)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Zapisano wykres: {filename}")

def plot_combined_shifted_scans(data_list, title, filename):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for label, df in data_list:
        # Znalezienie maksimum
        max_idx = df['Intensity_cps'].idxmax()
        max_angle = df['Angle'].iloc[max_idx]
        
        # Przesunięcie
        shifted_angle = df['Angle'] - max_angle
        
        ax.plot(shifted_angle, df['Intensity_cps'], label=f"{label}", linewidth=1)
    
    ax.set_xlabel('Relative Angle (deg)')
    ax.set_ylabel('Intensity (cps)')
    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    
    _apply_style(ax, title, filename)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Zapisano wykres przesunięty: {filename}")

def plot_fit(df, popt, func, title, filename, param_text=""):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Dane
    ax.plot(df['Angle'], df['Intensity_cps'], 'b.', markersize=3, label='Dane pomiarowe')
    
    # Model pełny
    x_fit = np.linspace(df['Angle'].min(), df['Angle'].max(), 1000)
    y_fit = func(x_fit, *popt)
    ax.plot(x_fit, y_fit, 'r-', linewidth=2, label='Dopasowanie (Gauss + Tło)')
    
    # Rysowanie samego tła (jeśli model to Gauss+Liniowa)
    # popt = [amp, mean, sigma, m, c]
    if len(popt) == 5:
        m, c = popt[3], popt[4]
        y_bg = m * x_fit + c
        ax.plot(x_fit, y_bg, 'k--', linewidth=1, alpha=0.5, label='Tło liniowe')

    ax.set_xlabel('2Theta (deg)')
    ax.set_ylabel('Intensity (cps)')
    
    if CONFIG.SHOW_PARAM_TEXT and param_text:
        ax.annotate(param_text, xy=(0.02, 0.98), xycoords='axes fraction',
                     verticalalignment='top', bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    
    _apply_style(ax, title, filename)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Zapisano wykres dopasowania: {filename}")