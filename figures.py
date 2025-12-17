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
        
    ax.grid(True, linestyle='--', alpha=0.6, which='both')
    
    ax.xaxis.label.set_size(14) 
    ax.yaxis.label.set_size(14)

    ax.tick_params(axis='both', which='major', labelsize=12)

def plot_single_scan(df, title, filename):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(df['Angle'], df['Intensity_cps'], color='blue', linewidth=1, label='Dane')
    ax.set_xlabel(r'$\omega$ [deg]')
    ax.set_ylabel('Natężenie [cps]')
    
    _apply_style(ax, title, filename)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Zapisano wykres: {filename}")

def plot_combined_scans(data_list, title, filename):
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, df in data_list:
        ax.plot(df['Angle'], df['Intensity_cps'], label=label, linewidth=1)
    
    ax.set_xlabel(r'$\omega$ [deg]')
    ax.set_ylabel('Natężenie [cps]')
    
    _apply_style(ax, title, filename)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Zapisano wykres: {filename}")

def plot_combined_shifted_scans(data_list, title, filename):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for label, df in data_list:
        max_idx = df['Intensity_cps'].idxmax()
        max_angle = df['Angle'].iloc[max_idx]
        shifted_angle = df['Angle'] - max_angle
        ax.plot(shifted_angle, df['Intensity_cps'], label=f"{label}", linewidth=1)
    
    ax.set_xlabel(r'Względny kąt [deg]]')
    ax.set_ylabel('Natężenie [cps]')
    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    
    _apply_style(ax, title, filename)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Zapisano wykres przesunięty: {filename}")

# ZMODYFIKOWANA FUNKCJA
def plot_fit(df, popt, func, title, filename, param_text="", log_scale=False):
    fig, ax = plt.subplots(figsize=(11, 4))
    
    # 1. Rysowanie danych
    ax.plot(df['Angle'], df['Intensity_cps'], 'b.', markersize=3, label='Dane pomiarowe')
    
    # 2. Generowanie modelu (obsługa ZOOM z CONFIG)
    x_min, x_max = df['Angle'].min(), df['Angle'].max()
    
    if hasattr(CONFIG, 'PLOT_ZOOM_FWHM_MULTIPLIER') and CONFIG.PLOT_ZOOM_FWHM_MULTIPLIER:
        if len(popt) >= 3:
            mean = popt[1]
            sigma = popt[2]
            fwhm = 2.355 * sigma
            span = CONFIG.PLOT_ZOOM_FWHM_MULTIPLIER * fwhm
            
            zoom_min = mean - span
            zoom_max = mean + span
            
            ax.set_xlim(zoom_min, zoom_max)
            x_min, x_max = zoom_min, zoom_max

    x_fit = np.linspace(x_min, x_max, 1000)
    y_fit = func(x_fit, *popt)
    ax.plot(x_fit, y_fit, 'r-', linewidth=2, label='Dopasowanie')
    
    # 3. Rysowanie tła
    if len(popt) == 5:
        m, c = popt[3], popt[4]
        y_bg = m * x_fit + c
        ax.plot(x_fit, y_bg, 'k--', linewidth=1, alpha=0.5, label='Tło liniowe')

    # 4. Obsługa skali logarytmicznej
    if log_scale:
        ax.set_yscale('log')
        # Opcjonalnie: ustawienie minimalnego limitu Y, żeby log nie wariował przy < 0
        min_pos = df[df['Intensity_cps'] > 0]['Intensity_cps'].min()
        if not np.isnan(min_pos):
             ax.set_ylim(bottom=min_pos * 0.8) # Trochę poniżej najmniejszej wartości

    ax.set_xlabel(r'$2\theta$ [deg]')
    ax.set_ylabel('Natężenie [cps]')
    
    if CONFIG.SHOW_PARAM_TEXT and param_text:
        ax.annotate(param_text, xy=(0.02, 0.98), xycoords='axes fraction',
                     verticalalignment='top', bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    
    _apply_style(ax, title, filename)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Zapisano wykres ({'LOG' if log_scale else 'LIN'}): {filename}")