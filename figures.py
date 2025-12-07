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
        
    ax.grid(True, linestyle='--', alpha=0.6, which='both') # which='both' dla log scale

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
        max_idx = df['Intensity_cps'].idxmax()
        max_angle = df['Angle'].iloc[max_idx]
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
    
    # 1. Rysowanie danych
    ax.plot(df['Angle'], df['Intensity_cps'], 'b.', markersize=3, label='Dane pomiarowe')
    
    # 2. Generowanie modelu
    # Jeśli zoom jest włączony, generujemy gęstsze punkty w zakresie zoomu
    # w przeciwnym razie w całym zakresie danych
    x_min, x_max = df['Angle'].min(), df['Angle'].max()
    
    # Obsługa zakresu na podstawie FWHM
    # popt = [amp, mean, sigma, m, c]
    xlim_set = False
    
    if hasattr(CONFIG, 'PLOT_ZOOM_FWHM_MULTIPLIER') and CONFIG.PLOT_ZOOM_FWHM_MULTIPLIER:
        # Sprawdzamy czy popt ma sensowną długość (czy zawiera mean i sigma)
        if len(popt) >= 3:
            mean = popt[1]
            sigma = popt[2]
            fwhm = 2.355 * sigma
            span = CONFIG.PLOT_ZOOM_FWHM_MULTIPLIER * fwhm
            
            # Nowe limity
            zoom_min = mean - span
            zoom_max = mean + span
            
            # Upewniamy się, że nie wychodzimy drastycznie poza zakres danych (opcjonalne)
            # Ale zazwyczaj chcemy widzieć tło, więc zostawiamy tak jak jest.
            ax.set_xlim(zoom_min, zoom_max)
            xlim_set = True
            
            # Aktualizujemy zakres do rysowania krzywej, żeby była gładka w powiększeniu
            x_min, x_max = zoom_min, zoom_max

    x_fit = np.linspace(x_min, x_max, 1000)
    y_fit = func(x_fit, *popt)
    ax.plot(x_fit, y_fit, 'r-', linewidth=2, label='Dopasowanie (Gauss + Tło)')
    
    # 3. Rysowanie tła
    if len(popt) == 5:
        m, c = popt[3], popt[4]
        y_bg = m * x_fit + c
        ax.plot(x_fit, y_bg, 'k--', linewidth=1, alpha=0.5, label='Tło liniowe')

    # 4. Skala logarytmiczna
    if hasattr(CONFIG, 'PLOT_LOG_SCALE') and CONFIG.PLOT_LOG_SCALE:
        ax.set_yscale('log')
        # Przy skali logarytmicznej warto uważać na wartości <= 0
        # Matplotlib zazwyczaj sobie radzi, ale można ustawić dolny limit
        # np. min_positive = df[df['Intensity_cps']>0]['Intensity_cps'].min()
        # ax.set_ylim(bottom=min_positive)

    ax.set_xlabel('2Theta (deg)')
    ax.set_ylabel('Intensity (cps)')
    
    # Ramka z parametrami
    if CONFIG.SHOW_PARAM_TEXT and param_text:
        # Jeśli jest log scale, czasem tekst lepiej umieścić inaczej, 
        # ale standardowe xy=(0.02, 0.98) w coords='axes fraction' jest niezależne od skali danych.
        ax.annotate(param_text, xy=(0.02, 0.98), xycoords='axes fraction',
                     verticalalignment='top', bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    
    _apply_style(ax, title, filename)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Zapisano wykres dopasowania: {filename}")