# main.py
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os
import sys

# Importy
try:
    import CONFIG
    import utils
    import figures
except ImportError as e:
    print("BŁĄD: Brakuje plików pomocniczych.")
    print(f"Szczegóły: {e}")
    sys.exit(1)

# --- KONFIGURACJA ŚCIEŻEK ---
DATA_DIR = 'data'
RESULTS_DIR = 'results'

# --- FUNKCJE FIZYCZNE I MATEMATYCZNE ---

def gaussian_linear(x, amp, mean, sigma, m, c):
    """
    Model: Gauss + Funkcja liniowa (tło).
    """
    return amp * np.exp(-(x - mean)**2 / (2 * sigma**2)) + m * x + c

def fit_peak(df):
    """
    Dopasowuje model Gauss + Tło liniowe.
    """
    if df.empty or 'Angle' not in df or 'Intensity_cps' not in df: return None
    x = df['Angle'].values
    y = df['Intensity_cps'].values
    if len(x) < 5: return None

    try:
        # 1. Estymacja tła liniowego (na podstawie pierwszych i ostatnich punktów)
        n_bg = max(1, int(len(x) * 0.1)) # 10% punktów z brzegów
        x_bg = np.concatenate([x[:n_bg], x[-n_bg:]])
        y_bg = np.concatenate([y[:n_bg], y[-n_bg:]])
        
        # Proste dopasowanie liniowe do tła: y = mx + c
        if len(x_bg) > 1:
            p_bg = np.polyfit(x_bg, y_bg, 1)
            m_guess, c_guess = p_bg[0], p_bg[1]
        else:
            m_guess, c_guess = 0, min(y)

        # 2. Odjęcie tła, żeby oszacować parametry Gaussa
        y_pure = y - (m_guess * x + c_guess)
        
        # 3. Estymacja parametrów Gaussa
        mean_guess = x[np.argmax(y_pure)] # Pozycja maksimum po odjęciu tła
        amp_guess = np.max(y_pure)        # Amplituda
        
        # Szacowanie szerokości (sigma)
        # Obliczamy moment drugiego rzędu wokół średniej, waony intensywnością (zgrubnie)
        # albo po prostu arbitralnie mała wartość jeśli peak jest ostry
        total_intensity = np.sum(y_pure[y_pure > 0])
        if total_intensity > 0:
            sigma_guess = np.sqrt(np.abs(np.sum((x - mean_guess)**2 * y_pure) / total_intensity))
        else:
            sigma_guess = 0.1
            
        if sigma_guess == 0 or np.isnan(sigma_guess): sigma_guess = 0.05

        # Initial guess: [amp, mean, sigma, m, c]
        p0 = [amp_guess, mean_guess, sigma_guess, m_guess, c_guess]
        
        # Dopasowanie
        popt, pcov = curve_fit(gaussian_linear, x, y, p0=p0)
        return popt
    except Exception as e:
        # print(f"Fit error: {e}")
        return None

def calculate_theoretical_2theta(a, wavelength, hkl=(1,1,1)):
    h, k, l = hkl
    d = a / np.sqrt(h**2 + k**2 + l**2)
    sin_theta = wavelength / (2 * d)
    if sin_theta > 1: return 0
    return 2 * np.degrees(np.arcsin(sin_theta))

def calculate_lattice_from_2theta(two_theta_deg, wavelength, hkl=(1,1,1)):
    theta_rad = np.radians(two_theta_deg / 2)
    if np.sin(theta_rad) == 0: return 0
    d = wavelength / (2 * np.sin(theta_rad))
    h, k, l = hkl
    return d * np.sqrt(h**2 + k**2 + l**2)

def calculate_uncertainty(two_theta_deg, delta_2theta_deg, wavelength, hkl=(1,1,1)):
    # delta_a = |da/dtheta| * (delta_2theta / 2)
    theta_rad = np.radians(two_theta_deg / 2)
    delta_theta_rad = np.radians(delta_2theta_deg / 2)
    
    h, k, l = hkl
    sqrt_hkl = np.sqrt(h**2 + k**2 + l**2)
    
    num = wavelength * sqrt_hkl * np.cos(theta_rad)
    den = 2 * (np.sin(theta_rad)**2)
    if den == 0: return 0
    
    return (np.abs(num / den)) * delta_theta_rad

# --- MAIN ---

def main():
    print("--- Rozpoczynam zaawansowaną analizę (Gauss + Tło) ---")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    if not os.path.exists(DATA_DIR):
        print(f"BŁĄD: Brak folderu {DATA_DIR}")
        return

    # --- ETAP 1: Kalibracja (wiązka) ---
    print("\n[1/4] Analiza wiązki (pliki 01, 02, 03)...")
    calib_files = ['01_2t0za.csv', '02_2t0ba.csv', '03_2t0ba.csv']
    calib_data = []

    for fname in calib_files:
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            meta, df = utils.parse_xrd_file(path)
            if not df.empty:
                calib_data.append((fname.replace('.csv',''), df))
                figures.plot_single_scan(df, f'Scan {fname}', os.path.join(RESULTS_DIR, f'plot_{fname[:-4]}.pdf'))

    if calib_data:
        figures.plot_combined_scans(calib_data, 'Porownanie skanow kalibracyjnych', 
                                    os.path.join(RESULTS_DIR, 'plot_calib_combined.pdf'))
        figures.plot_combined_shifted_scans(calib_data, 'Porownanie ksztaltu wiazki (zcentrowane)', 
                                            os.path.join(RESULTS_DIR, 'plot_calib_centered.pdf'))

    # --- ETAP 2: Wyznaczenie poprawki na próbce #1 ---
    print("\n[2/4] Kalibracja na podstawie Próbki #1 (GaAs)...")
    
    theo_2theta_ref = calculate_theoretical_2theta(CONFIG.REF_LATTICE_CONSTANT_A, CONFIG.WAVELENGTH_KALPHA1)
    print(f"  Teoretyczne 2Theta dla GaAs: {theo_2theta_ref:.4f} deg")
    
    file_ref = os.path.join(DATA_DIR, '05#1_2to.csv')
    correction_delta = 0.0
    uncertainty_correction = 0.0
    
    if os.path.exists(file_ref):
        _, df_ref = utils.parse_xrd_file(file_ref)
        popt_ref = fit_peak(df_ref)
        
        if popt_ref is not None:
            # popt = [amp, mean, sigma, m, c]
            meas_2theta_ref = popt_ref[1]
            meas_sigma_ref = popt_ref[2]
            
            correction_delta = meas_2theta_ref - theo_2theta_ref
            uncertainty_correction = 2.355 * meas_sigma_ref
            
            print(f"  Zmierzone 2Theta (ref): {meas_2theta_ref:.4f} deg")
            print(f"  -> POPRAWKA (Shift): {correction_delta:.4f} deg")
        else:
            print("  Błąd dopasowania piku referencyjnego.")
    else:
        print("  Brak pliku referencyjnego (05).")

    # --- ETAP 3: Analiza próbek ---
    print("\n[3/4] Analiza Próbek (z korekcją)...")
    
    samples_map = {
        '#1': {'2to': '05#1_2to.csv', 'oza': '04#1_oza.csv'},
        '#2': {'2to': '07#2_2to.csv', 'oza': '06#2_oza.csv'},
        '#3': {'2to': '09#3_2to.csv', 'oza': '08#3_oza.csv'}
    }
    
    results = []
    all_2to_scans = []

    for name, files in samples_map.items():
        # Omega
        f_oza = os.path.join(DATA_DIR, files['oza'])
        if os.path.exists(f_oza):
            _, df_o = utils.parse_xrd_file(f_oza)
            if not df_o.empty:
                figures.plot_single_scan(df_o, f'Skan Omega {name}', os.path.join(RESULTS_DIR, f'plot_{name}_omega.pdf'))

        # 2Theta/Omega
        f_2to = os.path.join(DATA_DIR, files['2to'])
        if not os.path.exists(f_2to): continue
        
        _, df_2t = utils.parse_xrd_file(f_2to)
        if df_2t.empty: continue
        
        all_2to_scans.append((f"Próbka {name}", df_2t))
        
        popt = fit_peak(df_2t)
        if popt is not None:
            # Rozpakowanie 5 parametrów
            amp, meas_2theta, sigma, m, c = popt
            
            fwhm = 2.355 * sigma
            corrected_2theta = meas_2theta - correction_delta
            
            # Niepewność całkowita (pomiar + poprawka)
            total_unc = np.sqrt(fwhm**2 + uncertainty_correction**2)
            
            a = calculate_lattice_from_2theta(corrected_2theta, CONFIG.WAVELENGTH_KALPHA1)
            da = calculate_uncertainty(corrected_2theta, total_unc, CONFIG.WAVELENGTH_KALPHA1)
            
            results.append({
                'Probka': name,
                'Raw_2Theta': round(meas_2theta, 4),
                'Corrected_2Theta': round(corrected_2theta, 4),
                'FWHM': round(fwhm, 4),
                'Lattice_a': round(a, 5),
                'Uncertainty_a': round(da, 5)
            })
            
            info = (f"Corr 2T = {corrected_2theta:.3f}\n"
                    f"a = {a:.4f} +/- {da:.4f} A")
            
            figures.plot_fit(df_2t, popt, gaussian_linear, f'Fit {name}', 
                             os.path.join(RESULTS_DIR, f'plot_{name}_fit.pdf'), info)

    # --- ETAP 4: Raport ---
    print("\n[4/4] Zapisywanie wyników...")
    
    if all_2to_scans:
        figures.plot_combined_scans(all_2to_scans, 'Zestawienie 2Theta/Omega', 
                                    os.path.join(RESULTS_DIR, 'plot_all_samples_2to.pdf'))

    if results:
        res_df = pd.DataFrame(results)
        print("\n--- WYNIKI KOŃCOWE ---")
        print(res_df[['Probka', 'Corrected_2Theta', 'Lattice_a', 'Uncertainty_a']])
        res_df.to_csv(os.path.join(RESULTS_DIR, 'final_results.csv'), index=False, sep=';')
        print(f"\nPlik CSV: {os.path.join(RESULTS_DIR, 'final_results.csv')}")
    else:
        print("Brak wyników do zapisania.")

if __name__ == "__main__":
    main()