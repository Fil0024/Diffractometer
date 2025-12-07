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
    """Model: Gauss + Funkcja liniowa (tło)."""
    return amp * np.exp(-(x - mean)**2 / (2 * sigma**2)) + m * x + c

def fit_peak(df):
    """Dopasowuje funkcję Gaussa z tłem liniowym."""
    if df.empty or 'Angle' not in df or 'Intensity_cps' not in df: return None
    x = df['Angle'].values
    y = df['Intensity_cps'].values
    if len(x) < 5: return None

    try:
        # Estymacja tła
        n_bg = max(1, int(len(x) * 0.1))
        x_bg = np.concatenate([x[:n_bg], x[-n_bg:]])
        y_bg = np.concatenate([y[:n_bg], y[-n_bg:]])
        
        if len(x_bg) > 1:
            p_bg = np.polyfit(x_bg, y_bg, 1)
            m_guess, c_guess = p_bg[0], p_bg[1]
        else:
            m_guess, c_guess = 0, min(y)

        y_pure = y - (m_guess * x + c_guess)
        mean_guess = x[np.argmax(y_pure)]
        amp_guess = np.max(y_pure)
        if amp_guess <= 0: amp_guess = np.max(y)
        
        total_intensity = np.sum(y_pure[y_pure > 0])
        if total_intensity > 0:
            sigma_guess = np.sqrt(np.abs(np.sum((x - mean_guess)**2 * y_pure) / total_intensity))
        else:
            sigma_guess = 0.05
        if sigma_guess == 0 or np.isnan(sigma_guess): sigma_guess = 0.05

        p0 = [amp_guess, mean_guess, sigma_guess, m_guess, c_guess]
        popt, pcov = curve_fit(gaussian_linear, x, y, p0=p0)
        return popt, pcov
    except:
        return None

def calculate_theoretical_2theta(a, wavelength, hkl=(1,1,1)):
    h, k, l = hkl
    d = a / np.sqrt(h**2 + k**2 + l**2)
    sin_theta = wavelength / (2 * d)
    if abs(sin_theta) > 1: return 0
    return 2 * np.degrees(np.arcsin(sin_theta))

def calculate_lattice_from_2theta(two_theta_deg, wavelength, hkl=(1,1,1)):
    theta_rad = np.radians(two_theta_deg / 2)
    if np.sin(theta_rad) == 0: return 0
    d = wavelength / (2 * np.sin(theta_rad))
    h, k, l = hkl
    return d * np.sqrt(h**2 + k**2 + l**2)

def calculate_uncertainty(two_theta_deg, delta_2theta_deg, wavelength, hkl=(1,1,1)):
    theta_rad = np.radians(two_theta_deg / 2)
    delta_theta_rad = np.radians(delta_2theta_deg / 2)
    h, k, l = hkl
    sqrt_hkl = np.sqrt(h**2 + k**2 + l**2)
    num = wavelength * sqrt_hkl * np.cos(theta_rad)
    den = 2 * (np.sin(theta_rad)**2)
    if den == 0: return 0
    return (np.abs(num / den)) * delta_theta_rad

def calculate_vegard_x(a_measured, da_measured, a_A, a_B):
    denominator = a_B - a_A
    if denominator == 0: return 0, 0
    x = (a_measured - a_A) / denominator
    dx = da_measured / np.abs(denominator)
    return x, dx

def plot_vegard_line(results, filename):
    plt.figure(figsize=(8, 6))
    x_vals = [r['Vegard_x'] for r in results]
    max_x = max(x_vals) if x_vals else 1.0
    if max_x == 0: max_x = 0.1
    x_range = np.linspace(0, max_x * 1.2, 100)
    
    a_A = CONFIG.REF_LATTICE_CONSTANT_A
    a_B = CONFIG.LATTICE_CONSTANT_B
    y_range = (1 - x_range) * a_A + x_range * a_B
    
    plt.plot(x_range, y_range, 'k--', label='Prawo Vegarda (teoria)')
    
    for res in results:
        plt.errorbar(res['Vegard_x'], res['Lattice_a'], 
                     yerr=res['Uncertainty_a'], xerr=res['Uncertainty_x'], 
                     fmt='o', label=f"{res['Probka']}", capsize=5)

    plt.xlabel('Zawartość składnika x')
    plt.ylabel('Stała sieci a [A]')
    plt.title('Weryfikacja Prawa Vegarda')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Zapisano wykres Vegarda: {filename}")


# --- MAIN ---

def main():
    print("--- Rozpoczynam analizę (Gauss+Tło, Poprawka, Vegard, FWHM) ---")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    if not os.path.exists(DATA_DIR):
        print(f"BŁĄD: Brak folderu {DATA_DIR}")
        return

    # 1. Kalibracja (wiązka)
    print("\n[1/4] Analiza wiązki...")
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

    # 2. Kalibracja na Próbce #1
    print("\n[2/4] Kalibracja na podstawie Próbki #1 (GaAs)...")
    theo_2theta_ref = calculate_theoretical_2theta(CONFIG.REF_LATTICE_CONSTANT_A, CONFIG.WAVELENGTH_KALPHA1)
    
    file_ref = os.path.join(DATA_DIR, '05#1_2to.csv')
    correction_delta = 0.0
    uncertainty_correction = 0.0
    
    if os.path.exists(file_ref):
        _, df_ref = utils.parse_xrd_file(file_ref)
        fit_res = fit_peak(df_ref)
        if fit_res is not None:
            popt_ref, pcov_ref = fit_res
            meas_2theta_ref = popt_ref[1]
            meas_sigma_ref = popt_ref[2]
            correction_delta = meas_2theta_ref - theo_2theta_ref
            uncertainty_correction = 2.355 * meas_sigma_ref
            print(f"  Poprawka instrumentalna: {correction_delta:.4f} deg")
        else:
            print("  Błąd dopasowania piku referencyjnego.")
    else:
        print("  Brak pliku referencyjnego.")

    # 3. Analiza Próbek
    print("\n[3/4] Analiza Próbek i Prawo Vegarda...")
    samples_map = {
        '#1': {'2to': '05#1_2to.csv', 'oza': '04#1_oza.csv'},
        '#2': {'2to': '07#2_2to.csv', 'oza': '06#2_oza.csv'},
        '#3': {'2to': '09#3_2to.csv', 'oza': '08#3_oza.csv'}
    }
    
    results = []
    all_2to_scans = []

    for name, files in samples_map.items():
        f_oza = os.path.join(DATA_DIR, files['oza'])
        if os.path.exists(f_oza):
            _, df_o = utils.parse_xrd_file(f_oza)
            if not df_o.empty:
                figures.plot_single_scan(df_o, f'Omega {name}', os.path.join(RESULTS_DIR, f'plot_{name}_omega.pdf'))

        f_2to = os.path.join(DATA_DIR, files['2to'])
        if not os.path.exists(f_2to): continue
        _, df_2t = utils.parse_xrd_file(f_2to)
        if df_2t.empty: continue
        all_2to_scans.append((f"Próbka {name}", df_2t))
        
        fit_res = fit_peak(df_2t)
        if fit_res is not None:
            popt, pcov = fit_res
            amp, meas_2theta, sigma, m, c = popt
            
            fwhm = 2.355 * sigma
            sigma_err = np.sqrt(pcov[2, 2])
            fwhm_err = 2.355 * sigma_err
            
            corrected_2theta = meas_2theta - correction_delta
            total_unc_pos = np.sqrt(fwhm**2 + uncertainty_correction**2)
            
            a = calculate_lattice_from_2theta(corrected_2theta, CONFIG.WAVELENGTH_KALPHA1)
            da = calculate_uncertainty(corrected_2theta, total_unc_pos, CONFIG.WAVELENGTH_KALPHA1)
            x_vegard, dx_vegard = calculate_vegard_x(a, da, CONFIG.REF_LATTICE_CONSTANT_A, CONFIG.LATTICE_CONSTANT_B)
            
            results.append({
                'Probka': name,
                'Raw_2Theta': round(meas_2theta, 4),
                'Corrected_2Theta': round(corrected_2theta, 4),
                'FWHM': round(fwhm, 4),
                'Uncertainty_FWHM': round(fwhm_err, 5),
                'Lattice_a': round(a, 5),
                'Uncertainty_a': round(da, 5),
                'Vegard_x': round(x_vegard, 4),
                'Uncertainty_x': round(dx_vegard, 4),
                'Background_Slope': round(m, 4),
                'Background_Intercept': round(c, 4)
            })
            
            info = (f"Corr 2T = {corrected_2theta:.3f}\n"
                    f"FWHM = {fwhm:.3f} +/- {fwhm_err:.3f}\n"
                    f"a = {a:.4f}\n"
                    f"x = {x_vegard:.3f}")
            
            # --- ZMIANA: Generowanie dwóch wykresów (LIN i LOG) ---
            
            # 1. Skala Liniowa (suffix _lin)
            figures.plot_fit(df_2t, popt, gaussian_linear, f'Fit {name} (Lin)', 
                             os.path.join(RESULTS_DIR, f'plot_{name}_fit.pdf'), 
                             info, log_scale=False)

            # 2. Skala Logarytmiczna (suffix _log)
            figures.plot_fit(df_2t, popt, gaussian_linear, f'Fit {name} (Log)', 
                             os.path.join(RESULTS_DIR, f'plot_{name}_fit_log.pdf'), 
                             info, log_scale=True)

    # 4. Zapis
    print("\n[4/4] Generowanie raportu CSV...")
    if all_2to_scans:
        figures.plot_combined_scans(all_2to_scans, 'Zestawienie 2Theta/Omega', 
                                    os.path.join(RESULTS_DIR, 'plot_all_samples_2to.pdf'))

    if results:
        plot_vegard_line(results, os.path.join(RESULTS_DIR, 'plot_vegard_law.pdf'))
        
        # Zapis CSV (wszystkie kolumny, w tym tło)
        res_df = pd.DataFrame(results)
        csv_path = os.path.join(RESULTS_DIR, 'final_results.csv')
        res_df.to_csv(csv_path, index=False, sep=';')
        print(f"Plik CSV: {csv_path}")

        # Podgląd w konsoli
        cols_to_show = ['Probka', 'Corrected_2Theta', 'FWHM', 'Lattice_a', 'Vegard_x', 'Background_Slope']
        print("\n--- PODGLĄD WYNIKÓW ---")
        print(res_df[cols_to_show])

if __name__ == "__main__":
    main()