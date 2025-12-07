# main.py
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os
import sys

# Importy z Twoich plików
try:
    import CONFIG
    import utils
    import figures
except ImportError as e:
    print("BŁĄD: Brakuje jednego z plików pomocniczych (CONFIG.py, utils.py, figures.py).")
    print(f"Szczegóły: {e}")
    sys.exit(1)

# --- KONFIGURACJA ---
DATA_DIR = 'data'       # Folder z danymi wejściowymi
RESULTS_DIR = 'results' # Folder na wyniki (PDF i CSV)

# --- FUNKCJE OBLICZENIOWE ---

def gaussian(x, amp, mean, sigma):
    """Funkcja Gaussa do dopasowania."""
    return amp * np.exp(-(x - mean)**2 / (2 * sigma**2))

def fit_peak(df):
    """Dopasowuje funkcję Gaussa do danych."""
    if df.empty or 'Angle' not in df or 'Intensity_cps' not in df:
        return None

    x = df['Angle'].values
    y = df['Intensity_cps'].values

    if len(x) == 0 or len(y) == 0:
        return None

    try:
        sum_y = sum(y)
        if sum_y == 0:
            return None

        mean_guess = sum(x * y) / sum_y
        sigma_guess = np.sqrt(sum(y * (x - mean_guess)**2) / sum_y)
        amp_guess = max(y)

        if sigma_guess == 0:
            sigma_guess = 0.1

        popt, pcov = curve_fit(gaussian, x, y, p0=[amp_guess, mean_guess, sigma_guess])
        return popt
    except Exception as e:
        return None

def calculate_lattice_constant(theta_deg, wavelength):
    """Oblicza stałą sieci dla struktury regularnej i refleksu (111)."""
    theta_rad = np.radians(theta_deg)
    if np.sin(theta_rad) == 0:
        return 0
    d = wavelength / (2 * np.sin(theta_rad))
    a = d * np.sqrt(3)
    return a

def calculate_uncertainty(theta_deg, delta_theta_deg, wavelength):
    """Propagacja niepewności dla stałej sieci."""
    theta_rad = np.radians(theta_deg)
    delta_theta_rad = np.radians(delta_theta_deg)

    numerator = wavelength * np.sqrt(3) * np.cos(theta_rad)
    denominator = 2 * (np.sin(theta_rad)**2)
    if denominator == 0:
        return 0
    derivative = numerator / denominator

    return np.abs(derivative) * delta_theta_rad

# --- GŁÓWNA PĘTLA PROGRAMU ---

def main():
    print("--- Rozpoczynam analizę danych ---")
    current_dir = os.getcwd()
    print(f"Katalog roboczy: {current_dir}")
    print(f"Dane pobierane z: {os.path.join(DATA_DIR)}")
    print(f"Wyniki trafią do: {os.path.join(RESULTS_DIR)}\n")

    # Tworzenie folderu results, jeśli nie istnieje
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.exists(DATA_DIR):
        print(f"BŁĄD: Nie znaleziono folderu z danymi '{DATA_DIR}'!")
        return

    # 1. Analiza plików 01, 02, 03 (Kalibracja / Wiązka)
    calibration_files = ['01_2t0za.csv', '02_2t0ba.csv', '03_2t0ba.csv']
    calibration_data = []

    print("--- Przetwarzanie plików kalibracyjnych ---")
    for fname in calibration_files:
        full_path = os.path.join(DATA_DIR, fname)

        if not os.path.exists(full_path):
            print(f"  [OSTRZEŻENIE] Brak pliku: {full_path}")
            continue

        meta, df = utils.parse_xrd_file(full_path)
        if not df.empty:
            base_name = os.path.basename(fname).replace(".csv", "")
            # Zmiana na PDF i ścieżkę results/
            plot_filename = os.path.join(RESULTS_DIR, f'plot_{base_name}.pdf')

            figures.plot_single_scan(df, f'Scan {base_name}', plot_filename)
            calibration_data.append((base_name, df))
        else:
            print(f"  [OSTRZEŻENIE] Plik {fname} jest pusty lub niepoprawny.")

    if calibration_data:
        combined_filename = os.path.join(RESULTS_DIR, 'plot_combined_01_02_03.pdf')
        figures.plot_combined_scans(calibration_data, 'Porównanie skanów 01, 02, 03', combined_filename)

    # 2. Analiza próbek #1, #2, #3
    samples = {
        '#1': {'2to': '05#1_2to.csv', 'oza': '04#1_oza.csv'},
        '#2': {'2to': '07#2_2to.csv', 'oza': '06#2_oza.csv'},
        '#3': {'2to': '09#3_2to.csv', 'oza': '08#3_oza.csv'}
    }

    results = []

    print("\n--- Analiza Próbek ---")
    for sample_name, files in samples.items():
        print(f"Analizuję próbkę {sample_name}...")

        path_2to = os.path.join(DATA_DIR, files['2to'])
        path_oza = os.path.join(DATA_DIR, files['oza'])

        # A. Skan Omega (OZA)
        if os.path.exists(path_oza):
            meta_oza, df_oza = utils.parse_xrd_file(path_oza)
            if not df_oza.empty:
                plot_name = os.path.join(RESULTS_DIR, f'plot_{sample_name}_omega.pdf')
                figures.plot_single_scan(df_oza, f'Próbka {sample_name} - Skan Omega', plot_name)
        else:
             print(f"  Brak pliku OZA: {path_oza}")

        # B. Skan 2Theta/Omega (2TO)
        if not os.path.exists(path_2to):
            print(f"  Brak pliku 2TO: {path_2to} - pomijam próbkę.")
            continue

        meta_2to, df_2to = utils.parse_xrd_file(path_2to)

        if df_2to.empty:
            print(f"  Nie udało się wczytać danych z pliku {path_2to}")
            continue

        # Dopasowanie
        popt = fit_peak(df_2to)

        if popt is not None:
            amp, mean_2theta, sigma = popt

            fwhm_2theta = 2.355 * sigma
            theta_b = mean_2theta / 2
            uncertainty_2theta = fwhm_2theta
            uncertainty_theta = uncertainty_2theta / 2

            a = calculate_lattice_constant(theta_b, CONFIG.WAVELENGTH_KALPHA1)
            da = calculate_uncertainty(theta_b, uncertainty_theta, CONFIG.WAVELENGTH_KALPHA1)

            res = {
                'Próbka': sample_name,
                'Plik': files['2to'],
                '2Theta_Mean': round(mean_2theta, 4),
                'Theta_B': round(theta_b, 4),
                'FWHM_2Theta': round(fwhm_2theta, 4),
                'Stala_sieci_a': round(a, 5),
                'Niepewnosc_a': round(da, 5)
            }
            results.append(res)

            info_text = (f"a = {a:.4f} +/- {da:.4f} A\n"
                            f"2Theta = {mean_2theta:.3f} deg\n"
                            f"FWHM = {fwhm_2theta:.3f} deg")

            fit_plot_name = os.path.join(RESULTS_DIR, f'plot_{sample_name}_fit_2to.pdf')
            figures.plot_fit(df_2to, popt, gaussian,
                                f'Próbka {sample_name} - Dopasowanie 2Theta/Omega',
                                fit_plot_name, info_text)
            print(f"  -> Sukces: a={a:.4f}")
        else:
            print(f"  -> Ostrzeżenie: Nie udało się dopasować krzywej Gaussa dla {sample_name}")

    # 3. Zapis wyników
    print("\n--- Podsumowanie ---")
    if not results:
        print("BŁĄD: Nie wygenerowano żadnych wyników.")
    else:
        results_df = pd.DataFrame(results)
        cols_to_show = ['Próbka', 'Stala_sieci_a', 'Niepewnosc_a', '2Theta_Mean']
        print(results_df[cols_to_show])

        # Zapis CSV również do folderu results
        output_file = os.path.join(RESULTS_DIR, 'wyniki_analizy_stałej_sieci.csv')
        results_df.to_csv(output_file, index=False, sep=';')
        print(f"\nPełne wyniki zapisano do pliku: {output_file}")

if __name__ == "__main__":
    main()
