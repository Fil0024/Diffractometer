# CONFIG.py

# --- PARAMETRY FIZYCZNE ---
WAVELENGTH_KALPHA1 = 1.54056

# --- MATERIAŁ REFERENCYJNY (Podłoże / Składnik A) ---
REF_LATTICE_CONSTANT_A = 5.653
REF_HKL = (1, 1, 1)

# --- MATERIAŁ DOMIESZKI (Składnik B) ---
LATTICE_CONSTANT_B = 6.0583 

# --- KONFIGURACJA WYGLĄDU WYKRESÓW ---
SHOW_TITLES = True
SHOW_LEGENDS = True
SHOW_PARAM_TEXT = True

# NOWE OPCJE:
# Ustawienie True włączy skalę logarytmiczną na osi Y dla wykresów dopasowania
PLOT_LOG_SCALE = True

# Mnożnik FWHM dla zakresu osi X na wykresach dopasowania.
# Np. 5.0 oznacza zakres: środek +/- 5 * FWHM.
# Ustaw None lub 0, aby zostawić domyślny, pełny zakres.
PLOT_ZOOM_FWHM_MULTIPLIER = 5.0

CUSTOM_TITLES = {}