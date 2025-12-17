# CONFIG.py

# --- PARAMETRY FIZYCZNE ---
WAVELENGTH_KALPHA1 = 1.54056

# --- MATERIAŁ REFERENCYJNY (Podłoże / Składnik A) ---
REF_LATTICE_CONSTANT_A = 5.653
REF_HKL = (1, 1, 1)

# --- MATERIAŁ DOMIESZKI (Składnik B) ---
LATTICE_CONSTANT_B = 6.0583 

# --- KONFIGURACJA WYGLĄDU WYKRESÓW ---
SHOW_TITLES = False
SHOW_LEGENDS = False
SHOW_PARAM_TEXT = False

# Mnożnik FWHM dla zakresu osi X na wykresach dopasowania.
# Np. 5.0 oznacza zakres: środek +/- 5 * FWHM.
# Ustaw None lub 0, aby zostawić domyślny, pełny zakres.
PLOT_ZOOM_FWHM_MULTIPLIER = 4

CUSTOM_TITLES = {}