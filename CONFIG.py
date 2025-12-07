# CONFIG.py

# --- PARAMETRY FIZYCZNE ---

# Długość fali promieniowania K-Alpha1 (Cu) [Angstrem]
WAVELENGTH_KALPHA1 = 1.54056

# --- MATERIAŁ REFERENCYJNY (Podłoże / Składnik A) ---
# Np. GaAs
REF_LATTICE_CONSTANT_A = 5.653
REF_HKL = (1, 1, 1)

# --- MATERIAŁ DOMIESZKI (Składnik B) ---
# Używany do prawa Vegarda: a_mix = (1-x)*a_A + x*a_B
# Np. dla InAs a = 6.0583
# Np. dla AlAs a = 5.6605
# Zmień tę wartość zgodnie z tym, co jest "drugim" składnikiem w Twoich próbkach!
LATTICE_CONSTANT_B = 6.0583 

# --- KONFIGURACJA WYGLĄDU WYKRESÓW ---
SHOW_TITLES = True
SHOW_LEGENDS = True
SHOW_PARAM_TEXT = True
CUSTOM_TITLES = {}