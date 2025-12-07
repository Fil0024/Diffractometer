# CONFIG.py

# --- PARAMETRY FIZYCZNE ---

# Długość fali promieniowania K-Alpha1 (Cu) [Angstrem]
WAVELENGTH_KALPHA1 = 1.54056

# Parametry odniesienia dla próbki #1 (GaAs)
# Używane do obliczenia poprawki instrumentalnej (przesunięcia zera)
REF_LATTICE_CONSTANT_A = 5.653  # [Angstrem]
REF_HKL = (1, 1, 1)             # Indeksy Millera refleksu

# --- KONFIGURACJA WYGLĄDU WYKRESÓW ---

# Czy pokazywać tytuły na wykresach? (True/False)
SHOW_TITLES = True

# Czy pokazywać legendę na wykresach? (True/False)
SHOW_LEGENDS = True

# Czy pokazywać tekst z wynikami (parametry dopasowania) na wykresach?
SHOW_PARAM_TEXT = True

# Opcjonalnie: Nadpisanie tytułów. 
# Jeśli klucz (nazwa pliku bez rozszerzenia) istnieje w słowniku, zostanie użyty podany tytuł.
CUSTOM_TITLES = {
    # 'plot_01_2t0za': 'Skan kalibracyjny 1',
    # 'plot_all_samples_2to': 'Moje porownanie',
}