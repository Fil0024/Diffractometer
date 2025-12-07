# utils.py
import pandas as pd
import io
import os

def parse_xrd_file(file_path):
    """
    Wczytuje plik CSV z dyfraktometru, parsuje nagłówek i dane.
    Zwraca słownik metadanych i DataFrame z danymi.
    """
    metadata = {}
    data_lines = []
    reading_data = False
    
    if not os.path.exists(file_path):
        print(f"BŁĄD: Nie znaleziono pliku {file_path}")
        return {}, pd.DataFrame()

    # Próba otwarcia z różnymi kodowaniami (pliki mogą mieć polskie znaki)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='latin-1') as f:
            lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Wykrycie początku sekcji danych
        if line == '[Scan points]':
            reading_data = True
            continue
        
        # Parsowanie nagłówka (Metadane)
        if not reading_data:
            if ';' in line:
                parts = line.split(';')
                key = parts[0].strip()
                if len(parts) > 1:
                    value = parts[1].strip()
                    metadata[key] = value
        else:
            data_lines.append(line)
    
    # Tworzenie DataFrame z danych
    if not data_lines:
        return metadata, pd.DataFrame()

    # Pierwszy wiersz to zazwyczaj 'Angle;Intensity' - pomijamy go w liście danych
    # Zamieniamy przecinki na kropki dla liczb zmiennoprzecinkowych
    data_content = [line.replace(',', '.') for line in data_lines[1:]]
    
    data_str = '\n'.join(data_content)
    try:
        # Wczytujemy dane, separator to średnik
        df = pd.read_csv(io.StringIO(data_str), sep=';', names=['Angle', 'Intensity'], header=None)
    except Exception as e:
        print(f"Błąd parsowania danych w pliku {file_path}: {e}")
        return metadata, pd.DataFrame()
    
    # Obliczenie Intensity CPS (Counts Per Second)
    # Pobranie czasu na krok z metadanych, domyślnie 1s
    time_str = metadata.get('Time per step', '1').replace(',', '.')
    try:
        time_per_step = float(time_str)
    except ValueError:
        time_per_step = 1.0
    
    df['Intensity_cps'] = df['Intensity'] / time_per_step
    
    return metadata, df
