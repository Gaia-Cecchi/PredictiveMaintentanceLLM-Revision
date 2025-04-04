import pandas as pd
import re
import os

def fix_prediction_results(input_csv, output_csv=None):
    """
    Corregge le righe problematiche nel file CSV delle predizioni.
    
    Args:
        input_csv (str): Percorso al file CSV con righe problematiche
        output_csv (str): Percorso dove salvare il file corretto (se None, usa input_csv + "_fixed")
    
    Returns:
        tuple: (DataFrame corretto, numero di predizioni corrette, numero di predizioni totali)
    """
    if output_csv is None:
        base_name = os.path.splitext(input_csv)[0]
        output_csv = f"{base_name}.csv"
    
    print(f"Leggendo il file CSV: {input_csv}")
    
    # Leggi il file CSV
    df = pd.read_csv(input_csv)
    
    print(f"Analizzando {len(df)} righe...")
    
    # Conta righe problematiche prima della correzione
    problematic_rows = df[pd.isna(df['is_correct'])].shape[0]
    print(f"Trovate {problematic_rows} righe problematiche")
    
    # Analizza ogni riga per identificare e correggere i problemi
    fixed_rows = 0
    
    for i, row in df.iterrows():
        # Verifica se is_correct è NaN ma c'è un valore booleano in un'altra colonna
        if pd.isna(row['is_correct']):
            fixed_rows += 1
            
            # Cerca il valore 'True' o 'False' nelle ultime colonne
            last_cols = df.columns[-5:] # Controlla le ultime 5 colonne
            for col in last_cols:
                if isinstance(row[col], str) and (row[col].lower() == 'true' or row[col].lower() == 'false'):
                    # Trovato il valore is_correct in una colonna sbagliata
                    df.at[i, 'is_correct'] = row[col].lower() == 'true'
                    print(f"  Riga {i+2}: spostato '{row[col]}' nella colonna is_correct")
                    break
    
    # Verifica se ci sono ancora righe con is_correct mancante
    still_problematic = df[pd.isna(df['is_correct'])].shape[0]
    if still_problematic > 0:
        print(f"Attenzione: {still_problematic} righe hanno ancora is_correct mancante")
        
        # Ulteriore tentativo: analizza il campo key_indicators alla ricerca di True/False
        for i, row in df[pd.isna(df['is_correct'])].iterrows():
            for col in ['key_indicators', 'recommendation']:
                if isinstance(row[col], str):
                    # Cerca pattern come: "...valori", False
                    match = re.search(r'\",(True|False)$', str(row[col]))
                    if match:
                        df.at[i, 'is_correct'] = match.group(1).lower() == 'true'
                        print(f"  Riga {i+2}: estratto '{match.group(1)}' da '{col}' come is_correct")
                        fixed_rows += 1
    
    # Converti la colonna is_correct in booleano
    df['is_correct'] = df['is_correct'].astype(bool)
    
    # Calcola statistiche
    total_predictions = len(df)
    correct_predictions = df['is_correct'].sum()
    accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    
    # Salva il file corretto
    df.to_csv(output_csv, index=False)
    
    print(f"\nCorrezione completata: {fixed_rows} righe corrette su {problematic_rows} problematiche")
    print(f"File corretto salvato in: {output_csv}")
    print(f"\nStatistiche finali:")
    print(f"- Predizioni totali: {total_predictions}")
    print(f"- Predizioni corrette: {correct_predictions} ({accuracy:.2f}%)")
    print(f"- Predizioni errate: {total_predictions - correct_predictions} ({100-accuracy:.2f}%)")
    
    return df, correct_predictions, total_predictions

if __name__ == "__main__":
    # Chiedi all'utente il percorso del file
    input_file = input("Inserisci il percorso del file CSV da correggere: ")
    fix_prediction_results(input_file)