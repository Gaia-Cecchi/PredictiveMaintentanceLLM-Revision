import pandas as pd
from datetime import datetime
import re
import locale
from pathlib import Path

def convert_failures_to_csv():
    failures_data = {
        'Data': [
            '2024-05-15 10:00:00',
            '2024-05-22 14:00:00',
            '2024-05-25 16:00:00',
            '2024-06-01 12:00:00',
            '2024-06-08 10:00:00',
            '2024-06-15 14:00:00',
            '2024-06-22 16:00:00',
            '2024-06-25 10:00:00',
            '2024-06-28 14:00:00'
        ],
        'Tipo Guasto': [
            'Fusibile bruciato',
            'Corrosione interna',
            'Filtri aria ostruiti',
            'Ventilatore ostruito',
            'Perdita refrigerante',
            'Valvole difettose',
            'Controllore difettoso',
            'Cinghie allentate',
            'Livelli lubrificante bassi'
        ],
        'Categoria': [
            'Guasto Elettrico',
            'Accumulo di Ossidazione',
            'Mancanza di Manutenzione',
            'Surriscaldamento',
            'Livelli del Refrigerante',
            'Guasti Meccanici',
            'Sistema di Controllo',
            'Vibrazioni',
            'Lubrificazione'
        ],
        'Frequenza': [
            'Primo guasto',
            'Secondo guasto',
            'Terzo guasto',
            'Primo guasto',
            'Secondo guasto',
            'Primo guasto',
            'Primo guasto',
            'Secondo guasto',
            'Terzo guasto'
        ],
        'Causa': [
            'Sovraccarico elettrico dovuto a cortocircuito nel cablaggio',
            'Uso di refrigeranti e lubrificanti di bassa qualità',
            'Mancata pulizia regolare dei filtri',
            'Polvere accumulata sul ventilatore',
            'Guarnizione difettosa nel sistema di refrigerazione',
            'Usura delle valvole dovuta a mancanza di manutenzione',
            'Configurazione errata del controllore',
            'Mancata regolazione delle cinghie',
            'Mancata verifica dei livelli di lubrificante'
        ],
        'Soluzione': [
            'Sostituzione fusibile e verifica cablaggio',
            'Pulizia interna e sostituzione refrigeranti/lubrificanti',
            'Pulizia e sostituzione filtri aria',
            'Pulizia ventilatore e verifica ventilazione',
            'Sostituzione guarnizione e rabbocco refrigerante',
            'Sostituzione valvole e verifica condizioni',
            'Ri-configurazione controllore e verifica funzionalità',
            'Regolazione cinghie e verifica condizioni',
            'Rabbocco lubrificante e implementazione verifiche regolari'
        ],
        'Impatto': [
            'Interruzione funzionamento: 2 ore',
            'Riduzione efficienza: 20%',
            'Riduzione flusso aria: 30%',
            'Aumento temperatura: 5°C',
            'Riduzione efficienza raffreddamento: 25%',
            'Perdita pressione: 20%',
            'Interruzione funzionamento: 3 ore',
            'Aumento vibrazioni: 30%',
            'Riduzione efficienza: 15%'
        ],
        'ID Compressore': ['CSD102'] * 9
    }
    
    df = pd.DataFrame(failures_data)
    df.to_csv('guasti.csv', index=False, encoding='utf-8')

def parse_italian_date(date_string):
    # Mapping dei mesi italiani
    MONTHS = {
        'gennaio': '01', 'febbraio': '02', 'marzo': '03', 'aprile': '04',
        'maggio': '05', 'giugno': '06', 'luglio': '07', 'agosto': '08',
        'settembre': '09', 'ottobre': '10', 'novembre': '11', 'dicembre': '12'
    }
    
    # Formato atteso: "21 maggio 2024"
    parts = date_string.lower().split()
    if len(parts) == 3:
        day, month, year = parts
        if month in MONTHS:
            return f"{year}-{MONTHS[month]}-{day.zfill(2)}"
    return None

def convert_carico_operativo():
    data = []
    current_date = None
    
    with open('materiali/carico_e_condizioni_operative.txt', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Try to parse as Italian date
            parsed_date = parse_italian_date(line)
            if parsed_date:
                current_date = parsed_date
                continue
            
            if line.startswith('*') and current_date:
                time_match = re.search(r'(\d{2}:\d{2})-(\d{2}:\d{2})', line)
                if time_match:
                    start_time, end_time = time_match.groups()
                    
                    ore_match = re.search(r'Ore di Funzionamento: (\d+) ora', line)
                    carico_match = re.search(r'Carico Operativo: (\d+)%', line)
                    temp_match = re.search(r'Temperatura (\d+)°C', line)
                    umid_match = re.search(r'Umidità (\d+)%', line)
                    vibr_match = re.search(r'Vibrazioni ([\d.]+) mm/s', line)
                    press_match = re.search(r'Pressione ([\d.]+) bar', line)
                    
                    if all([ore_match, carico_match, temp_match, umid_match, vibr_match, press_match]):
                        data.append({
                            'Data': current_date,
                            'Ora_Inizio': start_time,
                            'Ora_Fine': end_time,
                            'Ore_Funzionamento': int(ore_match.group(1)),
                            'Carico_Operativo': int(carico_match.group(1)),
                            'Temperatura': int(temp_match.group(1)),
                            'Umidita': int(umid_match.group(1)),
                            'Vibrazioni': float(vibr_match.group(1)),
                            'Pressione': float(press_match.group(1))
                        })
    
    df = pd.DataFrame(data)
    df.to_csv('carico_operativo.csv', index=False)

def convert_feedback():
    """Convert feedback data from text to CSV, ensuring all entries are included"""
    data = []
    current_type = None
    
    script_path = Path(__file__).resolve()
    input_file = script_path.parent / "feedback_utenti.txt"
    output_file = script_path.parent / "feedback.csv"
    
    print(f"Reading feedback from: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        section_lines = []

        # First collect all content and determine where sections start
        positive_section_start = -1
        negative_section_start = -1
        
        for i, line in enumerate(lines):
            if 'Segnalazioni Positive' in line:
                positive_section_start = i
            elif 'Segnalazioni Negative' in line:
                negative_section_start = i
        
        # Process positive feedback section first
        current_type = 'Positivo'
        if positive_section_start >= 0:
            i = positive_section_start + 1  # Start from line after section header
            while i < len(lines) and (negative_section_start < 0 or i < negative_section_start):
                line = lines[i].strip()
                if line.startswith(str(len(data) + 1) + '.') or re.match(r'^\d+\.', line):
                    title = re.sub(r'^\d+\.\s*', '', line)
                    
                    # Look ahead for the rest of this entry's data
                    entry_data = {}
                    for j in range(i+1, min(i+10, len(lines))):
                        next_line = lines[j].strip()
                        
                        if 'Data:' in next_line:
                            data_match = re.search(r'Data:\s*(\d{2}/\d{2}/\d{4})', next_line)
                            if data_match:
                                entry_data['Data'] = data_match.group(1)
                        elif 'Ora:' in next_line:
                            ora_match = re.search(r'Ora:\s*(\d{2}:\d{2})', next_line)
                            if ora_match:
                                entry_data['Ora'] = ora_match.group(1)
                        elif 'Descrizione:' in next_line:
                            desc_match = re.search(r'Descrizione:\s*(.*)', next_line)
                            if desc_match:
                                entry_data['Descrizione'] = desc_match.group(1)
                        elif 'Valutazione:' in next_line:
                            val_match = re.search(r'Valutazione:\s*(\d)/5', next_line)
                            if val_match:
                                entry_data['Valutazione'] = int(val_match.group(1))
                        elif 'Commento:' in next_line:
                            comm_match = re.search(r'Commento:\s*"(.*)"', next_line)
                            if comm_match:
                                entry_data['Commento'] = comm_match.group(1)
                    
                    # If we found all required fields, add to data
                    required_fields = ['Data', 'Ora', 'Descrizione', 'Valutazione', 'Commento']
                    if all(field in entry_data for field in required_fields):
                        full_entry = {
                            'Tipo': current_type,
                            'Titolo': title,
                            **entry_data
                        }
                        data.append(full_entry)
                
                i += 1
        
        # Process negative feedback section
        current_type = 'Negativo'
        if negative_section_start >= 0:
            i = negative_section_start + 1  # Start from line after section header
            while i < len(lines):
                line = lines[i].strip()
                if line.startswith(str(len(data) + 1) + '.') or re.match(r'^\d+\.', line):
                    title = re.sub(r'^\d+\.\s*', '', line)
                    
                    # Look ahead for the rest of this entry's data
                    entry_data = {}
                    for j in range(i+1, min(i+15, len(lines))):  # Increased range to catch all fields
                        next_line = lines[j].strip()
                        
                        if 'Data:' in next_line:
                            data_match = re.search(r'Data:\s*(\d{2}/\d{2}/\d{4})', next_line)
                            if data_match:
                                entry_data['Data'] = data_match.group(1)
                        elif 'Ora:' in next_line:
                            ora_match = re.search(r'Ora:\s*(\d{2}:\d{2})', next_line)
                            if ora_match:
                                entry_data['Ora'] = ora_match.group(1)
                        elif 'Descrizione:' in next_line:
                            desc_match = re.search(r'Descrizione:\s*(.*)', next_line)
                            if desc_match:
                                entry_data['Descrizione'] = desc_match.group(1)
                        elif 'Valutazione:' in next_line:
                            val_match = re.search(r'Valutazione:\s*(\d)/5', next_line)
                            if val_match:
                                entry_data['Valutazione'] = int(val_match.group(1))
                        elif 'Commento:' in next_line:
                            comm_match = re.search(r'Commento:\s*"(.*)"', next_line)
                            if comm_match:
                                entry_data['Commento'] = comm_match.group(1)
                    
                    # If we found all required fields, add to data
                    required_fields = ['Data', 'Ora', 'Descrizione', 'Valutazione', 'Commento']
                    if all(field in entry_data for field in required_fields):
                        full_entry = {
                            'Tipo': current_type,
                            'Titolo': title,
                            **entry_data
                        }
                        data.append(full_entry)
                
                i += 1
    
    # Ensure all entries have consistent fields
    all_fields = ['Tipo', 'Titolo', 'Data', 'Ora', 'Descrizione', 'Valutazione', 'Commento']
    for entry in data:
        for field in all_fields:
            if field not in entry:
                entry[field] = None
    
    # Add compressor_id field to all entries
    for entry in data:
        entry['compressor_id'] = 'CSD102'
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add debug info
    print(f"Processed {len(df)} feedback entries")
    print("\nFeedback types found:")
    print(df['Tipo'].value_counts().to_dict())
    
    print("\nRating distribution:")
    print(df['Valutazione'].value_counts().sort_index().to_dict())
    
    # Special check for low ratings (≤ 3)
    low_ratings = df[df['Valutazione'] <= 3]
    print(f"\nFound {len(low_ratings)} entries with low ratings (≤ 3)")
    for _, row in low_ratings.iterrows():
        print(f"- {row['Data']} | {row['Titolo']} | Rating: {row['Valutazione']} | Type: {row['Tipo']}")
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"\nSaved feedback data to {output_file}")
    return df

def convert_manutenzioni():
    data = []
    current_intervention = None
    
    with open('materiali/manutenzioni.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
        for i in range(len(lines)):
            line = lines[i].strip()
            
            if line.startswith('Intervento'):
                intervention_match = re.search(r'Intervento (\d+) – (.+)', line)
                if intervention_match:
                    current_intervention = {
                        'Numero_Intervento': int(intervention_match.group(1)),
                        'Tipo_Intervento': intervention_match.group(2),
                        'Data': None,
                        'Ora': None,
                        'Ore_Funzionamento': None,
                        'Attivita': [],
                        'Anomalie': [],
                        'Raccomandazioni': []
                    }
                    
                    j = i + 1
                    current_section = None
                    
                    while j < len(lines) and not lines[j].strip().startswith('Intervento'):
                        next_line = lines[j].strip()
                        
                        if 'Data e Ora:' in next_line:
                            datetime_match = re.search(r'(\d+ \w+ \d+), (\d+:\d+)', next_line)
                            if datetime_match:
                                data_str = parse_italian_date(datetime_match.group(1))
                                if data_str:
                                    current_intervention['Data'] = data_str
                                    current_intervention['Ora'] = datetime_match.group(2)
                        elif 'Ore di funzionamento:' in next_line:
                            ore_match = re.search(r'(\d+) ore', next_line)
                            if ore_match:
                                current_intervention['Ore_Funzionamento'] = int(ore_match.group(1))
                        elif next_line == 'Attività eseguite:':
                            current_section = 'attivita'
                        elif next_line == 'Anomalie riscontrate:':
                            current_section = 'anomalie'
                        elif next_line == 'Raccomandazioni future:':
                            current_section = 'raccomandazioni'
                        elif next_line.startswith('*') and current_section:
                            item = next_line.replace('* ', '').strip()
                            if current_section == 'attivita':
                                current_intervention['Attivita'].append(item)
                            elif current_section == 'anomalie':
                                current_intervention['Anomalie'].append(item)
                            elif current_section == 'raccomandazioni':
                                current_intervention['Raccomandazioni'].append(item)
                        
                        j += 1
                    
                    if current_intervention['Data']:
                        data.append(current_intervention)
    
    for item in data:
        item['Attivita'] = '|'.join(item['Attivita'])
        item['Anomalie'] = '|'.join(item['Anomalie'])
        item['Raccomandazioni'] = '|'.join(item['Raccomandazioni'])
    
    df = pd.DataFrame(data)
    df.to_csv('manutenzioni.csv', index=False)

if __name__ == "__main__":
    convert_failures_to_csv()
    convert_carico_operativo()
    convert_feedback()
    convert_manutenzioni()