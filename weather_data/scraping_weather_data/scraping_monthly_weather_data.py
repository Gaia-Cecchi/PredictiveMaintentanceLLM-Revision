import os
import time
import sys
import subprocess
import logging

# Attempt to import required modules, install if not available
try:
    import pandas as pd
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.common.exceptions import NoSuchElementException, TimeoutException
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from webdriver_manager.chrome import ChromeDriverManager
    from datetime import datetime, timedelta
except ImportError as e:
    print(f"Missing dependency: {e.name}. Attempting to install...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", e.name])
        print(f"Successfully installed {e.name}, restarting script...")
        # Restart the script
        os.execv(sys.executable, ['python'] + sys.argv)
    except Exception as install_error:
        print(f"Failed to install {e.name}: {install_error}")
        print("Please manually install requirements by running: pip install -r requirements.txt")
        sys.exit(1)

# Get the current working directory and project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

# Set up logging
log_file = os.path.join(current_dir, "weather_scraping.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# Creazione directory di output se non esiste
output_dir = os.path.join(current_dir, "output")
os.makedirs(output_dir, exist_ok=True)

# Configurazione del webdriver
def setup_driver():
    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-web-security")
        chrome_options.add_argument("--allow-running-insecure-content")
        chrome_options.add_argument("--ignore-certificate-errors")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        # Disable logging
        chrome_options.add_argument("--log-level=3")
        chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])
        
        # Use the local chromedriver.exe
        chromedriver_path = os.path.join(current_dir, "chromedriver.exe")
        logger.info(f"Using local ChromeDriver at: {chromedriver_path}")
        
        if not os.path.exists(chromedriver_path):
            logger.error(f"ChromeDriver not found at {chromedriver_path}")
            raise FileNotFoundError(f"ChromeDriver executable not found at {chromedriver_path}")
        
        service = Service(executable_path=chromedriver_path)
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        logger.info("WebDriver initialized successfully")
        return driver
    except Exception as e:
        logger.error(f"Failed to initialize WebDriver: {str(e)}")
        raise

# Add a wrapper function that handles driver crashes
def with_driver_recovery(max_retries=3):
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Driver error: {str(e)}")
                    retries += 1
                    if retries < max_retries:
                        logger.info(f"Restarting driver (attempt {retries+1}/{max_retries})...")
                        # Close any existing drivers
                        try:
                            if 'driver' in args and args[0] is not None:
                                args[0].quit()
                        except:
                            pass
                        
                        # Create a new driver
                        new_driver = setup_driver()
                        
                        # Replace the driver in args
                        args = list(args)
                        args[0] = new_driver
                        args = tuple(args)
                        
                        time.sleep(2)  # Brief pause before retry
                    else:
                        logger.error(f"Max retries ({max_retries}) exceeded. Giving up.")
                        raise
            return None
        return wrapper
    return decorator

@with_driver_recovery(max_retries=3)
def extract_weather_data(driver, url, max_retries=3):
    retries = 0
    while retries < max_retries:
        try:
            driver.get(url)
            
            # Attendi che la tabella dei dati meteo sia caricata
            table_selector = "#main_content > div.main_content__col1 > table:nth-child(6)"
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, table_selector))
            )
            
            # Estrai i dati usando i selettori CSS forniti
            data = {}
            
            # Trova tutte le righe della tabella dei dati meteo
            rows = driver.find_elements(By.CSS_SELECTOR, f"{table_selector} > tbody > tr")
            
            if not rows:
                logger.warning(f"Nessuna riga trovata nella tabella per {url}")
                return {}
                
            for i in range(1, len(rows)):
                try:
                    # Per ogni riga, estrai il nome del valore e il valore stesso
                    name = rows[i].find_element(By.CSS_SELECTOR, "td:nth-child(1)").text.strip()
                    value = rows[i].find_element(By.CSS_SELECTOR, "td:nth-child(2)").text.strip()
                    if name and value:  # Solo se sia nome che valore sono non vuoti
                        data[name] = value
                except NoSuchElementException:
                    continue
                except Exception as e:
                    logger.warning(f"Errore nell'estrazione di una riga: {e}")
            
            return data
        
        except TimeoutException:
            logger.warning(f"Timeout nell'attesa della tabella dati per {url}")
            retries += 1
            time.sleep(2)  # Attendi un po' prima di riprovare
            
        except Exception as e:
            logger.error(f"Errore nell'estrazione dei dati da {url}: {e}")
            retries += 1
            time.sleep(2)
    
    logger.error(f"Fallito il recupero dei dati dopo {max_retries} tentativi per {url}")
    return {}

# Definizione dei mesi in italiano con il loro numero di giorni nel 2024 (anno bisestile)
months = [
    {"name": "Gennaio", "days": 31},
    {"name": "Febbraio", "days": 29},  # 2024 è bisestile
    {"name": "Marzo", "days": 31},
    {"name": "Aprile", "days": 30},
    {"name": "Maggio", "days": 31},
    {"name": "Giugno", "days": 30},
    {"name": "Luglio", "days": 31},
    {"name": "Agosto", "days": 31},
    {"name": "Settembre", "days": 30},
    {"name": "Ottobre", "days": 31},
    {"name": "Novembre", "days": 30},
    {"name": "Dicembre", "days": 31}
]

def main():
    driver = None
    try:
        driver = setup_driver()
        
        # Track progress for recovery
        progress_file = os.path.join(current_dir, "scraping_progress.txt")
        last_month = 1
        last_day = 1
        
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r') as f:
                    progress = f.read().strip().split(',')
                    if len(progress) == 2:
                        last_month = int(progress[0])
                        last_day = int(progress[1])
                        logger.info(f"Resuming scraping from month {last_month}, day {last_day}")
            except:
                logger.warning("Could not read progress file. Starting from beginning.")
        
        # Ciclo attraverso tutti i mesi e giorni per il 2024
        for month_idx, month in enumerate(months, 1):
            # Skip months we've already processed
            if month_idx < last_month:
                logger.info(f"Skipping month {month_idx} ({month['name']}), already processed")
                continue
                
            month_name = month["name"]
            days = month["days"]
            
            logger.info(f"Elaborazione del mese di {month_name}...")
            
            # Verifica se esiste già un file per questo mese
            output_file = os.path.join(output_dir, f"weather_data_{month_name.lower()}_2024.csv")
            existing_data = []
            existing_dates = set()
            
            # Se il file esiste, carica i dati già scaricati
            if os.path.exists(output_file):
                try:
                    existing_df = pd.read_csv(output_file)
                    existing_data = existing_df.to_dict('records')
                    existing_dates = set(existing_df['Data'].tolist())
                    logger.info(f"Caricati {len(existing_data)} giorni già scaricati per {month_name}")
                except Exception as e:
                    logger.warning(f"Errore nel caricamento del file esistente {output_file}: {e}")
            
            # Inizializza una lista per memorizzare i dati di tutti i giorni del mese
            month_data = existing_data.copy()
            
            # Ciclo attraverso tutti i giorni del mese
            for day in range(1, days + 1):
                # Skip days we've already processed within the current month
                if month_idx == last_month and day < last_day:
                    logger.info(f"Skipping day {day}, already processed")
                    continue
                    
                date_str = f"2024-{month_idx:02d}-{day:02d}"
                
                # Save our current progress
                with open(progress_file, 'w') as f:
                    f.write(f"{month_idx},{day}")
                
                # Salta i giorni già scaricati
                if date_str in existing_dates:
                    logger.info(f"Saltato {date_str} - già scaricato")
                    continue
                
                # Verifica che non si cerchi di accedere a una data futura
                current_date = datetime.now()
                target_date = datetime(2024, month_idx, day)
                
                if target_date > current_date:
                    logger.info(f"Saltata la data futura: {day} {month_name} 2024")
                    continue
                
                # Costruisci l'URL per il giorno corrente
                url = f"https://www.ilmeteo.it/portale/archivio-meteo/Santa+Croce+sull%27Arno/2024/{month_name}/{day}"
                logger.info(f"Scraping dati per il {day} {month_name} 2024: {url}")
                
                # Estrai i dati
                day_data = extract_weather_data(driver, url)
                
                # Aggiungi la data ai dati estratti
                day_data["Data"] = date_str
                day_data["Giorno"] = day
                day_data["Mese"] = month_name
                
                # Aggiungi i dati del giorno alla lista del mese
                month_data.append(day_data)
                
                # Salva i dati dopo ogni giorno per evitare perdite in caso di errori
                temp_df = pd.DataFrame(month_data)
                if not temp_df.empty:
                    # Assicurati che la colonna Data sia la prima
                    cols = ["Data", "Giorno", "Mese"] + [col for col in temp_df.columns if col not in ["Data", "Giorno", "Mese"]]
                    temp_df = temp_df[cols]
                    temp_df.to_csv(output_file, index=False)
                
                # Pausa per non sovraccaricare il server
                time.sleep(2)
            
            # Alla fine del mese, verifica che i dati siano stati salvati
            if month_data:
                logger.info(f"Dati per {month_name} 2024 salvati in {output_file} - {len(month_data)} giorni totali")
            else:
                logger.warning(f"Nessun dato raccolto per {month_name} 2024")
    
    except Exception as e:
        logger.error(f"Errore durante l'esecuzione dello scraping: {e}")
    
    finally:
        # Chiudi il driver
        if driver is not None:
            try:
                driver.quit()
            except:
                pass
        logger.info("Scraping dei dati meteo completato!")

if __name__ == "__main__":
    main()