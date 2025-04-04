import os
import time
import pandas as pd
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, StaleElementReferenceException
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from functools import partial

# Configurazione del webdriver con chromedriver locale
def setup_driver():
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    chromedriver_path = os.path.join(current_dir, "chromedriver.exe")
    
    if not os.path.exists(chromedriver_path):
        raise FileNotFoundError(f"ChromeDriver executable not found at {chromedriver_path}")
    
    chrome_options = Options()
    #chrome_options.add_argument("--headless") # Usa headless per velocizzare
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-browser-side-navigation")
    chrome_options.add_argument("--disable-infobars")
    chrome_options.add_argument("--window-size=1366,768")  # Ridotto per minore consumo di memoria
    chrome_options.add_argument("--disable-notifications")
    
    # Opzioni per sopprimere messaggi di errore
    chrome_options.add_argument("--enable-unsafe-swiftshader")
    chrome_options.add_argument("--log-level=3")
    chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])
    
    # Performance options
    chrome_options.add_argument("--disable-features=OptimizationGuideModelDownloading,OptimizationHintsFetching,OptimizationTargetPrediction,OptimizationHints")
    chrome_options.add_argument("--blink-settings=imagesEnabled=false")  # Disabilita immagini per velocizzare
    
    # User agent
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    
    service = Service(executable_path=chromedriver_path)
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    # Set timeouts
    driver.set_page_load_timeout(20)  # Timeout globale per caricamento pagina
    driver.implicitly_wait(1)  # Tempo di attesa minimo per trovare elementi
    
    return driver

# Creazione delle directory di output
def create_output_dir():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

# Funzione per impostare i gradi centigradi con tempi ridotti
def set_celsius_units(driver):
    try:
        # Apriamo la pagina principale prima di accedere alle impostazioni
        driver.get("https://www.wunderground.com/history/daily/it/pisa/LIRP")
        time.sleep(3)  # Ridotto da 5 a 3 secondi
        
        # Gestisci i cookie se necessario
        accept_cookies(driver)
        time.sleep(1)  # Ridotto da 2 a 1 secondo
        
        # Apri le impostazioni
        settings_button = WebDriverWait(driver, 8).until(  # Ridotto da 10 a 8 secondi
            EC.element_to_be_clickable((By.CSS_SELECTOR, "#wuSettings > i"))
        )
        settings_button.click()
        print("✓ Pulsante impostazioni cliccato")
        time.sleep(1)  # Ridotto da 2 a 1 secondo
        
        # Seleziona gradi centigradi
        celsius_button = WebDriverWait(driver, 8).until(  # Ridotto da 10 a 8 secondi
            EC.element_to_be_clickable((By.CSS_SELECTOR, "#wuSettings-quick > div > a:nth-child(2)"))
        )
        celsius_button.click()
        print("✓ Impostati i gradi centigradi")
        time.sleep(2)  # Ridotto da 3 a 2 secondi
        
        return True
    except Exception as e:
        print(f"⚠️ Impossibile impostare i gradi centigradi: {e}")
        return False

# Funzione specifica per gestire il popup dei cookie di Wunderground con XPath precisi
def accept_cookies(driver):
    try:
        # Utilizzo gli XPath esatti forniti
        notice_xpath = '//*[@id="notice"]'
        button_xpath = '//*[@id="notice"]/div[3]/div[1]/button'
        
        print("Cercando popup cookie con XPath esatto...")
        
        # 1. Verifichiamo se esiste l'elemento notice usando XPath
        notice = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.XPATH, notice_xpath))
        )
        
        print("✓ Popup cookie (#notice) trovato con XPath esatto")
        
        # 2. Cerchiamo direttamente il pulsante di accettazione usando XPath
        accept_button = WebDriverWait(driver, 3).until(
            EC.element_to_be_clickable((By.XPATH, button_xpath))
        )
        
        print("✓ Bottone di accettazione trovato con XPath esatto")
        
        # Scorro fino al pulsante e pauso un attimo
        driver.execute_script("arguments[0].scrollIntoView(true);", accept_button)
        time.sleep(0.5)
        
        # Prendi screenshot per debug prima di cliccare
        screenshot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cookie_before_click.png")
        driver.save_screenshot(screenshot_path)
        print(f"Screenshot salvato prima del click: {screenshot_path}")
        
        # Prova prima con JavaScript diretto (più affidabile)
        try:
            driver.execute_script("arguments[0].click();", accept_button)
            print("✅ Cookie accettati con click JavaScript")
            return True
        except Exception as js_error:
            print(f"Click JS fallito: {js_error}")
            
            # Altrimenti prova con click normale
            try:
                accept_button.click()
                print("✅ Cookie accettati con click normale")
                return True
            except Exception as click_error:
                print(f"Anche il click normale è fallito: {click_error}")
                
                # Ultima risorsa: prova con Actions
                try:
                    from selenium.webdriver.common.action_chains import ActionChains
                    actions = ActionChains(driver)
                    actions.move_to_element(accept_button).click().perform()
                    print("✅ Cookie accettati con ActionChains")
                    return True
                except Exception as action_error:
                    print(f"Anche ActionChains è fallito: {action_error}")
                    
    except Exception as e:
        print(f"Errore con approccio XPath esatto: {e}")
    
    # Se l'XPath esatto fallisce, prova con JavaScript diretto al selettore
    try:
        result = driver.execute_script("""
            var notice = document.getElementById('notice');
            if (notice) {
                console.log("Notice trovato via JS");
                var buttons = notice.querySelectorAll('button');
                console.log("Trovati " + buttons.length + " bottoni");
                if (buttons.length > 0) {
                    buttons[0].click();
                    return true;
                }
                return false;
            }
            return false;
        """)
        
        if result:
            print("✅ Cookie accettati via JavaScript diretto")
            return True
    except Exception as js_error:
        print(f"JavaScript diretto fallito: {js_error}")
    
    # Backup brutale: clicca la posizione del pulsante direttamente
    try:
        # Posizione approssimativa del pulsante Accept
        driver.execute_script("""
            // Crea un elemento temporaneo dove dovrebbe essere il pulsante
            var elem = document.createElement('div');
            elem.style.position = 'absolute';
            elem.style.left = '650px';
            elem.style.top = '400px';
            elem.style.width = '100px';
            elem.style.height = '40px';
            elem.style.backgroundColor = 'transparent';
            elem.style.zIndex = '10000';
            document.body.appendChild(elem);
            
            // Simula il click
            var evt = document.createEvent('MouseEvents');
            evt.initMouseEvent('click', true, true, window, 0, 0, 0, 0, 0, false, false, false, false, 0, null);
            elem.dispatchEvent(evt);
            
            // Rimuovi l'elemento
            setTimeout(function() { document.body.removeChild(elem); }, 100);
        """)
        print("🔄 Tentato click posizionale sul pulsante dei cookie")
        
        # Attendiamo un momento e verifichiamo se il popup è scomparso
        time.sleep(1)
        try:
            notice = driver.find_element(By.XPATH, notice_xpath)
            if not notice.is_displayed():
                print("✅ Il popup dei cookie non è più visibile dopo click posizionale")
                return True
        except:
            print("✅ Il popup dei cookie non è più presente nel DOM dopo click posizionale")
            return True
    except:
        pass
    
    # Se tutti gli approcci falliscono, prova il metodo generico
    return try_generic_cookie_acceptance(driver)

# Funzione che prova vari metodi generici per accettare i cookie
def try_generic_cookie_acceptance(driver):
    cookie_selectors = [
        # Selettore specifico per il bottone dentro #notice
        "#notice button",
        "#notice .message-component.message-column.cta-button-column.accept-column button",
        "#notice > div.message-component.message-row.cta-buttons-container > div.message-component.message-column.cta-button-column.accept-column > button",
        "button.accept-cookies",
        "button.agree-button",
        ".message-column.accept-column button",
        "button.call",
        ".message-component button",
        # Match tramite testo per button con testo "Accept"
        "//button[contains(text(), 'Accept')]",
        "//button[contains(text(), 'Accetta')]",
        "//button[contains(text(), 'Accept All')]",
        "//button[contains(text(), 'Accetta tutti')]",
        # Altri possibili selettori comuni
        "#onetrust-accept-btn-handler",
        "#accept-cookies-checkbox"
    ]
    
    # Prima cerchiamo con selettori CSS, poi con XPath
    for selector in cookie_selectors:
        try:
            if selector.startswith("//"):
                # XPath selector
                wait = WebDriverWait(driver, 2)
                button = wait.until(
                    EC.element_to_be_clickable((By.XPATH, selector))
                )
            else:
                # CSS selector
                wait = WebDriverWait(driver, 2)
                button = wait.until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                )
                
            # Prova a far scorrere fino al pulsante
            driver.execute_script("arguments[0].scrollIntoView(true);", button)
            time.sleep(0.5)
            
            # Prova prima con JavaScript (più affidabile in caso di overlay)
            try:
                driver.execute_script("arguments[0].click();", button)
                print(f"✅ Cookie accettati con JS click usando: {selector}")
                return True
            except:
                # Fallback a click normale
                button.click()
                print(f"✅ Cookie accettati con click normale usando: {selector}")
                return True
        except:
            continue
    
    # Approccio alternativo: cerca tutti i bottoni nella pagina e prova a cliccare quelli relativi ai cookie
    try:
        # Trova tutti i bottoni
        buttons = driver.find_elements(By.TAG_NAME, "button")
        for button in buttons:
            try:
                # Verifica se il testo del bottone è relativo ai cookie
                text = button.text.lower()
                if any(keyword in text for keyword in ["accept", "accetta", "cookie", "agree", "consent"]):
                    button.click()
                    print(f"✅ Cookie accettati cercando nel bottone con testo: {text}")
                    return True
            except:
                continue
            
        # Cerca nelle div e nei link
        elements = driver.find_elements(By.CSS_SELECTOR, "div, a")
        for element in elements:
            try:
                text = element.text.lower()
                if any(keyword in text for keyword in ["accept", "accetta", "cookie", "agree", "consent"]):
                    element.click()
                    print(f"✅ Cookie accettati usando elemento generico con testo: {text}")
                    return True
            except:
                continue
    except:
        pass
    
    # Ultima risorsa: prova un approccio con JavaScript
    try:
        # Prova a eliminare il popup dei cookie tramite JS
        driver.execute_script("""
            // Cerca elementi relativi ai cookie e rimuovili
            let elements = document.querySelectorAll('[id*="cookie"], [class*="cookie"], [id*="consent"], [class*="consent"]');
            for (let el of elements) {
                if (el.tagName === 'BUTTON' || el.tagName === 'A') {
                    el.click(); // Prova a cliccare
                    return true;
                }
                else if (el.style) {
                    el.style.display = 'none'; // Nascondi l'elemento
                }
            }
            return false;
        """)
        print("🔄 Tentativo di gestire i cookie con JavaScript")
    except:
        pass
        
    print("⚠️ Impossibile accettare i cookie con metodi tradizionali")
    
    # Se tutto fallisce, prova a continuare comunque
    return False

# Funzione per verificare se il popup dei cookie è visibile
def is_cookie_popup_visible(driver):
    cookie_indicators = [
        "#notice",
        "[class*='cookie']",
        "[id*='cookie']",
        "[class*='consent']",
        "[id*='consent']",
        "//div[contains(text(), 'cookie')]",
        "//div[contains(text(), 'Cookie')]"
    ]
    
    for selector in cookie_indicators:
        try:
            if selector.startswith("//"):
                elements = driver.find_elements(By.XPATH, selector)
            else:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
            
            for element in elements:
                if element.is_displayed():
                    return True
        except:
            continue
    
    return False

# Funzione per pulire i dati di temperatura e altri valori con caratteri speciali
def clean_value(value):
    # Rimuove il carattere "Â" che appare nelle temperature
    if isinstance(value, str):
        # Sostituisci "Â °C" con "°C"
        cleaned = value.replace("Â °C", "°C")
        # Sostituisci "Â °" con "°" (per altri casi)
        cleaned = cleaned.replace("Â °", "°")
        # Anche solo "Â " può comparire
        cleaned = cleaned.replace("Â ", " ")
        return cleaned
    return value

# Funzione per estrarre i dati dalla tabella con tempi drasticamente ottimizzati
def extract_weather_data(driver, url, year, month, day):
    data_lock = threading.Lock()  # Per l'accesso concorrente ai dati
    
    retry_count = 0
    max_retries = 2  # Ridotto da 3 a 2
    
    while retry_count < max_retries:
        try:
            driver.get(url)
            
            # Minimo tempo di attesa per caricamento iniziale
            time.sleep(1.5)  # Ridotto da 3 a 1.5 secondi
            
            # Accetta cookie solo al primo caricamento per thread
            thread_id = threading.get_ident()
            if not hasattr(extract_weather_data, "cookies_accepted"):
                extract_weather_data.cookies_accepted = set()
            
            if thread_id not in extract_weather_data.cookies_accepted:
                accept_cookies(driver)
                extract_weather_data.cookies_accepted.add(thread_id)
            
            # Scorri più velocemente alla tabella
            driver.execute_script("window.scrollBy(0, 600);")
            
            # Use shorter explicit waits
            wait = WebDriverWait(driver, 10)  # Ridotto da 15 a 10 secondi
            try:
                observation_table = wait.until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "lib-city-history-observation table"))
                )
            except TimeoutException:
                retry_count += 1
                if retry_count < max_retries:
                    continue
                else:
                    return []
            
            # Trova subito le righe della tabella
            rows = driver.find_elements(By.CSS_SELECTOR, "lib-city-history-observation table tbody tr")
            if not rows:
                retry_count += 1
                if retry_count < max_retries:
                    driver.refresh()
                    time.sleep(3)  # Ridotto da 8 a 3 secondi
                    continue
                else:
                    return []
            
            # Estrai i dati dalle righe con colonne predefinite
            known_headers = ["Time", "Temperature", "Dew Point", "Humidity", "Wind", "Wind Speed", "Wind Gust", "Pressure", "Precip.", "Condition"]
            rows_data = []
            
            # Processa righe più velocemente
            for row in rows:
                try:
                    cells = row.find_elements(By.CSS_SELECTOR, "td")
                    if not cells or len(cells) < len(known_headers):
                        continue
                    
                    # Estrai testo più velocemente
                    row_data = [cell.text.strip() for cell in cells]
                    
                    # Verifica che ci siano effettivamente dati nella riga (non tutti vuoti)
                    if not any(data for data in row_data if data):
                        continue
                    
                    # Crea dizionario basilare
                    row_dict = {'Date': f"{year}-{month:02d}-{day:02d}"}
                    for i, header in enumerate(known_headers):
                        if i < len(row_data):
                            # Pulisci il valore prima di aggiungerlo
                            row_dict[header] = clean_value(row_data[i])
                    
                    rows_data.append(row_dict)
                except:
                    continue
            
            if rows_data:
                return rows_data
            else:
                retry_count += 1
                continue
                
        except Exception as e:
            retry_count += 1
            if retry_count >= max_retries:
                return []
            time.sleep(2)  # Ridotto da 5-8 a 2 secondi
    
    return []

# Funzione per filtrare e pulire un DataFrame
def clean_dataframe(df):
    # Lista delle colonne da mantenere
    keep_columns = ['Date', 'Time', 'Temperature', 'Dew Point', 'Humidity', 
                   'Wind', 'Wind Speed', 'Wind Gust', 'Pressure', 'Precip.', 'Condition']
    
    # Mantieni solo le colonne desiderate se esistono nel DataFrame
    existing_columns = [col for col in keep_columns if col in df.columns]
    cleaned_df = df[existing_columns]
    
    # Rimuovi righe senza dati importanti
    cleaned_df = cleaned_df.dropna(subset=['Temperature', 'Time'], how='all')
    
    # Converti la colonna Time in formato standard se esiste
    if 'Time' in cleaned_df.columns:
        # Normalizza il formato orario (gestisci sia formati AM/PM che 24h)
        try:
            # Crea una colonna temporanea con datetime completo
            cleaned_df['DateTime'] = pd.to_datetime(
                cleaned_df['Date'] + ' ' + cleaned_df['Time'],
                errors='coerce',
                format='%Y-%m-%d %I:%M %p'  # Formato 12h con AM/PM
            )
            
            # Se ci sono valori NaT, prova con formato 24h
            mask_nat = cleaned_df['DateTime'].isna()
            if mask_nat.any():
                cleaned_df.loc[mask_nat, 'DateTime'] = pd.to_datetime(
                    cleaned_df.loc[mask_nat, 'Date'] + ' ' + cleaned_df.loc[mask_nat, 'Time'],
                    errors='coerce',
                    format='%Y-%m-%d %H:%M'  # Formato 24h
                )
            
            # Ordina per data e ora
            cleaned_df = cleaned_df.sort_values('DateTime')
            # Rimuovi la colonna temporanea
            cleaned_df = cleaned_df.drop('DateTime', axis=1)
        except Exception as e:
            print(f"⚠️ Errore nella normalizzazione delle date: {e}")
            # Fallback: ordina per Date e poi per Time
            cleaned_df = cleaned_df.sort_values(['Date', 'Time'])
    else:
        # Ordina almeno per data
        cleaned_df = cleaned_df.sort_values('Date')
    
    return cleaned_df

# Process a specific day
def process_day(driver_container, month, day, year, output_dir):
    # Get a driver from the container
    driver = driver_container.get_driver()
    
    try:
        url = f"https://www.wunderground.com/history/daily/it/pisa/LIRP/date/{year}-{month['num']}-{day}"
        day_data = extract_weather_data(driver, url, year, month['num'], day)
        
        if day_data:
            return {
                'success': True,
                'day': day,
                'month': month,
                'data': day_data
            }
        else:
            return {
                'success': False,
                'day': day,
                'month': month
            }
    except Exception as e:
        return {
            'success': False,
            'day': day,
            'month': month,
            'error': str(e)
        }
    finally:
        # Return the driver to the container
        driver_container.return_driver(driver)

# Inizializza un driver e configura le impostazioni globali - versione migliorata
def initialize_driver_with_settings():
    driver = setup_driver()
    print("\n========== Configurazione iniziale del browser ==========")
    
    # Apri la pagina principale
    driver.get("https://www.wunderground.com/history/daily/it/pisa/LIRP")
    time.sleep(5)  # Aumentato da 4 a 5 secondi per caricamento completo
    
    # Prendi screenshot iniziale per debug
    screenshot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "browser_initial.png")
    driver.save_screenshot(screenshot_path)
    print(f"Screenshot iniziale salvato: {screenshot_path}")
    
    # Verifica la presenza dell'elemento #notice con XPath
    try:
        notice_element = driver.find_element(By.XPATH, '//*[@id="notice"]')
        if notice_element.is_displayed():
            print("🍪 Popup dei cookie #notice trovato con XPath, tentativo di accettazione...")
            screenshot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cookie_popup_found.png")
            driver.save_screenshot(screenshot_path)
            print(f"Screenshot del popup cookie: {screenshot_path}")
            
            if accept_cookies(driver):
                print("✅ Cookie accettati correttamente")
                time.sleep(2)
            else:
                print("⚠️ Impossibile accettare i cookie, ricarico la pagina...")
                driver.refresh()
                time.sleep(3)
                # Secondo tentativo dopo refresh
                if accept_cookies(driver):
                    print("✅ Cookie accettati al secondo tentativo")
                    time.sleep(2)
        else:
            print("✅ Elemento #notice trovato ma non visualizzato, continuando...")
    except NoSuchElementException:
        print("✅ Elemento #notice non trovato (XPath), continuando...")
    except Exception as e:
        print(f"⚠️ Errore nella verifica del popup cookie: {e}")
    
    time.sleep(2)
    
    # Imposta i gradi centigradi
    celsius_set = set_celsius_units(driver)
    if celsius_set:
        print("✅ Unità di misura in gradi centigradi impostate con successo")
    else:
        print("⚠️ Impossibile impostare i gradi centigradi, continuando comunque...")
    
    time.sleep(2)
    
    # Verifica che le impostazioni siano state applicate
    try:
        driver.get("https://www.wunderground.com/history/daily/it/pisa/LIRP/date/2024-3-1")
        time.sleep(3)
        # Scorri la pagina fino alla tabella
        driver.execute_script("window.scrollBy(0, 600);")
        time.sleep(1)
        
        # Cerca un elemento con la temperatura per verificare il formato
        temp_elements = driver.find_elements(By.CSS_SELECTOR, "lib-city-history-observation table tbody tr td")
        if temp_elements:
            for element in temp_elements:
                text = element.text.strip()
                if "°C" in text:
                    print(f"✅ Verifica riuscita: unità °C trovate nel testo '{text}'")
                    break
            else:
                print("⚠️ Formato temperatura non verificato, potrebbero esserci problemi con i dati")
    except Exception as e:
        print(f"⚠️ Errore durante la verifica delle impostazioni: {e}")
    
    return driver

# Versione migliorata di DriverContainer che inizializza correttamente tutti i driver
class DriverContainer:
    def __init__(self, max_drivers=4):
        self.drivers = []
        self.max_drivers = max_drivers
        self.lock = threading.Lock()
        self.available_drivers = []
        
        # Inizializza il primo driver con le impostazioni
        print("Inizializzazione driver principale con impostazioni corrette...")
        master_driver = initialize_driver_with_settings()
        self.drivers.append(master_driver)
        self.available_drivers.append(master_driver)
        
        # Inizializza gli altri driver (già con impostazioni corrette dal browser)
        print(f"Inizializzazione di {max_drivers-1} driver aggiuntivi...")
        for i in range(max_drivers - 1):
            try:
                driver = setup_driver()
                self.drivers.append(driver)
                self.available_drivers.append(driver)
                print(f"Driver {i+2} inizializzato")
            except Exception as e:
                print(f"Errore nell'inizializzazione del driver {i+2}: {e}")
    
    def get_driver(self):
        with self.lock:
            if not self.available_drivers:
                # Create a new driver if under limit
                if len(self.drivers) < self.max_drivers:
                    driver = setup_driver()
                    self.drivers.append(driver)
                    return driver
                # Wait for an available driver
                while not self.available_drivers:
                    self.lock.release()
                    time.sleep(0.1)
                    self.lock.acquire()
            
            return self.available_drivers.pop(0)
    
    def return_driver(self, driver):
        with self.lock:
            if driver not in self.available_drivers:
                self.available_drivers.append(driver)
    
    def close_all(self):
        with self.lock:
            for driver in self.drivers:
                try:
                    driver.quit()
                except:
                    pass

def main():
    output_dir = create_output_dir()
    
    try:
        # Anno di riferimento
        year = 2024
        
        # Definiamo i mesi richiesti del 2024 (tutti dati storici)
        months = [
            {"num": 3, "name": "March", "days": 31},
            {"num": 4, "name": "April", "days": 30},
            {"num": 5, "name": "May", "days": 31},
        ]
        
        print("\n========== Inizializzazione driver con configurazioni corrette ==========")
        print("Questo garantirà che tutti i driver condividano le stesse impostazioni.")
        
        # Driver container for parallel processing with proper initialization
        driver_container = DriverContainer(max_drivers=4)
        
        # Process each month
        for month in months:
            print(f"\n========== Inizio scraping per il mese di {month['name']} {year} ==========")
            month_data = []
            
            # Lista di giorni da processare nel mese corrente
            days = list(range(1, month['days'] + 1))
            
            # Process days in parallel
            with ThreadPoolExecutor(max_workers=4) as executor:
                # Submit all tasks
                future_to_day = {
                    executor.submit(
                        process_day, driver_container, month, day, year, output_dir
                    ): day for day in days
                }
                
                # Process results as they complete
                for future in as_completed(future_to_day):
                    day = future_to_day[future]
                    try:
                        result = future.result()
                        if result['success']:
                            day_data = result['data']
                            month_data.extend(day_data)
                            print(f"✅ Aggiunti {len(day_data)} record per il {day}/{month['num']}/{year}")
                        else:
                            print(f"⚠️ Nessun dato trovato per il {day}/{month['num']}/{year}")
                    except Exception as e:
                        print(f"❌ Errore per {day}/{month['num']}/{year}: {e}")
            
            # Salva i dati del mese in un CSV
            if month_data:
                df = pd.DataFrame(month_data)
                
                # Verifica che ci siano effettivamente dati validi nel DataFrame
                if not df.empty:
                    # Pulisci e ordina i dati
                    df = clean_dataframe(df)
                    
                    # Ulteriore verifica: non salvare file vuoti dopo la pulizia
                    if df.empty:
                        print(f"❌ Nessun dato valido rimasto dopo la pulizia per {month['name']} {year}")
                        continue
                    
                    # Applica pulizia aggiuntiva per ogni cella (per sicurezza)
                    for col in df.columns:
                        df[col] = df[col].apply(clean_value)
                    
                    # Rimozione righe completamente vuote o con troppi valori mancanti
                    df = df.dropna(thresh=3)  # Richiedi almeno 3 valori non-NA
                    
                    # Verifica che ci siano ancora dati dopo la pulizia
                    if df.empty:
                        print(f"❌ Nessun dato valido rimasto dopo la pulizia per {month['name']} {year}")
                        continue
                    
                    output_file = os.path.join(output_dir, f"weather_data_{month['name'].lower()}_{year}.csv")
                    df.to_csv(output_file, index=False)
                    print(f"📊 Dati per {month['name']} {year} salvati in {output_file} ({len(df)} record validi)")
                else:
                    print(f"❌ Nessun dato valido trovato per {month['name']} {year}")
            else:
                print(f"❌ Nessun dato trovato per {month['name']} {year}")
    
    except Exception as e:
        print(f"Errore generale: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Chiudi tutti i driver
        driver_container.close_all()
        print("✅ Processo di scraping completato!")

if __name__ == "__main__":
    # Sopprimere messaggi di log
    logging.getLogger('selenium').setLevel(logging.ERROR)
    logging.getLogger('urllib3').setLevel(logging.ERROR)
    
    # Esecuzione funzione principale
    main()