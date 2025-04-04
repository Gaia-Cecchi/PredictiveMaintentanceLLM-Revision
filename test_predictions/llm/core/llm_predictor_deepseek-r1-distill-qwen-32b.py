import time
import logging
import os
import json
import pickle
from typing import Dict, Any, Optional
import traceback
import sys
from concurrent.futures import ThreadPoolExecutor

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

# Fix the import to use the correct function
from ..prompts.expert_prompts import definitive_prompt

class LLMPredictor:
    """Manages LLM communication and response processing for compressor anomaly detection"""
    
    def __init__(self, api_key: str, model: str = "deepseek-r1-distill-qwen-32b"):
        """
        Initialize the LLM predictor
        
        Args:
            api_key: Groq API key
            model: LLM model to use, defaults to deepseek-r1-distill-qwen-32b
        """
        # Configure logging
        logging.basicConfig(level=logging.INFO, 
                           format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                           stream=sys.stdout)
        self.logger = logging.getLogger(__name__)
        
        # Initialize the LLM with optimized settings for deepseek model
        self.logger.info(f"Initializing ChatGroq with model: {model}")
        try:
            self.llm = ChatGroq(api_key=api_key, model_name=model)
            self.logger.info(f"ChatGroq client initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize ChatGroq client: {str(e)}")
            raise
            
        # Use definitive_prompt() instead of get_optimized_prompt()
        self.prompt = definitive_prompt()
        self.model = model
        
        # Optimized rate limiting parameters for deepseek model
        self.last_request_time = 0
        self.current_interval = 1.5  # Start with 1.5 second interval (reduced from 2.0)
        self.min_interval = 0.8      # Minimum interval between requests (reduced from 1.0)
        self.max_interval = 40.0     # Maximum interval between requests (reduced from 60.0)
        self.backoff_factor = 1.5    # Factor to increase interval after 429s
        self.success_factor = 0.85   # Factor to decrease interval after successes (more aggressive reduction)
        self.consecutive_successes = 0
        self.consecutive_failures = 0
        
        # Enhanced cache for avoiding redundant API calls
        self.prediction_cache = {}
        self.cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load persistent cache if exists
        self._load_cache()
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=2)
    
    def _load_cache(self):
        """Load prediction cache from disk if available"""
        cache_file = os.path.join(self.cache_dir, f"prediction_cache_{self.model.replace('-', '_')}.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    loaded_cache = pickle.load(f)
                    # Merge with existing cache
                    self.prediction_cache.update(loaded_cache)
                self.logger.info(f"Loaded {len(loaded_cache)} cached predictions from disk")
            except Exception as e:
                self.logger.warning(f"Failed to load cache from {cache_file}: {e}")
    
    def _save_cache(self):
        """Save prediction cache to disk periodically"""
        if len(self.prediction_cache) > 0:
            cache_file = os.path.join(self.cache_dir, f"prediction_cache_{self.model.replace('-', '_')}.pkl")
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(self.prediction_cache, f)
                self.logger.info(f"Saved {len(self.prediction_cache)} predictions to cache")
            except Exception as e:
                self.logger.warning(f"Failed to save cache to {cache_file}: {e}")
    
    def predict(self, event_time: str, compressor_data_str: str, weather_data_str: str, max_retries: int = 3) -> Dict[str, Any]:
        """
        Send data to LLM and get prediction
        
        Args:
            event_time: The timestamp of the event being analyzed
            compressor_data_str: String representation of compressor sensor data
            weather_data_str: String representation of weather data
            max_retries: Maximum number of retry attempts for API failures
            
        Returns:
            Dictionary containing the parsed prediction results
        """
        # Create more unique cache key with better hashing
        cache_key = f"{event_time}_{hash(compressor_data_str[:200])}"
        if cache_key in self.prediction_cache:
            self.logger.info(f"Using cached prediction for {event_time}")
            return self.prediction_cache[cache_key]
        
        # Ottimizzazione dell'estrazione dati: più aggressiva e mirata
        exact_readings_only = self._extract_relevant_data(compressor_data_str, event_time)
        
        # Ottimizzazione: analizza il meteo solo se è necessario (alcuni modelli non lo usano)
        if "weather" in self.prompt.lower():
            weather_summary = self._extract_relevant_data(weather_data_str, event_time)
        else:
            weather_summary = "N/A - Weather data not used for classification"
        
        # Template ottimizzato per velocità e precisione
        query_template = """
        Analizza dati compressore per {datetime}:

        LETTURE ESATTE:
        {exact_readings}

        METEO:
        {weather_summary}

        RISPONDI SOLO NEL FORMATO:
        CLASSIFICAZIONE: [ANOMALIA o VALORE NORMALE]
        TIPO: [SOLO se ANOMALIA: guasto cuscinetti/surriscaldamento/calo di pressione/squilibrio motore/fluttuazione tensione]
        CONFIDENZA: [alta/media/bassa]
        INDICATORI_CHIAVE: [2-3 letture]
        RACCOMANDAZIONE: [1 frase]
        """
        
        # Prepare the message with optimized data
        prompt = query_template.format(
            datetime=event_time,
            exact_readings=exact_readings_only,
            weather_summary=weather_summary
        )
        
        messages = [
            SystemMessage(content=self.prompt),
            HumanMessage(content=prompt)
        ]
        
        self.logger.info(f"Prepared messages for LLM: system={len(self.prompt)} chars, user={len(prompt)} chars")
        
        # Apply dynamic rate limiting before sending request
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        if (elapsed < self.current_interval):
            wait_time = self.current_interval - elapsed
            self.logger.info(f"Rate limiting: Waiting {wait_time:.2f}s between requests (current interval: {self.current_interval:.2f}s)")
            time.sleep(wait_time)
        
        # Update last request time
        self.last_request_time = time.time()
        
        # Implement retry logic with dynamic backoff
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Sending request to LLM (attempt {attempt+1}/{max_retries})")
                
                # Track request start time to measure response time
                request_start_time = time.time()
                
                # Add explicit error handling with detailed debug info
                try:
                    # Ottimizzazione parametri per deepseek-r1-distill-qwen-32b
                    # Questo modello può sfruttare parametri leggermente diversi per maggiore velocità
                    response = self.llm.invoke(
                        messages,
                        temperature=0.01,       # Mantiene decisioni deterministiche
                        max_tokens=120,         # Ridotto da 150 a 120 - sufficiente per l'output strutturato
                        stop=["CLASSIFICAZIONE:"],  # Evita ripetizioni nel formato
                        top_p=0.05,             # Ridotto per maggiore determinismo e velocità
                        frequency_penalty=0.0,
                        presence_penalty=0.0
                    )
                    
                    # Calculate response time
                    response_time = time.time() - request_start_time
                    self.logger.info(f"Response received in {response_time:.2f}s")
                    
                    # Adjust interval based on successful response - more aggressively for speed
                    self.consecutive_successes += 1
                    self.consecutive_failures = 0
                    
                    # After several consecutive successes, decrease interval more rapidly
                    if self.consecutive_successes >= 2:  # Reduced from 3 to 2 for faster adaptation
                        # Reduce interval but don't go below minimum
                        self.current_interval = max(self.min_interval, self.current_interval * self.success_factor)
                        self.logger.info(f"Decreased request interval to {self.current_interval:.2f}s after successful requests")
                        self.consecutive_successes = 0  # Reset counter
                    
                    # Log successful response
                    self.logger.info(f"LLM response received, length: {len(response.content)} chars")
                    
                    # Basic validation to ensure we got a proper response
                    if not response or not response.content or len(response.content) < 10:
                        raise ValueError(f"Empty or invalid response received from LLM: {response}")
                    
                except Exception as invoke_error:
                    self.logger.error(f"LLM invoke error: {str(invoke_error)}")
                    self.logger.error(traceback.format_exc())
                    raise
                
                # Ottimizzazione: parsing più veloce quando la risposta è ben formattata
                if "CLASSIFICAZIONE:" in response.content:
                    parsed_response = self._parse_response(response.content)
                else:
                    self.logger.warning("Response doesn't follow expected format, using enhanced parsing")
                    parsed_response = self._enhanced_parse_response(response.content)
                
                self.logger.info(f"LLM classification: {parsed_response['classification']}")
                
                # Add to cache
                self.prediction_cache[cache_key] = parsed_response
                
                # Ottimizzazione: salva la cache periodicamente in modo asincrono
                if len(self.prediction_cache) % 10 == 0:
                    self.executor.submit(self._save_cache)
                
                return parsed_response
                
            except Exception as e:
                self.logger.warning(f"LLM API call failed (attempt {attempt+1}): {str(e)}")
                
                # Dynamically adjust based on error type
                if "429" in str(e) or "Too Many Requests" in str(e):
                    # Rate limit error - increase interval more aggressively
                    self.consecutive_failures += 1
                    self.consecutive_successes = 0
                    
                    # Increase interval based on consecutive failures
                    backoff_multiplier = self.backoff_factor * (1 + 0.5 * self.consecutive_failures)
                    self.current_interval = min(self.max_interval, self.current_interval * backoff_multiplier)
                    
                    wait_time = min(self.current_interval * 1.5, 60)  # Ridotto da 120 a 60 secondi max
                    self.logger.info(f"Rate limited by API. Increased interval to {self.current_interval:.2f}s. Waiting {wait_time:.2f}s before retry.")
                    time.sleep(wait_time)
                elif attempt < max_retries - 1:
                    # Other error types - use standard backoff with reduced waits
                    wait_time = 2 ** (attempt + 1)  # Ridotto usando +1 invece di +2
                    self.logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"All {max_retries} attempts failed")
                    
                    # Fallback più veloce: restituisci una predizione di default piuttosto che errore
                    self.logger.info("Using fallback prediction after all retries failed")
                    fallback = {
                        'classification': 'NORMAL VALUE',  # Default a NORMAL VALUE come opzione più sicura
                        'type': '',
                        'confidence': 'low',
                        'key_indicators': 'Fallback prediction - API failure',
                        'recommendation': 'Verificare manualmente il sistema'
                    }
                    return fallback
    
    def _extract_relevant_data(self, data_str: str, event_time: str) -> str:
        """Extract only the most relevant lines from the data string for faster processing"""
        # Ottimizzazione: estrazione ancora più mirata e veloce
        relevant_lines = []
        
        # Convert event_time to a simpler format for string matching
        event_time_simple = str(event_time).split(" ")[0]
        
        # Prima riga è sempre l'intestazione
        header = data_str.split('\n')[0] if '\n' in data_str else "DateTime,Values"
        relevant_lines.append(header)
        
        # Estrai SOLO la riga esatta dell'evento (è quella più importante)
        for line in data_str.split('\n')[1:]:  # Saltiamo l'intestazione
            if event_time_simple in line:
                relevant_lines.append(line)
                break
        
        # Se non è stata trovata la riga esatta, prendi la prima riga disponibile
        if len(relevant_lines) == 1:  # Solo l'intestazione
            if len(data_str.split('\n')) > 1:
                relevant_lines.append(data_str.split('\n')[1])
        
        # Ottimizzazione: restituisci risultato minimale ma sufficiente
        return "\n".join(relevant_lines)
    
    def _parse_response(self, response_text: str) -> Dict[str, Optional[str]]:
        """
        Parse the structured response from the LLM - Optimized for speed
        """
        result = {
            'classification': None,
            'type': None,
            'confidence': None,
            'key_indicators': None,
            'recommendation': None
        }
        
        # Algoritmo ottimizzato per parsing veloce
        lines = response_text.strip().split('\n')
        for line in lines:
            line = line.strip()
            
            # Ottimizzazione: match più rapidi con lookups diretti
            if line.startswith("CLASSIFICAZIONE:"):
                result['classification'] = line[16:].strip()
            elif line.startswith("TIPO:"):
                result['type'] = line[5:].strip()
            elif line.startswith("CONFIDENZA:"):
                result['confidence'] = line[11:].strip()
            elif line.startswith("INDICATORI_CHIAVE:"):
                result['key_indicators'] = line[18:].strip()
            elif line.startswith("RACCOMANDAZIONE:"):
                result['recommendation'] = line[16:].strip()
            
            # English format (fallback)
            elif line.startswith("CLASSIFICATION:"):
                result['classification'] = line[14:].strip()
            elif line.startswith("TYPE:"):
                result['type'] = line[5:].strip()
            elif line.startswith("CONFIDENCE:"):
                result['confidence'] = line[11:].strip()
            elif line.startswith("KEY_INDICATORS:"):
                result['key_indicators'] = line[14:].strip()
            elif line.startswith("RECOMMENDATION:"):
                result['recommendation'] = line[14:].strip()
        
        # Normalize classification values to standard format - faster implementation
        if result['classification']:
            result['classification'] = 'ANOMALY' if 'ANOMAL' in result['classification'].upper() else 'NORMAL VALUE'
        else:
            result['classification'] = 'NORMAL VALUE'
        
        # Ensure required fields have values
        result['confidence'] = result['confidence'] or 'medium'
        result['key_indicators'] = result['key_indicators'] or "Classification based on available parameters."
        result['recommendation'] = result['recommendation'] or "Monitor system based on classification."
        
        return result
    
    def _enhanced_parse_response(self, response_text: str) -> Dict[str, Optional[str]]:
        """
        Enhanced parsing for non-standard responses - Optimized for flexibility
        """
        # Start with default values
        result = {
            'classification': 'NORMAL VALUE',  # Default
            'type': '',
            'confidence': 'medium',
            'key_indicators': '',
            'recommendation': 'Monitor system based on classification.'
        }
        
        # Check for anomaly indicators in the full text
        text_lower = response_text.lower()
        
        # Determine classification
        if any(term in text_lower for term in ['anomalia', 'anomaly', 'guasto', 'failure', 'surriscaldamento', 'overheating']):
            result['classification'] = 'ANOMALY'
            
            # Try to determine type
            if 'cuscinett' in text_lower or 'bearing' in text_lower:
                result['type'] = 'guasto cuscinetti'
            elif 'surriscal' in text_lower or 'overheat' in text_lower:
                result['type'] = 'surriscaldamento'
            elif 'pressione' in text_lower or 'pressure' in text_lower:
                result['type'] = 'calo di pressione'
            elif 'mot' in text_lower and ('imbalan' in text_lower or 'balanc' in text_lower):
                result['type'] = 'squilibrio motore'
            elif 'tensione' in text_lower or 'voltage' in text_lower:
                result['type'] = 'fluttuazione tensione'
            else:
                result['type'] = 'anomalia generica'
                
            # Extract confidence - heuristic approach
            if 'alta confidenza' in text_lower or 'high confidence' in text_lower or 'certezza' in text_lower:
                result['confidence'] = 'alta'
            elif 'bassa confidenza' in text_lower or 'low confidence' in text_lower:
                result['confidence'] = 'bassa'
                
            # Try to extract key indicators
            # Find numeric values with units like temperature
            import re
            temp_match = re.search(r'(\d+\.?\d*)°C|temperatura.{1,20}(\d+\.?\d*)', text_lower)
            vib_match = re.search(r'(\d+\.?\d*)\s*mm\/s|vibrazion.{1,20}(\d+\.?\d*)', text_lower)
            press_match = re.search(r'(\d+\.?\d*)\s*bar|pression.{1,20}(\d+\.?\d*)', text_lower)
            
            indicators = []
            if temp_match:
                group = temp_match.group(1) or temp_match.group(2)
                indicators.append(f"Temperatura: {group}°C")
            if vib_match:
                group = vib_match.group(1) or vib_match.group(2)
                indicators.append(f"Vibrazione: {group} mm/s")
            if press_match:
                group = press_match.group(1) or press_match.group(2)
                indicators.append(f"Pressione: {group} bar")
                
            if indicators:
                result['key_indicators'] = "; ".join(indicators)
            else:
                result['key_indicators'] = "Valori anomali rilevati"
                
            # Try to extract recommendation
            recom_match = re.search(r'(verificare|controllare|sostituire|fermare|manutenzione).{5,50}\.', text_lower)
            if recom_match:
                result['recommendation'] = recom_match.group(0).capitalize()
        
        return result