import os
from datetime import datetime, timedelta
import logging
from pathlib import Path
import sqlite3
import pandas as pd
from typing import Dict, List, Optional, Any  # Add Any to the imports
from dotenv import load_dotenv

from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun  # Add this import
import json
import importlib.util

# Define solution template function here if import fails
def local_get_solution_template(parameter: str, issue_type: str = None) -> str:
    """
    Fallback function to provide solution templates if the module import fails.
    
    Args:
        parameter: The parameter of interest (current, cosphi, etc.)
        issue_type: The type of issue detected (optional)
        
    Returns:
        A solution template as a string
    """
    # Simplified default templates
    templates = {
        "current": """
        PROCEDURA PER ANOMALIE DI CORRENTE:
        
        1. DIAGNOSI:
           - Verificare l'alimentazione principale al quadro elettrico
           - Controllare i fusibili e gli interruttori magnetotermici
           - Ispezionare i contatti del teleruttore principale
        
        2. INTERVENTO:
           - Ripristinare l'alimentazione se interrotta
           - Sostituire eventuali fusibili bruciati
           - Serrare morsetti allentati
           - Reset della centralina di controllo
        
        3. VERIFICA:
           - Avviare il compressore in modalità test
           - Verificare i valori di corrente
        """,
        
        "cosphi": """
        PROCEDURA PER ANOMALIE DI FATTORE DI POTENZA:
        
        1. DIAGNOSI:
           - Verificare banco di condensatori di rifasamento
           - Controllare efficienza del motore elettrico
           - Verificare carichi induttivi nel circuito
        
        2. INTERVENTO:
           - Sostituire condensatori difettosi
           - Controllare e pulire le connessioni
           - Verificare bilanciamento fasi
        
        3. VERIFICA:
           - Misurare il fattore di potenza a diversi carichi
        """,
        
        "energy_consumption": """
        PROCEDURA PER ANOMALIE DI CONSUMO ENERGETICO:
        
        1. DIAGNOSI:
           - Verificare presenza di perdite
           - Controllare pressione di esercizio
           - Ispezionare filtri aria e olio
        
        2. INTERVENTO:
           - Riparare perdite pneumatiche
           - Ottimizzare pressione di esercizio
           - Sostituire filtri intasati
        
        3. VERIFICA:
           - Monitorare consumo energetico
           - Verificare valori operativi
        """,
        
        "reactive_energy": """
        PROCEDURA PER ANOMALIE DI ENERGIA REATTIVA:
        
        1. DIAGNOSI:
           - Verificare efficienza del sistema di rifasamento
           - Controllare lo stato dei condensatori
        
        2. INTERVENTO:
           - Sostituire batterie di condensatori difettose
           - Pulire i contatti
        
        3. VERIFICA:
           - Monitorare consumo di energia reattiva
           - Verificare fattore di potenza
        """,
        
        "voltage": """
        PROCEDURA PER ANOMALIE DI TENSIONE:
        
        1. DIAGNOSI:
           - Verificare stabilità della rete elettrica
           - Controllare connessioni
        
        2. INTERVENTO:
           - Installare stabilizzatore se necessario
           - Verificare sezione cavi
        
        3. VERIFICA:
           - Monitorare tensione durante il funzionamento
        """
    }
    
    # Return the appropriate template or a generic one
    return templates.get(parameter, """
    PROCEDURA STANDARD:
    
    1. DIAGNOSI:
       - Eseguire verifica completa del sistema
       - Ispezionare componenti relativi al parametro anomalo
    
    2. INTERVENTO:
       - Sostituire o riparare i componenti difettosi
       - Eseguire reset e ricalibratura
    
    3. VERIFICA:
       - Testare il sistema a diversi carichi
       - Monitorare i parametri per almeno un ciclo completo
    """)

# Try to import solution_templates, fall back to local implementation if it fails
try:
    # Try to import from a file in the same directory
    current_dir = Path(__file__).parent
    solution_templates_path = current_dir / 'solution_templates.py'
    
    if (solution_templates_path.exists()):
        spec = importlib.util.spec_from_file_location("solution_templates", solution_templates_path)
        solution_templates = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(solution_templates)
        get_solution_template = solution_templates.get_solution_template
        logging.info("Successfully imported solution_templates module")
    else:
        # If file doesn't exist, use the local implementation
        get_solution_template = local_get_solution_template
        logging.warning("solution_templates.py not found, using local implementation")
except Exception as e:
    # If import fails for any reason, use the local implementation
    logging.error(f"Error importing solution_templates: {str(e)}")
    get_solution_template = local_get_solution_template

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class KnowledgeChatAssistant:
    def __init__(self, base_path="/teamspace/studios/this_studio"):
        # Load environment variables
        load_dotenv()
        
        self.base_path = Path(base_path)
        self.processed_data_path = self.base_path / 'processed_data'
        
        logger.info("Initializing Knowledge Chat Assistant...")
        
        # Initialize embedding model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Initialize vector store with existing data
        self.vector_store = self.initialize_vector_store()
        
        # Initialize LLM
        logger.info("Initializing LLM...")
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
            
        self.llm = ChatGroq(
            temperature=0.2,  # Reduced temperature for more factual responses
            model_name="qwen-2.5-32b",  # Using qwen-2.5-32b as requested
            groq_api_key=groq_api_key,
            max_tokens=1024  # Reduced from 2048 to save context
        )
        
        # Initialize conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Initialize system context
        self.system_context = ""
        
        # Modifichiamo l'inizializzazione del current_simulated_time
        self.current_simulated_time = None  # Inizializziamo a None
        
        # Aggiungiamo un dizionario per memorizzare il contesto delle anomalie correnti
        self.current_anomalies_context = None
        
        # Setup the conversation chain
        self.setup_chain()
        
        self.db_path = self.processed_data_path / 'compressor_data.db'
        
        logger.info("Knowledge Chat Assistant initialized successfully")

    def initialize_vector_store(self):
        """Initialize or load vector store"""
        vector_store_path = self.processed_data_path / 'vector_store'
        index_path = vector_store_path / 'faiss_index'
        
        if index_path.exists():
            logger.info("Caricamento vector store esistente con deserializzazione sicura abilitata...")
            return FAISS.load_local(
                str(index_path), 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
        else:
            logger.error("Vector store non trovato. Assicurarsi che il percorso sia corretto.")
            raise FileNotFoundError(f"Vector store non trovato in {index_path}")

    def set_current_time(self, timestamp: datetime):
        """Imposta l'ora simulata corrente per contestualizzare le risposte"""
        if not isinstance(timestamp, datetime):
            raise ValueError(f"timestamp deve essere datetime, non {type(timestamp)}")
            
        self.current_simulated_time = timestamp
        logger.info(f"Chat assistant: ora simulata impostata a {timestamp}")

    def load_data_up_to(self, timestamp: datetime):
        """Load all relevant data up to the given timestamp"""
        try:
            logger.info(f"Loading data up to {timestamp}")
            
            # Connect to the database
            with sqlite3.connect(self.db_path) as conn:
                # Load anomalies - LIMIT reduced to 50
                anomalies_query = """
                    SELECT 
                        timestamp,
                        compressor_id,
                        parameter,
                        is_anomaly,
                        true_value,
                        predicted_value,
                        error_value
                    FROM anomalies
                    WHERE datetime(timestamp) <= datetime(?)
                    AND is_anomaly = 1
                    ORDER BY timestamp DESC
                    LIMIT 50
                """
                
                anomalies_df = pd.read_sql_query(
                    anomalies_query, 
                    conn, 
                    params=[timestamp.strftime('%Y-%m-%d %H:%M:%S')],
                    parse_dates=['timestamp']
                )
                
                # Load maintenance records - LIMIT reduced to 20
                maintenance_query = """
                    SELECT 
                        date,
                        intervention_number,
                        intervention_type,
                        operating_hours,
                        activities,
                        anomalies,
                        recommendations,
                        compressor_id
                    FROM maintenance
                    WHERE datetime(date) <= datetime(?)
                    ORDER BY date DESC
                    LIMIT 20
                """
                
                maintenance_df = pd.read_sql_query(
                    maintenance_query,
                    conn,
                    params=[timestamp.strftime('%Y-%m-%d %H:%M:%S')],
                    parse_dates=['date']
                )
                
                # Load failures - LIMIT reduced to 20
                # Updated failures query to match the new schema
                failures_query = """
                    SELECT 
                        date,
                        failure_type,
                        frequency,
                        cause,
                        solution,
                        additional_info,
                        feedback,
                        compressor_id
                    FROM failures
                    WHERE date <= ?
                    ORDER BY date DESC
                    LIMIT 20
                """
                
                failures_df = pd.read_sql_query(
                    failures_query,
                    conn,
                    params=[timestamp.strftime('%Y-%m-%d %H:%M:%S')],
                    parse_dates=['date']
                )
                
                # Debug log the failures
                logger.info(f"Found {len(failures_df)} failures records")
                if not failures_df.empty:
                    logger.info(f"Sample failures data: {failures_df.iloc[0].to_dict()}")
                
                # Process failures data for context
                failures_context = ""
                if not failures_df.empty:
                    failures_context = "GUASTI RECENTI:\n"
                    for _, row in failures_df.iterrows():
                        # Format failures using new schema
                        failures_context += (
                            f"- {row['date'].strftime('%Y-%m-%d %H:%M')}, {row['compressor_id']}, " 
                            f"{row['failure_type']}: {row['cause']} -> {row['solution']}\n"
                            f"  Frequenza: {row['frequency']}\n"
                            f"  Info: {row.get('additional_info', 'N/A')}\n"
                            f"  Feedback: {row.get('feedback', 'N/A')}\n"
                        )
                
                # Rest of processing remains the same
                # Process anomalies data for context
                anomalies_context = ""
                if not anomalies_df.empty:
                    # Limit to 10 most recent anomalies
                    recent_anomalies = anomalies_df.head(10)
                    anomalies_context = "ANOMALIE RECENTI (10 PIÙ RECENTI):\n"
                    for _, group in recent_anomalies.groupby(['timestamp', 'compressor_id']):
                        for _, row in group.iterrows():
                            deviation = ((row['true_value'] - row['predicted_value']) / row['predicted_value']) * 100 if row['predicted_value'] != 0 else 0
                            anomalies_context += f"- {row['timestamp']}, {row['parameter']}: {deviation:.1f}%\n"
                
                # Process maintenance data for context
                maintenance_context = ""
                if not maintenance_df.empty:
                    maintenance_context = "MANUTENZIONI RECENTI:\n"
                    for _, row in maintenance_df.iterrows():
                        maintenance_context += f"- {row['date'].strftime('%Y-%m-%d')}, {row['compressor_id']}, {row['intervention_type']}: {row['activities']}\n"
                
                # Combine all context
                combined_context = f"""
                CONTESTO DEL SISTEMA AL {timestamp.strftime('%Y-%m-%d %H:%M:%S')}:
                
                {failures_context}
                
                {anomalies_context}
                
                {maintenance_context}
                """
                
                # Store context in memory for the LLM
                self.system_context = combined_context
                logger.info("Data loading complete")
                
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}", exc_info=True)
            self.system_context = f"Error loading data: {str(e)}"

    def setup_chain(self):
        """Setup the conversational chain with custom prompt"""
        logger.info("Setting up conversation chain...")
        
        # Create the template with detailed system instructions - updated for more confidence and proactivity
        system_template = """Sei un esperto tecnico specializzato in compressori industriali con anni di esperienza sul campo.

        👉 **APPROCCIO ALLA RISPOSTA**:
        1. Rispondi con SICUREZZA e AUTOREVOLEZZA, come un tecnico esperto sul campo
        2. Sii PROATTIVO nel suggerire soluzioni e nel guidare l'utente alla risoluzione dei problemi
        3. Usa un tono DIRETTO e PRATICO, senza esprimere incertezze quando hai informazioni sufficienti
        4. Quando fai riferimento a dati storici, collegali SEMPRE alla situazione attuale
        5. Fai sempre riferimento a PROCEDURE standardizzate per la manutenzione

        👉 **STRUTTURA DELLE RISPOSTE PER PROBLEMI TECNICI**:
        1. DIAGNOSI - Identifica con sicurezza la causa probabile del problema
        2. CORRELAZIONE - Collega il problema attuale a situazioni simili del passato
        3. SOLUZIONE - Proponi SEMPRE almeno una soluzione concreta e operativa
        4. PREVENZIONE - Suggerisci come prevenire lo stesso problema in futuro

        👉 **REGOLE DI BASE**:
        1. Basati sulla knowledge base fornita, ma usa la tua esperienza per dare risposte operative
        2. Quando non hai informazioni precise, usa le best practice del settore
        3. Non dire MAI "la knowledge base non fornisce" o "non ho informazioni sufficienti"
        4. Se una soluzione precisa non è disponibile, proponi comunque interventi standard e line guida
        5. Rispondi SEMPRE in italiano con linguaggio tecnico appropriato
        6. Considera sempre che l'ora attuale simulata è: {current_time}
        7. Mantieni la consapevolezza temporale, ignorando eventi futuri rispetto all'ora simulata

        KNOWLEDGE BASE:
        {context}

        CONTESTO DEL SISTEMA:
        {system_context}

        DOMANDA UTENTE:
        {question}
        """
        
        prompt = ChatPromptTemplate.from_template(system_template)
        
        # Creazione della memory personalizzata che salverà solo il messaggio umano e la risposta
        # bypassando l'errore relativo all'output_key
        from langchain.memory import ConversationBufferWindowMemory
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            input_key="question",
            output_key="answer",  # Qui specifichiamo che solo "answer" deve essere salvato
            k=5,  # Conserva solo gli ultimi 5 scambi
            return_messages=True
        )
        
        # Modifichiamo la configurazione della catena
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 8}),
            chain_type="stuff",
            memory=self.memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": prompt},
            get_chat_history=lambda h: h,  # Funzione personalizzata per ottenere la cronologia
            verbose=True
        )
        
        logger.info("Conversation chain setup completed")

    async def chat(self, user_input: str) -> Dict:
        """Process user input and return a response"""
        try:
            # Verifichiamo che l'ora sia stata impostata
            if self.current_simulated_time is None:
                raise ValueError("L'ora simulata non è stata impostata")
                
            logger.info(f"Processing user input: {user_input} at time {self.current_simulated_time}")
            
            # Check for statistical queries about anomalies, frequency, patterns
            stats_keywords = ["quante anomalie", "quanto spesso", "frequenza", "statistiche", "andamento", "trend", "quanti guasti", "ultimo mese", "ultima settimana"]
            
            time_periods = {
                "giorno": 1,
                "settimana": 7,
                "mese": 30,
                "anno": 365
            }
            
            # Check if this is a statistical query about anomalies
            is_stats_query = any(keyword in user_input.lower() for keyword in stats_keywords)
            time_period = None
            
            for period, days in time_periods.items():
                if period in user_input.lower() or f"ultim{period}" in user_input.lower():
                    time_period = days
                    break
            
            if is_stats_query:
                # Default to monthly if no specific period mentioned
                if time_period is None:
                    time_period = 30  # Default to last month
                
                return await self._generate_anomaly_statistics(time_period)
            
            # Handle specific anomaly parameter queries
            if self.current_anomalies_context and any(p in user_input.lower() for p in ["current", "cosphi", "energia", "energy", "tensione", "voltage", "reattiva", "reactive"]):
                # Extract which parameter the user is asking about
                param_keywords = {
                    "current": ["current", "corrente", "amperaggio"],
                    "cosphi": ["cosphi", "cos phi", "fattore di potenza", "cos φ", "cosfase"],
                    "energy_consumption": ["energy", "energia", "consumo", "consumi", "kwh"],
                    "reactive_energy": ["reactive", "reattiva", "reattivo", "varh"],
                    "voltage": ["voltage", "tensione", "volt"]
                }
                
                # Find which parameter is being referenced
                target_param = None
                for param, keywords in param_keywords.items():
                    if any(keyword in user_input.lower() for keyword in keywords):
                        target_param = param
                        break
                
                # Find if it's a solution request
                is_solution_request = any(term in user_input.lower() for term in ["risolvere", "risoluzione", "come posso", "soluzioni", "risolverne", "soluzione"])
                
                if target_param:
                    # Get the specific anomaly data for this parameter
                    param_anomalies = [a for a in self.current_anomalies_context if a['parameter'] == target_param]
                    
                    if param_anomalies and is_solution_request:
                        # Create a focused dataframe for this parameter only
                        param_df = pd.DataFrame(param_anomalies)
                        # Generate a solution response for just this parameter
                        return await self._generate_specific_param_solution(param_df, target_param)
            
            # Handle generic anomaly-related queries
            if any(phrase in user_input.lower() for phrase in ["anomalia attuale", "anomalia corrente", "anomalie in corso", "ci sono anomalie"]):
                # Controlla le anomalie nel database per il timestamp corrente
                with sqlite3.connect(self.db_path) as conn:
                    query = """
                        SELECT 
                            timestamp,
                            compressor_id,
                            parameter,
                            is_anomaly,
                            true_value,
                            predicted_value,
                            error_value
                        FROM anomalies
                        WHERE datetime(timestamp) = datetime(?)
                        AND is_anomaly = 1
                        ORDER BY parameter
                    """
                    
                    df = pd.read_sql_query(
                        query,
                        conn,
                        params=[self.current_simulated_time.strftime('%Y-%m-%d %H:%M:%S')]
                    )
                    
                    if not df.empty:
                        # Memorizza le anomalie trovate come contesto
                        self.current_anomalies_context = df.to_dict('records')
                        
                        if "come" in user_input.lower() or "risolvere" in user_input.lower() or "risolverle" in user_input.lower():
                            # È una richiesta di risoluzione, fornisci soluzioni
                            return await self._generate_solutions_response(df)
                        else:
                            # È solo una richiesta informativa sulle anomalie
                            response_text = "Ho trovato le seguenti anomalie in corso:\n\n"
                            for _, row in df.iterrows():
                                deviation = ((row['true_value'] - row['predicted_value']) / row['predicted_value']) * 100
                                response_text += f"- Parametro {row['parameter']}: valore attuale {row['true_value']:.3f}, "
                                response_text += f"previsto {row['predicted_value']:.3f}, variazione {deviation:.1f}%\n"
                            
                            return {
                                "answer": response_text,
                                "sources": [f"anomalies_{self.current_simulated_time.strftime('%Y-%m-%d_%H:%M')}"]
                            }
                    elif self.current_anomalies_context:
                        # Non ci sono anomalie nel DB ma abbiamo quelle memorizzate nella conversazione
                        if "come" in user_input.lower() or "risolvere" in user_input.lower() or "risolverle" in user_input.lower():
                            # Converti il contesto memorizzato in DataFrame
                            context_df = pd.DataFrame(self.current_anomalies_context)
                            return await self._generate_solutions_response(context_df)
                        else:
                            # Mostra le anomalie memorizzate
                            response_text = "Basandomi sulle anomalie discusse in precedenza:\n\n"
                            for row in self.current_anomalies_context:
                                deviation = ((row['true_value'] - row['predicted_value']) / row['predicted_value']) * 100
                                response_text += f"- Parametro {row['parameter']}: valore attuale {row['true_value']:.3f}, "
                                response_text += f"previsto {row['predicted_value']:.3f}, variazione {deviation:.1f}%\n"
                            
                            return {
                                "answer": response_text,
                                "sources": [f"anomalies_{self.current_simulated_time.strftime('%Y-%m-%d_%H:%M')}"]
                            }
                    else:
                        return {
                            "answer": f"Non ci sono anomalie in corso alla data e ora attualmente considerate ({self.current_simulated_time.strftime('%Y-%m-%d %H:%M:%S')}).",
                            "sources": []
                        }
            
            # Handle follow-up questions about resolving anomalies in general
            if self.current_anomalies_context and any(term in user_input.lower() for term in ["risolvere", "risolverle", "risolverne", "come posso", "soluzioni"]):
                # Convert the context to DataFrame
                context_df = pd.DataFrame(self.current_anomalies_context)
                return await self._generate_solutions_response(context_df)
            
            # Per altre richieste, usa il comportamento standard
            formatted_time = self.current_simulated_time.strftime("%Y-%m-%d %H:%M:%S")
            
            # Special handling for feedback-related queries - MODIFIED to reduce feedback bias
            feedback_terms = ["feedback", "operatori", "recensioni", "valutazioni", "commenti", "opinioni", "problemi"]
            if any(term in user_input.lower() for term in feedback_terms):
                logger.info("Feedback-related query detected, but using balanced retrieval")
                # We'll now use regular retriever but with balanced parameters
                k_value = 12  # Use standard k value
                retriever = self.vector_store.as_retriever(
                    search_kwargs={
                        "k": k_value, 
                        "search_type": "mmr", 
                        "fetch_k": 30,
                        "lambda_mult": 0.7  # Adjusted for diversity
                    }
                )
            
            # Enhanced retrieval for manual-related queries - MODIFIED for balance
            elif any(term in user_input.lower() for term in ["manuale", "manual", "csd102", "pdf", "documentazione"]):
                logger.info("Manual-related query detected, using enhanced retrieval")
                k_value = 12  # Standard value
                search_type = "mmr"
                retriever = self.vector_store.as_retriever(
                    search_kwargs={"k": k_value, "search_type": search_type, "fetch_k": 25, "lambda_mult": 0.7}
                )
            else:
                # For standard queries, use a balanced approach
                k_value = 10  # Balanced value
                search_type = "mmr"  # Use MMR for diversity
                retriever = self.vector_store.as_retriever(
                    search_kwargs={"k": k_value, "search_type": search_type, "fetch_k": 20, "lambda_mult": 0.7}
                )
            
            # Add document type filter to ensure balanced results
            balanced_retriever = self._create_balanced_retriever(retriever, user_input)
            
            # Use balanced retriever instead of basic filtered retriever
            filtered_retriever = FilteredRetriever(
                retriever=balanced_retriever,
                current_time=self.current_simulated_time
            )
            
            # Define a more concise system template
            system_template = """Sei un assistente tecnico per compressori industriali.

            👉 **IMPORTANTE**:
            1. Risposte brevi e concise
            2. Basa le risposte sui dati disponibili
            3. Rispondi SEMPRE in italiano
            4. Ora simulata: {current_time}
            5. Per pressione atmosferica, usa sempre "PRESSIONESLM mb"
            6. Ignora dati futuri rispetto all'ora simulata

            KNOWLEDGE BASE:
            {context}

            CONTESTO DEL SISTEMA:
            {system_context}

            DOMANDA UTENTE:
            {question}
            """
            
            # Creazione della memoria personalizzata
            from langchain.memory import ConversationBufferWindowMemory
            memory = ConversationBufferWindowMemory(
                memory_key="chat_history",
                input_key="question",
                output_key="answer",
                k=3,  # Reduced from 5 to save context
                return_messages=True
            )
            
            # Restore previous messages
            if hasattr(self.memory, 'chat_memory') and hasattr(self.memory.chat_memory, 'messages'):
                memory.chat_memory.messages = self.memory.chat_memory.messages
            
            # Update class memory
            self.memory = memory
            
            # Crea una catena temporanea con il retriever filtrato
            temp_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=filtered_retriever,
                chain_type="stuff",
                memory=memory,
                return_source_documents=True,
                combine_docs_chain_kwargs={
                    "prompt": ChatPromptTemplate.from_template(system_template)
                },
                get_chat_history=lambda h: h,
                verbose=True
            )
            
            result = await temp_chain.ainvoke({
                "question": user_input,
                "current_time": formatted_time,
                "system_context": self.system_context if hasattr(self, 'system_context') else ""
            })
            
            # Extract sources for reference, ensuring they don't reference future events
            sources = []
            if "source_documents" in result and result["source_documents"]:
                for doc in result["source_documents"]:
                    source = doc.metadata.get('source', 'Non specificata')
                    
                    # Check if source contains a date and filter out future dates
                    skip = False
                    if isinstance(source, str):
                        date_patterns = [
                            r'(\d{4})-(\d{2})-(\d{2})',  # YYYY-MM-DD
                            r'(\d{4})(\d{2})(\d{2})',    # YYYYMMDD
                        ]
                        
                        for pattern in date_patterns:
                            import re
                            matches = re.findall(pattern, source)
                            for match in matches:
                                try:
                                    if isinstance(match, tuple) and len(match) == 3:
                                        year, month, day = int(match[0]), int(match[1]), int(match[2])
                                        doc_date = datetime(year, month, day)
                                        if doc_date > self.current_simulated_time:
                                            skip = True
                                            logger.warning(f"Filtered out future source: {source}, date: {doc_date}")
                                            break
                                except (ValueError, TypeError) as e:
                                    logger.error(f"Error parsing date from source {source}: {e}")
                    
                    if not skip and source not in sources:
                        sources.append(source)
                
                sources = list(set(sources))  # Remove duplicates
                logger.info(f"Found sources after filtering: {sources}")
            else:
                logger.warning("No source documents returned by the LLM chain")
            
            # Ensure the answer doesn't reference future events
            answer = result["answer"]
            answer = self._sanitize_answer_for_time_consistency(answer)
            
            return {
                "answer": answer,
                "sources": sources if sources else ["Nessuna fonte specifica"]
            }
                
        except Exception as e:
            logger.error(f"Error in chat: {str(e)}")
            return {"answer": f"Mi dispiace, si è verificato un errore: {str(e)}", "sources": []}

    async def _generate_anomaly_statistics(self, days: int) -> Dict:
        """Generate statistical information about anomalies over a time period"""
        try:
            # Calculate start date based on days and current simulated time
            start_date = self.current_simulated_time - timedelta(days=days)
            formatted_start = start_date.strftime('%Y-%m-%d %H:%M:%S')
            formatted_end = self.current_simulated_time.strftime('%Y-%m-%d %H:%M:%S')
            
            logger.info(f"Generating anomaly statistics from {formatted_start} to {formatted_end}")
            
            with sqlite3.connect(self.db_path) as conn:
                # Query for anomalies in the period
                query = """
                    SELECT 
                        strftime('%Y-%m-%d', timestamp) as date,
                        parameter,
                        COUNT(*) as count
                    FROM anomalies
                    WHERE datetime(timestamp) BETWEEN datetime(?) AND datetime(?)
                    AND is_anomaly = 1
                    GROUP BY date, parameter
                    ORDER BY date
                """
                
                anomalies_df = pd.read_sql_query(
                    query,
                    conn,
                    params=[formatted_start, formatted_end]
                )
                
                # Get total count and other statistics
                total_days_with_data = len(anomalies_df['date'].unique()) if not anomalies_df.empty else 0
                total_anomalies = anomalies_df['count'].sum() if not anomalies_df.empty else 0
                parameters_affected = anomalies_df['parameter'].nunique() if not anomalies_df.empty else 0
                
                # Get param-specific stats if available
                param_stats = {}
                if not anomalies_df.empty:
                    for param in anomalies_df['parameter'].unique():
                        param_count = anomalies_df[anomalies_df['parameter'] == param]['count'].sum()
                        param_stats[param] = param_count
                
                # Query for the most recent anomalies
                recent_query = """
                    SELECT 
                        timestamp,
                        parameter,
                        true_value,
                        predicted_value,
                        error_value
                    FROM anomalies
                    WHERE datetime(timestamp) BETWEEN datetime(?) AND datetime(?)
                    AND is_anomaly = 1
                    ORDER BY timestamp DESC
                    LIMIT 3
                """
                
                recent_anomalies = pd.read_sql_query(
                    recent_query,
                    conn,
                    params=[formatted_start, formatted_end],
                    parse_dates=['timestamp']
                )
                
                # Get anomaly frequency
                if total_days_with_data > 0:
                    avg_anomalies_per_day = total_anomalies / total_days_with_data
                    frequency_desc = f"In media, ci sono state {avg_anomalies_per_day:.1f} anomalie per giorno con dati."
                else:
                    frequency_desc = "Non sono stati trovati giorni con anomalie nel periodo richiesto."
                
                # Build response using string concatenation instead of problematic f-string
                time_period_name = "giorno" if days == 1 else "settimana" if days == 7 else "mese" if days <= 31 else "periodo"
                article = "" if days <= 3 else "l'" if days <= 31 else " "
                
                response = (
                    f"## Analisi Anomalie degli ultimi {days} giorni\n\n"
                    f"**Riepilogo**:\n"
                    f"- Ne{article}{time_period_name} ({formatted_start} - {formatted_end}) "
                    f"sono state rilevate **{total_anomalies} anomalie** in totale.\n"
                    f"- Sono stati interessati **{parameters_affected} parametri** diversi del compressore.\n"
                    f"- {frequency_desc}\n\n"
                )

                # Add parameter-specific details if available
                if param_stats:
                    response += "\n**Dettaglio per parametro**:\n"
                    for param, count in param_stats.items():
                        perc = (count / total_anomalies * 100) if total_anomalies > 0 else 0
                        response += f"- **{param}**: {count} anomalie ({perc:.1f}%)\n"
                
                # Add recent examples if available
                if not recent_anomalies.empty and len(recent_anomalies) > 0:
                    response += "\n**Anomalie recenti**:\n"
                    for _, row in recent_anomalies.iterrows():
                        ts = row['timestamp'].strftime('%Y-%m-%d %H:%M')
                        dev = ((row['true_value'] - row['predicted_value']) / row['predicted_value'] * 100) if row['predicted_value'] != 0 else 0
                        response += f"- {ts}: {row['parameter']}, deviazione {dev:.1f}%\n"
                
                # Add recommendations
                if total_anomalies > 0:
                    response += "\n**Raccomandazioni**:\n"
                    # If one parameter is dominant
                    if param_stats and max(param_stats.values()) / total_anomalies > 0.5:
                        dominant_param = max(param_stats.items(), key=lambda x: x[1])[0]
                        response += f"- Si consiglia un'ispezione approfondita del parametro **{dominant_param}** che rappresenta la maggioranza delle anomalie.\n"
                    
                    response += "- Verificare i trend e gli andamenti giornalieri per identificare possibili pattern ricorrenti.\n"
                    
                    if days > 7:
                        response += "- Confrontare con gli interventi di manutenzione effettuati nel periodo per valutare l'efficacia.\n"
                else:
                    response += "\n**Osservazioni**:\n"
                    response += "- Non sono state rilevate anomalie nel periodo selezionato. Il sistema ha funzionato entro i parametri previsti.\n"
                
                return {
                    "answer": response,
                    "sources": ["anomalies database", "historical_records", "statistical_analysis"]
                }
                
        except Exception as e:
            logger.error(f"Error generating anomaly statistics: {str(e)}")
            return {
                "answer": f"Mi dispiace, si è verificato un errore nell'analisi delle statistiche: {str(e)}",
                "sources": []
            }

    async def _generate_specific_param_solution(self, param_df, param_name):
        """Generate solution for a specific parameter anomaly"""
        try:
            # Get the first anomaly data (should only be one parameter)
            anomaly = param_df.iloc[0]
            
            # Calculate deviation
            deviation = ((anomaly['true_value'] - anomaly['predicted_value']) / anomaly['predicted_value']) * 100
            
            # Determine issue type based on parameter and deviation
            issue_type = None
            if param_name == 'current':
                if abs(deviation) > 90:
                    issue_type = "interruption"
                elif deviation < 0:
                    issue_type = "fluctuation"
                else:
                    issue_type = "overload"
            elif param_name == 'cosphi':
                if deviation < 0:
                    issue_type = "low"
                else:
                    issue_type = "unstable"
            elif param_name in ['energy_consumption', 'reactive_energy']:
                if deviation > 0:
                    issue_type = "high"
                else:
                    issue_type = "spike"
            
            # Get solution template
            solution_template = get_solution_template(param_name, issue_type)
            
            # Get past solutions for this parameter
            past_solutions = await self._find_past_solutions([param_name])
            
            # Create detailed response
            response = f"""## Soluzione per anomalia: {param_name.upper()}

**Analisi dettagliata**:
- Valore attuale: {anomaly['true_value']:.3f}
- Valore previsto: {anomaly['predicted_value']:.3f}
- Variazione: {deviation:.1f}%
- Gravità: {self._determine_severity(deviation)}

**Diagnosi**:
L'anomalia nel parametro {param_name} è classificata come "{issue_type if issue_type else 'standard'}", 
che tipicamente indica {self._get_issue_explanation(param_name, issue_type, deviation)}.

**Procedura di intervento raccomandata**:
{solution_template}

**Precedenti soluzioni efficaci**:
{past_solutions if len(past_solutions) > 50 else "Nessun intervento storico rilevante trovato per questo tipo specifico di anomalia."}
"""
            
            return {
                "answer": response,
                "sources": ["solution_templates.py", "historical_maintenance_data", f"anomalies_{self.current_simulated_time.strftime('%Y-%m-%d_%H:%M')}"]
            }
            
        except Exception as e:
            logger.error(f"Error generating parameter solution: {str(e)}")
            return {
                "answer": f"Mi dispiace, c'è stato un errore nell'analisi della soluzione per {param_name}: {str(e)}",
                "sources": []
            }

    def _determine_severity(self, deviation):
        """Determine severity level based on deviation percentage"""
        if abs(deviation) > 50:
            return "GRAVE"
        elif abs(deviation) > 20:
            return "MEDIA"
        else:
            return "LIEVE"

    def _get_issue_explanation(self, param, issue_type, deviation):
        """Get explanation for a specific issue type"""
        explanations = {
            "current": {
                "interruption": "un'interruzione o un guasto grave nel circuito elettrico",
                "fluctuation": "instabilità nell'alimentazione o problemi nei componenti rotanti",
                "overload": "un sovraccarico del sistema o un'eccessiva resistenza meccanica"
            },
            "cosphi": {
                "low": "un problema nel sistema di rifasamento o nei carichi induttivi",
                "unstable": "variazioni nel carico di lavoro o problemi intermittenti nei condensatori"
            },
            "energy_consumption": {
                "high": "un'eccessiva richiesta di energia dovuta a perdite o inefficienze",
                "spike": "picchi di consumo anomali che potrebbero indicare malfunzionamenti"
            },
            "reactive_energy": {
                "high": "uno squilibrio nel sistema elettrico o problemi nei condensatori",
                "spike": "un deterioramento del sistema di compensazione dell'energia reattiva"
            },
            "voltage": {
                "fluctuation": "problemi nella rete di alimentazione o nei componenti di regolazione"
            }
        }
        
        if param in explanations and issue_type in explanations[param]:
            return explanations[param][issue_type]
        else:
            # Generic explanation based on deviation direction
            if deviation > 0:
                return "un valore superiore al previsto, che può indicare un sovraccarico o un'inefficienza"
            else:
                return "un valore inferiore al previsto, che può indicare un malfunzionamento o un'interruzione parziale"

    async def _generate_solutions_response(self, anomalies_df):
        """Generate solutions for current anomalies"""
        try:
            # Prepara una risposta strutturata per le soluzioni
            response = "## Analisi e soluzioni per le anomalie rilevate\n\n"
            
            # Raggruppa le anomalie per parametro
            params = anomalies_df['parameter'].unique()
            
            # Per ogni parametro, prepara una sezione di soluzione
            for param in params:
                # Ottieni i dati dell'anomalia
                param_anomalies = anomalies_df[anomalies_df['parameter'] == param]
                anomaly = param_anomalies.iloc[0]
                
                # Calcola la deviazione percentuale
                deviation = ((anomaly['true_value'] - anomaly['predicted_value']) / anomaly['predicted_value']) * 100
                
                # Determina il tipo di problema in base alla deviazione
                if param == 'current' and abs(deviation) > 90:
                    issue_type = "interruption"
                elif param == 'current' and deviation < 0:
                    issue_type = "fluctuation"
                elif param == 'current' and deviation > 0:
                    issue_type = "overload"
                elif param == 'cosphi' and deviation < 0:
                    issue_type = "low"
                elif param == 'cosphi':
                    issue_type = "unstable"
                elif param in ['energy_consumption', 'reactive_energy'] and deviation > 0:
                    issue_type = "high"
                else:
                    issue_type = None
                
                # Ottieni il template della soluzione per questo parametro
                solution_template = get_solution_template(param, issue_type)
                
                # Aggiungi soluzione al contesto
                past_solutions = await self._find_past_solutions([param])
                
                response += f"### ANOMALIA: {param.upper()}\n\n"
                response += f"**Analisi**: Valore attuale {anomaly['true_value']:.3f} vs previsto {anomaly['predicted_value']:.3f} "
                response += f"(deviazione: {deviation:.1f}%)\n\n"
                
                if abs(deviation) > 50:
                    severity = "GRAVE"
                elif abs(deviation) > 20:
                    severity = "MEDIA"
                else:
                    severity = "LIEVE"
                
                response += f"**Gravità**: {severity}\n\n"
                response += "**Soluzione raccomandata**:\n"
                response += solution_template + "\n\n"
                
                if past_solutions and len(past_solutions) > 50:
                    response += "**Soluzioni utilizzate in passato per problemi simili**:\n"
                    response += past_solutions + "\n\n"
            
            # Aggiorna il contesto del sistema con queste informazioni
            self.system_context += "\n\nANOMALIE ANALIZZATE RECENTEMENTE:\n" + response
            
            return {
                "answer": response,
                "sources": ["solution_templates.py", "historical_maintenance_data", f"anomalies_{self.current_simulated_time.strftime('%Y-%m-%d_%H:%M')}"]
            }
            
        except Exception as e:
            logger.error(f"Error generating solutions: {str(e)}")
            return {
                "answer": f"Mi dispiace, c'è stato un errore nell'analisi delle soluzioni: {str(e)}",
                "sources": []
            }

    async def _find_past_solutions(self, parameters: list) -> str:
        """
        Find past solutions for similar problems related to the specified parameters
        """
        try:
            param_string = ", ".join(parameters) if parameters else "tutti i parametri"
            
            # Define search queries for each common problem type
            queries = [
                f"soluzioni per problemi di {param_string}",
                f"riparazione anomalie {param_string}",
                f"manutenzione correttiva {param_string}",
                f"interventi tecnici su {param_string}"
            ]
            
            all_solutions = []
            
            # Query the vector database for relevant past solutions
            for query in queries:
                docs = self.vector_store.similarity_search(
                    query, 
                    k=3,
                    filter={"type": {"$in": ["maintenance", "failure"]}}
                )                
                if docs:
                    for doc in docs:
                        solution_text = doc.page_content
                        source = doc.metadata.get('source', 'documento tecnico')
                        
                        # Format the solution nicely
                        formatted = f"SOLUZIONE PRECEDENTE ({source}):\n{solution_text}\n"
                        if formatted not in all_solutions:  # Avoid duplicates
                            all_solutions.append(formatted)
            
            if not all_solutions:
                # Provide standard solutions if no specific solutions found
                standard_solutions = {
                    "current": """
                    Procedura standard per anomalie di corrente:
                    1. Verificare connessioni elettriche
                    2. Controllare stato dei fusibili
                    3. Ispezionare il circuito di alimentazione
                    4. Controllare lo stato del motore elettrico
                    5. Verificare eventuali sovraccarichi
                    """,
                    "cosphi": """
                    Procedura standard per anomalie del fattore di potenza:
                    1. Verificare banco di condensatori
                    2. Controllare bilanciamento delle fasi
                    3. Verificare carichi induttivi
                    4. Controllare efficienza del motore
                    """,
                    "energy_consumption": """
                    Procedura standard per anomalie di consumo energetico:
                    1. Verificare perdite d'aria nel sistema
                    2. Controllare filtri intasati
                    3. Verificare stato della trasmissione
                    4. Controllare la pressione di esercizio
                    """,
                    "voltage": """
                    Procedura standard per anomalie di tensione:
                    1. Controllare stabilità dell'alimentazione
                    2. Verificare connessioni elettriche
                    3. Controllare resistenza degli avvolgimenti
                    4. Ispezionare quadro elettrico
                    """
                }
                
                for param in parameters:
                    if param in standard_solutions:
                        all_solutions.append(standard_solutions[param])
                
                # Add general procedure if no params specified or no specific solutions
                if not all_solutions:
                    all_solutions.append("""
                    PROCEDURA STANDARD:
                    1. Eseguire diagnosi strumentale completa
                    2. Verificare registro manutenzioni precedenti
                    3. Controllare componenti secondo manuale tecnico
                    4. Eseguire reset dei parametri operativi
                    5. Verificare funzionamento a carico ridotto
                    """)
            
            return "\n".join(all_solutions)
            
        except Exception as e:
            logger.error(f"Error finding past solutions: {str(e)}")
            return "Nessuna soluzione storica rilevante trovata. Applicare procedure standard."

    def _sanitize_answer_for_time_consistency(self, answer: str) -> str:
        """
        Sanitize the answer to ensure it doesn't reference future events
        relative to the current simulated time
        """
        # List of phrases that might indicate references to future events
        future_phrases = [
            "si verificherà", "accadrà", "avverrà", "sarà disponibile", "sarà presente",
            "ci sarà", "nel futuro", "prossimamente", "a breve", "tra poco",
            "futuro", "previsto per", "programmato per", "pianificato per",
            "in programma", "più avanti", "successivamente"
        ]
        
        # If any future phrase is detected, replace the answer with a standard one
        for phrase in future_phrases:
            if phrase in answer.lower():
                logger.warning(f"Found future reference in answer: '{phrase}'")
                return (
                    f"Mi dispiace, posso fornire informazioni solo fino alla data attuale simulata "
                    f"({self.current_simulated_time.strftime('%Y-%m-%d %H:%M:%S')}). "
                    f"Non posso fare riferimento a eventi futuri rispetto a questa data."
                )
        
        return answer

    def get_conversation_history(self):
        """Return the conversation history"""
        return self.memory.chat_memory.messages

    def clear_conversation_history(self):
        """Clear the conversation history"""
        self.memory.clear()
        # Resetta anche il contesto delle anomalie correnti
        self.current_anomalies_context = None
        return "Cronologia conversazione cancellata."

    def _create_balanced_retriever(self, base_retriever, query):
        """Create a balanced retriever that gives equal importance to different document types"""
        categories = [
            "maintenance", "failure", "anomaly", "operating_condition", "feedback"
        ]
        
        # Create a custom retriever class that ensures balanced results
        class BalancedRetriever(BaseRetriever):
            def __init__(self, retriever, categories):
                # Call parent class constructor
                super().__init__()
                # Initialize attributes properly
                self._retriever = retriever
                self._categories = categories
                
            def _get_relevant_documents(self, query, *, run_manager=None):
                # Get a larger number of documents
                docs = self._retriever.get_relevant_documents(query)
                return self._balance_documents(docs)
                
            async def _aget_relevant_documents(self, query, *, run_manager=None):
                docs = await self._retriever.aget_relevant_documents(query)
                return self._balance_documents(docs)
            
            def _balance_documents(self, docs):
                # Group documents by category
                categorized_docs = {}
                for cat in self._categories:
                    categorized_docs[cat] = []
                
                # Default category for uncategorized docs
                categorized_docs["other"] = []
                
                for doc in docs:
                    doc_type = doc.metadata.get("type", "")
                    # Check if document belongs to one of our categories
                    found = False
                    for cat in self._categories:
                        if cat in doc_type.lower():
                            categorized_docs[cat].append(doc)
                            found = True
                            break
                    
                    if not found:
                        categorized_docs["other"].append(doc)
                
                # Balance documents
                balanced_docs = []
                target_per_category = max(1, len(docs) // (len(self._categories) + 1))
                
                # Add documents from each category
                for cat in self._categories + ["other"]:
                    balanced_docs.extend(categorized_docs[cat][:target_per_category])
                
                # Fill remaining slots with leftover docs
                remaining_slots = len(docs) - len(balanced_docs)
                if remaining_slots > 0:
                    remaining_docs = []
                    for cat in self._categories + ["other"]:
                        remaining_docs.extend(categorized_docs[cat][target_per_category:])
                    balanced_docs.extend(remaining_docs[:remaining_slots])
                
                return balanced_docs

        # Create and return the balanced retriever
        return BalancedRetriever(base_retriever, categories)

class FilteredRetriever(BaseRetriever):
    """Retriever that filters out documents with future dates"""
    retriever: Any  # The base retriever
    current_time: datetime  # The current simulated time
    
    class Config:
        arbitrary_types_allowed = True
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """Synchronous method to get documents relevant to a query."""
        # Get documents from base retriever
        docs = self.retriever.get_relevant_documents(query)
        # Limit to 10 documents to save context
        return self._filter_documents(docs[:10])
    
    async def _aget_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """Asynchronous method to get documents relevant to a query."""
        # Get documents from base retriever
        docs = await self.retriever.aget_relevant_documents(query)
        # Limit to 10 documents to save context
        return self._filter_documents(docs[:10])
    
    def _filter_documents(self, docs: List[Document]) -> List[Document]:
        """Filter out documents with future dates"""
        filtered_docs = []
        
        for doc in docs:
            # Check if document metadata contains a date
            date_field = None
            if 'date' in doc.metadata:
                date_field = doc.metadata['date']
            elif 'timestamp' in doc.metadata:
                date_field = doc.metadata['timestamp']
            
            # If document has a date, check if it's in the future
            if date_field:
                try:
                    if isinstance(date_field, str):
                        doc_date = datetime.fromisoformat(date_field.replace('Z', '+00:00'))
                    elif isinstance(date_field, datetime):
                        doc_date = date_field
                    else:
                        # If date format is unknown, include the document
                        filtered_docs.append(doc)
                        continue
                        
                    if doc_date <= self.current_time:
                        filtered_docs.append(doc)
                    else:
                        logger.info(f"Filtered out future document with date: {doc_date}")
                except Exception as e:
                    # If there's an error parsing the date, include the document
                    logger.warning(f"Error parsing date from document: {e}")
                    filtered_docs.append(doc)
            else:
                # Check the content for date patterns
                skip = False
                content = doc.page_content
                
                # Look for date patterns in the content
                date_patterns = [r'(\d{4})-(\d{2})-(\d{2})']  # YYYY-MM-DD
                
                for pattern in date_patterns:
                    import re
                    matches = re.findall(pattern, content)
                    for match in matches:
                        try:
                            if len(match) == 3:
                                year, month, day = int(match[0]), int(match[1]), int(match[2])
                                doc_date = datetime(year, month, day)
                                if doc_date > self.current_time:
                                    skip = True
                                    logger.warning(f"Filtered out document with future date in content: {doc_date}")
                                    break
                        except (ValueError, TypeError):
                            # If there's an error parsing the date, continue
                            continue
                
                # Include document if it doesn't have a future date
                if not skip:
                    filtered_docs.append(doc)
                
        logger.info(f"Filtered from {len(docs)} to {len(filtered_docs)} documents")
        return filtered_docs

    async def _handle_recent_failures_query(self, days_lookback: int = 30) -> Dict:
        """Handle a query about recent failures"""
        try:
            end_date = self.current_simulated_time
            start_date = end_date - timedelta(days=days_lookback)
            
            logger.info(f"Querying failures between {start_date} and {end_date}")
            
            with sqlite3.connect(self.db_path) as conn:
                # Debug: Check actual date format in database
                check_query = """
                SELECT date, typeof(date) as date_type 
                FROM failures 
                LIMIT 1
                """
                check_df = pd.read_sql_query(check_query, conn)
                logger.info(f"Date format check: {check_df.to_dict('records')}")
                
                # Query failures with BETWEEN clause and date string handling
                query = """
                SELECT 
                    strftime('%Y-%m-%d %H:%M', date) as date,
                    failure_type,
                    cause,
                    solution,
                    additional_info,
                    feedback,
                    compressor_id
                FROM failures 
                WHERE datetime(date) BETWEEN datetime(?) AND datetime(?)
                AND compressor_id = 'CSD102'
                ORDER BY date DESC
                """
                
                params = [
                    start_date.strftime('%Y-%m-%d %H:%M:%S'),
                    end_date.strftime('%Y-%m-%d %H:%M:%S')
                ]
                
                logger.info(f"Query parameters: {params}")
                failures_df = pd.read_sql_query(query, conn, params=params)
                logger.info(f"Found {len(failures_df)} failures")
                
                # Convert date strings to datetime objects
                failures_df['date'] = pd.to_datetime(failures_df['date'])
                
                if not failures_df.empty:
                    logger.info(f"Date range in results: from {failures_df['date'].min()} to {failures_df['date'].max()}")
                    logger.info("Sample data:")
                    logger.info(failures_df.head().to_string())
                    
                    response_text = "Di recente, per il compressore CSD102, ci sono stati i seguenti guasti:\n\n"
                    for _, row in failures_df.iterrows():
                        date_str = row['date'].strftime('%Y-%m-%d')
                        
                        response_text += (
                            f"Data: {date_str} - "
                            f"{row['failure_type']} "
                            f"({row['cause']})\n"
                        )
                    
                    response_text += "\nQuesti guasti hanno avuto impatti vari, da un aumento della temperatura fino a una riduzione dell'efficienza del compressore."
                    
                    return {
                        "answer": response_text,
                        "sources": [
                            f"failures_CSD102_{pd.to_datetime(row['date']).strftime('%Y-%m-%d')}"
                            for _, row in failures_df.iterrows()
                        ]
                    }
                else:
                    logger.warning(f"No failures found between {start_date} and {end_date}")
                    return {
                        "answer": f"Non ci sono guasti registrati negli ultimi {days_lookback} giorni.",
                        "sources": ["failures_database"]
                    }
                    
        except Exception as e:
            logger.error(f"Error in _handle_recent_failures_query: {str(e)}", exc_info=True)
            return {
                "answer": f"Mi dispiace, c'è stato un errore nella ricerca dei guasti: {str(e)}",
                "sources": []
            }

def chat_loop():
    """Simple command-line interface for interacting with the assistant"""
    assistant = KnowledgeChatAssistant()
    print("\n===== Knowledge Chat Assistant =====")
    print("Digita 'exit' o 'quit' per uscire.")
    print("Digita 'clear' per cancellare la cronologia della conversazione.")
    print("=======================================\n")
    
    import asyncio
    
    async def process_input():
        while True:
            user_input = input("\nTu: ")
            
            if user_input.lower() in ['exit', 'quit']:
                print("\nGrazie per aver utilizzato il Knowledge Chat Assistant. Arrivederci!")
                break
            
            if user_input.lower() == 'clear':
                print(assistant.clear_conversation_history())
                continue
                
            print("\nAssistant: ", end="")
            
            response = await assistant.chat(user_input)
            print(response["answer"])
            
            if response["sources"]:
                print("\nFonti:", ", ".join(response["sources"]))
    
    # Run the async function
    asyncio.run(process_input())

def main():
    chat_loop()

if __name__ == "__main__":
    main()