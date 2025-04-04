import os
from datetime import datetime, timedelta
import logging
from pathlib import Path
import sqlite3
import pandas as pd
from typing import Dict, List, Optional
from dotenv import load_dotenv

from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CompressorAssistant:
    def __init__(self, base_path="/teamspace/studios/this_studio"):
        # Load environment variables
        load_dotenv()
        
        self.base_path = Path(base_path)
        self.processed_data_path = self.base_path / 'processed_data'
        
        logger.info("Initializing Compressor Assistant...")
        
        # Initialize embedding model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Initialize vector store with new data
        self.vector_store = self.initialize_vector_store()
        
        # Initialize LLM
        logger.info("Initializing LLM...")
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
            
        self.llm = ChatGroq(
            temperature=0.3,
            model_name="qwen-2.5-32b",
            groq_api_key=groq_api_key,
            max_tokens=4096
        )
        
        # Initialize conversation memory with explicit output key
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"  # Explicitly set output key
        )
        
        # Setup the conversation chain without chat functionality
        self.setup_chain()
        
        logger.info("Compressor Assistant initialized successfully")
        
        # NON impostiamo più un timestamp fisso qui
        self.current_time = None
        
        # Definiamo solo il range valido
        self.valid_dates = {
            'start': datetime(2024, 3, 1),
            'end': datetime(2024, 6, 30)
        }
        
        logger.info(f"Valid date range: {self.valid_dates['start']} to {self.valid_dates['end']}")
        
        # Add database path
        self.db_path = self.processed_data_path / 'compressor_data.db'
        
        # Initialize system context
        self.system_context = ""
        
        # Load initial data
        self.load_data_up_to(datetime.now())
        
        logger.info("Compressor Assistant initialized successfully")

    def initialize_vector_store(self):
        """Initialize or load vector store"""
        vector_store_path = self.processed_data_path / 'vector_store'
        index_path = vector_store_path / 'faiss_index'
        
        if (index_path.exists()):
            logger.info("Caricamento vector store esistente con deserializzazione sicura abilitata...")
            return FAISS.load_local(
                str(index_path), 
                self.embeddings, 
                allow_dangerous_deserialization=True  # Abilitiamo la deserializzazione sicura
            )
        
        # Load and process combined data
        combined_data_path = vector_store_path / 'combined_data.txt'
        with open(combined_data_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_text(text)
        
        # Create vector store
        vector_store = FAISS.from_texts(texts, self.embeddings)
        vector_store.save_local(str(index_path))
        return vector_store

    def set_current_time(self, timestamp: datetime):
        """Imposta il timestamp corrente per tutte le operazioni"""
        if not (self.valid_dates['start'] <= timestamp <= self.valid_dates['end']):
            logger.warning(f"Warning: timestamp {timestamp} is outside valid range")
            
        self.current_time = timestamp
        logger.info(f"Set current_time to: {self.current_time}")

    def get_weather_data(self, date: datetime) -> Dict:
        """Retrieve weather data for a specific date"""
        with sqlite3.connect(self.processed_data_path / 'compressor_data.db') as conn:
            query = """
                SELECT *
                FROM weather_data
                WHERE date = ?
            """
            df = pd.read_sql_query(query, conn, params=[date.date()])
            return df.to_dict('records')[0] if not df.empty else None

    def get_anomaly_data(self, timestamp: datetime, compressor_id: str) -> Dict:
        """Retrieve anomaly data for a specific timestamp and compressor"""
        try:
            # Ensure timestamp is string in correct format
            if isinstance(timestamp, (pd.Timestamp, datetime)):
                timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
            else:
                timestamp_str = pd.to_datetime(timestamp).strftime('%Y-%m-%d %H:%M:%S')
            
            logger.info(f"Querying with timestamp: {timestamp_str}")
            
            with sqlite3.connect(self.processed_data_path / 'compressor_data.db') as conn:
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
                    WHERE strftime('%Y-%m-%d %H:%M:%S', timestamp) = strftime('%Y-%m-%d %H:%M:%S', ?)
                    AND compressor_id = ?
                """
                
                # Debug: print exact query and parameters
                logger.info(f"SQL Query: {query}")
                logger.info(f"Parameters: timestamp={timestamp_str}, compressor_id={compressor_id}")
                
                # Execute query with string parameters
                df = pd.read_sql_query(
                    query,
                    conn,
                    params=(timestamp_str, compressor_id),
                    parse_dates=['timestamp']
                )
                
                logger.info(f"Query returned {len(df)} rows")
                if not df.empty:
                    logger.info(f"First row: {df.iloc[0].to_dict()}")
                
                return df.to_dict('records') if not df.empty else None
                
        except Exception as e:
            logger.error(f"Error in get_anomaly_data: {str(e)}", exc_info=True)
            raise  # Rilanciamo l'eccezione per debug

    def get_compressor_measurements(self, timestamp: datetime, compressor_id: str) -> Dict:
        """Retrieve compressor measurements for a specific timestamp"""
        # Ensure timestamp is string in correct format
        if isinstance(timestamp, (pd.Timestamp, datetime)):
            timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
        else:
            timestamp_str = pd.to_datetime(timestamp).strftime('%Y-%m-%d %H:%M:%S')
            
        with sqlite3.connect(self.processed_data_path / 'compressor_data.db') as conn:
            query = """
                SELECT *
                FROM compressor_measurements
                WHERE strftime('%Y-%m-%d %H:%M:%S', timestamp) = strftime('%Y-%m-%d %H:%M:%S', ?)
                AND compressor_id = ?
            """
            
            df = pd.read_sql_query(
                query,
                conn,
                params=(timestamp_str, compressor_id),
                parse_dates=['timestamp']
            )
            
            return df.to_dict('records')[0] if not df.empty else None

    def get_recent_failures(self, analysis_timestamp: datetime, compressor_id: str) -> List[Dict]:
        """
        Retrieve failures in the last two weeks before the analysis timestamp for a specific compressor
        """
        try:
            two_weeks_ago = analysis_timestamp - timedelta(days=14)
            
            # Debug logging
            logger.info(f"""
            Debug get_recent_failures:
            - Analysis timestamp: {analysis_timestamp}
            - Two weeks ago: {two_weeks_ago}
            - Compressor ID: {compressor_id}
            """)
            
            with sqlite3.connect(self.processed_data_path / 'compressor_data.db') as conn:
                # First check column names in the failures table
                check_schema_query = """
                    PRAGMA table_info(failures)
                """
                schema_df = pd.read_sql_query(check_schema_query, conn)
                logger.info(f"Failures table schema: {schema_df['name'].tolist()}")
                
                # Now check what's actually in the database
                check_query = """
                    SELECT * FROM failures 
                    LIMIT 3
                """
                check_df = pd.read_sql_query(check_query, conn)
                logger.info(f"Sample failures data:\n{check_df.head() if not check_df.empty else 'No data'}")
                
                # Then do the filtered query with the correct schema
                query = """
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
                    WHERE compressor_id = ?
                    AND date BETWEEN ? AND ?
                    ORDER BY date DESC
                """
                
                params = [
                    compressor_id,
                    two_weeks_ago.strftime('%Y-%m-%d %H:%M:%S'),
                    analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S')
                ]
                
                logger.info(f"Query params: {params}")
                
                df = pd.read_sql_query(query, conn, params=params, parse_dates=['date'])
                
                # Add days_ago for easier reference
                if not df.empty:
                    df['days_ago'] = df['date'].apply(lambda x: (analysis_timestamp.date() - x.date()).days)
                    logger.info(f"Found {len(df)} failures. Sample: {df.iloc[0].to_dict() if not df.empty else 'No data'}")
                else:
                    logger.warning("No failures found in the given date range")
                
                return df.to_dict('records') if not df.empty else []
                
        except Exception as e:
            logger.error(f"Error in get_recent_failures: {str(e)}")
            logger.exception("Full traceback:")
            return []

    def get_workload_data(self, timestamp: datetime, compressor_id: str) -> List[Dict]:
        """Get workload data for a specific timestamp and compressor"""
        try:
            # Get data from the week before the timestamp
            week_before = timestamp - timedelta(days=7)
            
            with sqlite3.connect(self.processed_data_path / 'compressor_data.db') as conn:
                query = """
                    SELECT 
                        date,
                        start_time,
                        end_time,
                        operating_hours,
                        load_percentage,
                        temperature,
                        humidity,
                        vibration,
                        pressure
                    FROM operating_conditions
                    WHERE date BETWEEN ? AND ?
                    AND compressor_id = ?
                    ORDER BY date DESC
                """
                
                df = pd.read_sql_query(
                    query,
                    conn,
                    params=(week_before.strftime('%Y-%m-%d'), 
                            timestamp.strftime('%Y-%m-%d'),
                            compressor_id)
                )
                
                if df.empty:
                    return []
                
                return df.to_dict('records')
                
        except Exception as e:
            logger.error(f"Error getting workload data: {str(e)}")
            return []

    def load_data_up_to(self, timestamp: datetime):
        """Load all relevant data up to the given timestamp"""
        try:
            logger.info(f"Loading data up to {timestamp}")
            
            with sqlite3.connect(self.db_path) as conn:
                # Load only failures from the last 7 days instead of 50 records
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
                    WHERE date BETWEEN ? AND ?
                    ORDER BY date DESC
                """
                
                # Get only last 7 days of failures
                start_date = timestamp - timedelta(days=7)
                
                failures_df = pd.read_sql_query(
                    failures_query,
                    conn,
                    params=[start_date.strftime('%Y-%m-%d %H:%M:%S'),
                           timestamp.strftime('%Y-%m-%d %H:%M:%S')],
                    parse_dates=['date']
                )
                
                # Process failures data for context - keep it brief
                failures_context = ""
                if not failures_df.empty:
                    failures_context = "GUASTI RECENTI:\n"
                    for _, row in failures_df.iterrows():
                        # Simplified failure entry with only essential info
                        failures_context += f"""
                        {row['date'].strftime('%Y-%m-%d %H:%M')} - {row['failure_type']}
                        Causa: {row.get('cause', 'N/A')}
                        ---
                        """
                
                # Store minimal context
                self.system_context = f"""
                CONTESTO AL {timestamp.strftime('%Y-%m-%d %H:%M:%S')}:
                {failures_context}
                """
                
                logger.info("Data loading complete")
                
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}", exc_info=True)
            self.system_context = f"Error loading data: {str(e)}"

    async def get_historical_failures(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Get failure records between dates from the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # First verify table exists and schema
                check_query = "PRAGMA table_info(failures)"
                schema = pd.read_sql_query(check_query, conn)
                logger.info(f"Failures table columns: {schema['name'].tolist()}")
                
                query = """
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
                    WHERE date BETWEEN ? AND ?
                    AND compressor_id = 'CSD102'
                    ORDER BY date DESC
                """
                
                logger.info(f"Executing failures query for {start_date} to {end_date}")
                
                params = [
                    start_date.strftime('%Y-%m-%d %H:%M:%S'),
                    end_date.strftime('%Y-%m-%d %H:%M:%S')
                ]
                
                df = pd.read_sql_query(query, conn, params=params, parse_dates=['date'])
                
                # Log results for debugging
                logger.info(f"Found {len(df)} failures")
                if not df.empty:
                    logger.info(f"Sample failure: {df.iloc[0].to_dict()}")
                
                return df.to_dict('records') if not df.empty else []
                
        except Exception as e:
            logger.error(f"Error in get_historical_failures: {str(e)}")
            logger.exception("Full traceback:")
            return []

    def setup_chain(self):
        """Setup the conversational chain with custom prompt"""
        logger.info("Setting up conversation chain...")

        # Create a direct LLM chain instead of ConversationalRetrievalChain
        from langchain.chains import LLMChain
        
        template = """Sei un esperto assistente di manutenzione per compressori industriali.
        
        Analizza le seguenti informazioni e fornisci una valutazione strutturata.
        
        FORMATO RISPOSTA OBBLIGATORIO:
        1. ANOMALIE ATTUALI
           - Per ogni parametro anomalo:
             * Nome parametro
             * Valore attuale vs previsto 
             * Variazione % con segno (+ o -)
             * Gravità (ALTA >50%, MEDIA 20-50%, BASSA <20%)

        2. CORRELAZIONI STORICHE
           - Pattern simili nel passato
           - Guasti correlati negli ultimi 14 giorni
           - Condizioni operative correlate

        3. ANALISI E RACCOMANDAZIONI
           - Cause probabili
           - Azioni immediate
           - Misure preventive

        CONTESTO:
        {context}

        DATI DA ANALIZZARE:
        {input_data}
        """

        # Create a simple prompt template with two variables
        prompt = ChatPromptTemplate.from_template(template)
        
        # Use a simple LLMChain for anomaly analysis
        self.analysis_chain = LLMChain(
            llm=self.llm,
            prompt=prompt,
            verbose=True
        )
        
        # Keep the standard chain for retrieval functions
        from langchain.chains import RetrievalQA
        
        qa_prompt = ChatPromptTemplate.from_template(
            "Risponde alle seguenti domande sui compressori in base al contesto fornito:\n\nContesto: {context}\n\nDomanda: {question}"
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 2}),
            chain_type_kwargs={"prompt": qa_prompt}
        )
        
        logger.info("Chains setup completed")

    async def analyze_anomaly(self, timestamp: datetime, compressor_id: str) -> str:
        """Analyze anomaly data for a specific timestamp and compressor"""
        try:
            logger.info(f"Analyzing anomaly for {compressor_id} at {timestamp}")
            
            # Update the system context with current data
            self.load_data_up_to(timestamp)
            
            # Get only current anomalies
            anomalies = self.get_anomaly_data(timestamp, compressor_id)
            if not anomalies:
                return "Nessun dato di anomalia trovato per il timestamp specificato."
                
            # Get minimal weather data
            weather_data = self.get_weather_data(timestamp)
            weather_info = "N/A"
            if weather_data:
                weather_info = f"T:{weather_data.get('avg_temp', 'N/A')}°C, H:{weather_data.get('humidity', 'N/A')}%"
            
            # Format anomalies with detailed percentages and severity
            anomalies_text = []
            for a in anomalies:
                if a['is_anomaly']:
                    deviation = ((a['true_value'] - a['predicted_value'])/a['predicted_value'] * 100)
                    severity = "ALTA" if abs(deviation) > 50 else "MEDIA" if abs(deviation) > 20 else "BASSA"
                    anomalies_text.append(
                        f"{a['parameter']}: "
                        f"attuale {a['true_value']:.2f} vs previsto {a['predicted_value']:.2f} "
                        f"({'+' if deviation > 0 else ''}{deviation:.1f}%) "
                        f"- Gravità: {severity}"
                    )
            
            # Get historical context from vector store
            search_text = f"issues with {compressor_id}"
            if anomalies:
                search_text += " " + " " .join([a['parameter'] for a in anomalies if a['is_anomaly']])
            
            past_context = self.vector_store.similarity_search(
                search_text,
                k=2  # Limit to 2 most relevant documents
            )
            past_context_text = "\n".join([doc.page_content for doc in past_context])[:500]  # Limit length
            
            # Format input data
            analysis_input = f"""TIMESTAMP CORRENTE: {timestamp}

            ANOMALIE RILEVATE ({len(anomalies_text)} parametri):
            {chr(10).join(anomalies_text)}
            
            CONDIZIONI ATTUALI:
            {weather_info}
            """

            # Use the direct LLM chain with correctly named parameters
            result = await self.analysis_chain.ainvoke({
                "context": self.system_context + "\n" + past_context_text,
                "input_data": analysis_input
            })
            
            return result["text"]
            
        except Exception as e:
            logger.error(f"Error in analyze_anomaly: {str(e)}")
            logger.exception("Full traceback:")
            return f"Error analyzing anomaly: {str(e)}"

    async def _generate_anomaly_statistics(self, days: int) -> Dict:
        """Generate statistical information about anomalies over a time period"""
        try:
            start_date = self.current_time - timedelta(days=days)
            formatted_start = start_date.strftime('%Y-%m-%d %H:%M:%S')
            formatted_end = self.current_time.strftime('%Y-%m-%d %H:%M:%S')
            
            with sqlite3.connect(self.db_path) as conn:
                # Get all anomalies in time period
                query = """
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
                """
                
                df = pd.read_sql_query(
                    query,
                    conn,
                    params=[formatted_start, formatted_end],
                    parse_dates=['timestamp']
                )
                
                # Calculate statistics
                total_anomalies = len(df)
                unique_params = df['parameter'].nunique()
                days_with_data = len(df['timestamp'].dt.date.unique())
                
                # Format response with correct article
                if days <= 31:
                    period_text = "nell'ultimo mese"
                else:
                    period_text = "nel periodo"
                
                response = (
                    f"## Analisi Anomalie degli ultimi {days} giorni\n\n"
                    f"**Riepilogo**:\n"
                    f"- {period_text} ({formatted_start} - {formatted_end}) "
                    f"sono state rilevate **{total_anomalies} anomalie** nei parametri monitorati.\n"
                    f"- Sono stati interessati **{unique_params} parametri** diversi del compressore.\n"
                )
                
                return {
                    "answer": response,
                    "sources": ["anomalies_database"]
                }
                
        except Exception as e:
            logger.error(f"Error generating anomaly statistics: {str(e)}")
            return {
                "answer": f"Errore nell'analisi delle statistiche: {str(e)}",
                "sources": []
            }

    async def _handle_recent_failures_query(self, days_lookback: int = 30) -> Dict:
        """Handle a query about recent failures"""
        try:
            end_date = self.current_time
            start_date = end_date - timedelta(days=days_lookback)
            
            with sqlite3.connect(self.db_path) as conn:
                query = """
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
                    WHERE date BETWEEN ? AND ?
                    ORDER BY date DESC
                """
                
                failures_df = pd.read_sql_query(
                    query,
                    conn,
                    params=[
                        start_date.strftime('%Y-%m-%d %H:%M:%S'),
                        end_date.strftime('%Y-%m-%d %H:%M:%S')
                    ],
                    parse_dates=['date']
                )
                
                if not failures_df.empty:
                    response_text = f"""## Guasti degli ultimi {days_lookback} giorni:

                    Nel periodo dal {start_date.strftime('%d/%m/%Y')} al {end_date.strftime('%d/%m/%Y')}, 
                    il compressore CSD102 ha registrato {len(failures_df)} guasti:

                    """
                    for _, row in failures_df.iterrows():
                        response_text += f"""
                        • {row['date'].strftime('%d/%m/%Y %H:%M')} - {row['failure_type']}
                          Causa: {row['cause']}
                          Soluzione: {row['solution']}
                          ---
                        """
                    
                else:
                    response_text = f"Non sono stati registrati guasti negli ultimi {days_lookback} giorni."

                return {
                    "answer": response_text,
                    "sources": ["failures_database"]
                }
                
        except Exception as e:
            logger.error(f"Error handling failures query: {str(e)}")
            return {
                "answer": f"Errore nell'analisi dei guasti: {str(e)}",
                "sources": []
            }

    async def chat(self, user_input: str) -> Dict:
        """Process user input and return a response"""
        try:
            # More precise detection of query type
            failure_keywords = [
                "guasto", "guasti", "rottura", "rotture", "malfunzionamento",
                "si è rotto", "non funziona più"
            ]
            anomaly_keywords = [
                "anomalia", "anomalie", "deviazione", "errore", "parametri",
                "valori anomali", "misure fuori norma"
            ]
            general_issue_keywords = [
                "problema", "problemi", "andato storto", "non funziona",
                "cosa succede", "come va", "situazione", "stato"
            ]

            input_lower = user_input.lower()
            is_failure_query = any(keyword in input_lower for keyword in failure_keywords)
            is_anomaly_query = any(keyword in input_lower for keyword in anomaly_keywords)
            is_general_query = (
                any(keyword in input_lower for keyword in general_issue_keywords) or
                "cosa è successo" in input_lower or 
                "come sta" in input_lower
            )

            if is_failure_query and not is_anomaly_query:
                # Only failures - explicit distinction
                response = await self._handle_recent_failures_query()
                if "Non sono stati registrati guasti" in response["answer"]:
                    response["answer"] = (
                        f"{response['answer']}\n\n"
                        f"Nota: Questa risposta riguarda solo i guasti effettivi del compressore. "
                        f"Per informazioni sulle anomalie nei parametri operativi, "
                        f"poni una domanda specifica sulle anomalie."
                    )
                return response
            elif is_anomaly_query and not is_failure_query:
                # Only anomalies - explicit distinction
                response = await self._generate_anomaly_statistics(30)
                response["answer"] += (
                    "\n\nNota: Questa risposta riguarda solo le anomalie nei parametri monitorati. "
                    "Per informazioni sui guasti meccanici effettivi, "
                    "poni una domanda specifica sui guasti."
                )
                return response
            elif is_general_query or (is_failure_query and is_anomaly_query):
                # Complete analysis with clear sections
                failures = await self._handle_recent_failures_query()
                anomalies = await self._generate_anomaly_statistics(30)
                
                combined_response = (
                    f"## Analisi Complessiva dell'ultimo mese\n\n"
                    f"### 1. Guasti e Malfunzionamenti Meccanici\n"
                    f"{failures['answer']}\n\n"
                    f"### 2. Anomalie nei Parametri Operativi\n"
                    f"{anomalies['answer']}"
                )
                
                return {
                    "answer": combined_response,
                    "sources": failures['sources'] + anomalies['sources']
                }

            # Use regular chain for other queries
            result = await self.qa_chain.ainvoke({
                "query": user_input
            })
            
            return {
                "answer": result["result"],
                "sources": ["knowledge_base"]
            }
                
        except Exception as e:
            logger.error(f"Error in chat: {str(e)}")
            return {
                "answer": f"Mi dispiace, si è verificato un errore: {str(e)}",
                "sources": []
            }

    async def ask_question(self, question: str) -> str:
        """Answer a question using the QA chain"""
        try:
            result = await self.qa_chain.ainvoke({"query": question})
            return result["result"]
        except Exception as e:
            logger.error(f"Error in ask_question: {str(e)}")
            return f"Error answering question: {str(e)}"

def main():
    # Initialize the assistant
    assistant = CompressorAssistant()
    
    # Example usage
    current_time = datetime.utcnow()
    logger.info(f"Running example analysis for current time: {current_time}")
    
    # Example anomaly analysis
    analysis = assistant.analyze_anomaly(
        timestamp=current_time,
        compressor_id="CSD102"
    )
    print("\nAnomaly Analysis:")
    print(analysis)
    
    # Example question
    question = "What are the common causes of high current draw in the CSD102 compressor?"
    answer = assistant.ask_question(question)
    print("\nQuestion Response:")
    print(answer)

if __name__ == "__main__":
    main()