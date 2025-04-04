import logging
import pandas as pd
from pathlib import Path
import os
import sqlite3
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import traceback

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def reindex_workload_data():
    """Reindexa i dati di carico operativo nel vector store"""
    try:
        base_path = Path("/teamspace/studios/this_studio")
        processed_data_path = base_path / 'processed_data'
        vector_store_path = processed_data_path / 'vector_store' / 'faiss_index'
        workload_path = base_path / 'materiali' / 'carico_operativo.csv'
        db_path = processed_data_path / 'compressor_data.db'
        
        # Verificare che il file esiste
        if not workload_path.exists():
            logger.error(f"File carico operativo non trovato in: {workload_path}")
            return False
        
        # Inizializza embedding model
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Carica vector store esistente
        if not vector_store_path.exists():
            logger.error(f"Vector store non trovato in: {vector_store_path}")
            return False
        
        vector_store = FAISS.load_local(
            str(vector_store_path), 
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        # Caricare i dati da CSV
        df = pd.read_csv(workload_path)
        logger.info(f"Caricati {len(df)} record dal file carico_operativo.csv")
        
        # Anche caricarli dal database per verificare l'esistenza della tabella
        try:
            with sqlite3.connect(db_path) as conn:
                # Verificare se la tabella esiste
                result = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='operating_conditions'").fetchone()
                if not result:
                    logger.warning("Tabella operating_conditions non trovata nel database")
                else:
                    db_df = pd.read_sql("SELECT * FROM operating_conditions", conn)
                    logger.info(f"Trovati {len(db_df)} record nella tabella operating_conditions")
                    
                    # Se mancano dati nel database, inserirli
                    if len(db_df) < len(df):
                        logger.warning(f"Trovati {len(df) - len(db_df)} record mancanti nel database")
                        
                        # Se la tabella esiste ma è vuota, inserire tutti i dati
                        if len(db_df) == 0:
                            # Rinomina colonne se necessario
                            mapping = {
                                'Data': 'date',
                                'Ora_Inizio': 'start_time',
                                'Ora_Fine': 'end_time',
                                'Ore_Funzionamento': 'operating_hours',
                                'Carico_Operativo': 'load_percentage',
                                'Temperatura': 'temperature',
                                'Umidita': 'humidity',
                                'Vibrazioni': 'vibration',
                                'Pressione': 'pressure'
                            }
                            
                            # Rinomina le colonne solo se esistono
                            renamed_df = df.copy()
                            for old_col, new_col in mapping.items():
                                if old_col in renamed_df.columns:
                                    renamed_df = renamed_df.rename(columns={old_col: new_col})
                            
                            # Assicurati che compressor_id sia presente
                            if 'compressor_id' not in renamed_df.columns:
                                renamed_df['compressor_id'] = 'CSD102'
                            
                            # Salva nel database
                            renamed_df.to_sql('operating_conditions', conn, if_exists='replace', index=False)
                            logger.info(f"Inseriti {len(renamed_df)} record nella tabella operating_conditions")
        except Exception as e:
            logger.error(f"Errore nell'accesso al database: {str(e)}")
            logger.error(traceback.format_exc())
        
        # Crea documenti per vector store con tipo specifico e formato ottimizzato
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        docs = []
        for _, row in df.iterrows():
            content = f"""
            CARICO OPERATIVO DEL COMPRESSORE
            ---
            Data: {row['Data']}
            Ore di funzionamento: {row['Ore_Funzionamento']}
            Carico operativo: {row['Carico_Operativo']}%
            Temperatura: {row['Temperatura']}°C
            Umidità: {row['Umidita']}%
            Vibrazioni: {row['Vibrazioni']} mm/s
            Pressione: {row['Pressione']} bar
            Intervallo orario: {row['Ora_Inizio']} - {row['Ora_Fine']}
            
            Questo documento contiene informazioni sul carico operativo del compressore CSD102.
            I dati mostrano i livelli di carico, temperatura, umidità e altri parametri operativi.
            """
            
            # Aggiungi esplicitamente parole chiave per migliorare il retrieval
            keywords = "\nParole chiave: carico operativo, carico di lavoro, ore di funzionamento, temperatura, pressione, compressore"
            content += keywords
            
            doc = Document(
                page_content=content.strip(),
                metadata={
                    "source": f"carico_operativo_{row['Data']}",
                    "type": "workload",  # Assegna un tipo specifico
                    "date": row['Data'],
                    "title": "Dati Carico Operativo",
                    "compressor_id": row.get('compressor_id', 'CSD102'),
                    "keywords": "carico operativo, carico di lavoro"
                }
            )
            docs.append(doc)
        
        logger.info(f"Creati {len(docs)} documenti per il vector store")
        
        # Dividi i documenti se necessario e aggiungi al vector store
        chunks = text_splitter.split_documents(docs)
        logger.info(f"Creati {len(chunks)} chunks dai documenti")
        
        # Aggiungi i documenti al vector store
        vector_store.add_documents(chunks)
        vector_store.save_local(str(vector_store_path))
        logger.info(f"Vector store aggiornato e salvato in {vector_store_path}")
        
        # Test di query sul vector store
        test_queries = [
            "carico operativo recente", 
            "quali carichi di lavoro ci sono stati",
            "dati recenti sul carico operativo",
            "ore di funzionamento del compressore"
        ]
        
        logger.info("\nTest delle query:")
        for query in test_queries:
            docs = vector_store.similarity_search(query, k=2)
            logger.info(f"\nQuery: '{query}'")
            logger.info(f"Trovati {len(docs)} documenti")
            for doc in docs:
                logger.info(f"Tipo: {doc.metadata.get('type')}")
                logger.info(f"Fonte: {doc.metadata.get('source')}")
                logger.info(f"Contenuto: {doc.page_content[:100]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"Errore nella reindicizzazione dei dati: {str(e)}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    print("Reindicizzazione dei dati di carico operativo...")
    success = reindex_workload_data()
    if success:
        print("\n✅ Reindicizzazione completata con successo")
        print("I dati di carico operativo sono ora disponibili nel vector store")
        print("Prova a fare queste domande all'assistente:")
        print("- Quali sono i carichi di lavoro recenti?")
        print("- Qual è stato il carico operativo nelle ultime settimane?")
        print("- Ci sono stati giorni con carichi operativi particolarmente elevati?")
    else:
        print("\n❌ Errore nella reindicizzazione")
        print("Controlla i log per maggiori dettagli")
