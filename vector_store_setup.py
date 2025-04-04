import logging
from pathlib import Path
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
import sqlite3
import pandas as pd
from langchain.docstore.document import Document
import shutil
from datetime import datetime
import json
from typing import List, Dict, Optional, Union, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VectorStoreSetup:
    def __init__(self, base_path="/teamspace/studios/this_studio"):
        self.base_path = Path(base_path)
        self.processed_docs_path = self.base_path / 'processed_data' / 'processed_docs'
        self.vector_store_path = self.base_path / 'processed_data' / 'vector_store'
        self.vector_store_path.mkdir(exist_ok=True)
        self.db_path = self.base_path / 'processed_data' / 'compressor_data.db'
        
        # Initialize the embedding model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        )

    def load_and_split_documents(self):
        """Load documents from processed_docs directory and split them into chunks"""
        logger.info("Loading and splitting documents...")
        
        # Define directories to process
        doc_dirs = [
            self.processed_docs_path,
            self.processed_docs_path / 'materials'
        ]
        
        all_documents = []
        
        # Load text files from directories
        for doc_dir in doc_dirs:
            if doc_dir.exists():
                loader = DirectoryLoader(
                    doc_dir,
                    glob="*.txt",
                    loader_cls=TextLoader,
                    loader_kwargs={"encoding": "utf-8"}
                )
                documents = loader.load()
                all_documents.extend(documents)
                logger.info(f"Loaded {len(documents)} documents from {doc_dir}")
        
        # Load database tables for additional data
        db_documents = self.load_db_documents()
        all_documents.extend(db_documents)
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        chunks = text_splitter.split_documents(all_documents)
        logger.info(f"Created {len(chunks)} chunks from {len(all_documents)} documents")
        
        return chunks

    def load_db_documents(self) -> List[Document]:
        """Extract data from database tables and convert to documents"""
        documents = []
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Load failures data
                failures_df = pd.read_sql_query("""
                    SELECT 
                        date,
                        failure_type,
                        category,
                        frequency,
                        cause,
                        solution,
                        impact,
                        compressor_id
                    FROM failures
                    ORDER BY date DESC
                """, conn)
                
                if not failures_df.empty:
                    # Convert failures to text format
                    failures_text = ""
                    for _, row in failures_df.iterrows():
                        failures_text += f"""
                        Data: {row['date']}
                        Compressore: {row['compressor_id']}
                        Tipo Guasto: {row['failure_type']}
                        Categoria: {row['category']}
                        Frequenza: {row['frequency']}
                        Causa: {row['cause']}
                        Soluzione: {row['solution']}
                        Impatto: {row['impact']}
                        ---\n"""
                    
                    # Create Document object for failures
                    failures_doc = Document(
                        page_content=failures_text,
                        metadata={'source': 'failures_database', 'type': 'failure_records'}
                    )
                    documents.append(failures_doc)
                    logger.info(f"Added {len(failures_df)} failure records to documents")
                    
                # Load maintenance data
                maintenance_df = pd.read_sql_query("""
                    SELECT 
                        date,
                        intervention_type,
                        operating_hours,
                        activities,
                        anomalies,
                        recommendations,
                        compressor_id
                    FROM maintenance
                    ORDER BY date DESC
                """, conn)
                
                if not maintenance_df.empty:
                    # Convert maintenance to text format
                    maintenance_text = ""
                    for _, row in maintenance_df.iterrows():
                        maintenance_text += f"""
                        Data: {row['date']}
                        Compressore: {row['compressor_id']}
                        Tipo Intervento: {row['intervention_type']}
                        Ore di Funzionamento: {row['operating_hours']}
                        Attività: {row['activities']}
                        Anomalie: {row['anomalies']}
                        Raccomandazioni: {row['recommendations']}
                        ---\n"""
                    
                    # Create Document object for maintenance
                    maintenance_doc = Document(
                        page_content=maintenance_text,
                        metadata={'source': 'maintenance_database', 'type': 'maintenance_records'}
                    )
                    documents.append(maintenance_doc)
                    logger.info(f"Added {len(maintenance_df)} maintenance records to documents")
                
                # Load anomalies data (only significant ones)
                anomalies_df = pd.read_sql_query("""
                    SELECT 
                        timestamp,
                        compressor_id,
                        parameter,
                        true_value,
                        predicted_value,
                        error_value
                    FROM anomalies
                    WHERE is_anomaly = 1
                    ORDER BY timestamp DESC
                    LIMIT 1000
                """, conn)
                
                if not anomalies_df.empty:
                    # Group anomalies by timestamp and compressor
                    anomalies_df['date'] = pd.to_datetime(anomalies_df['timestamp']).dt.date
                    grouped = anomalies_df.groupby(['date', 'compressor_id'])
                    
                    # Convert to text format by day
                    for (date, compressor_id), group in grouped:
                        anomaly_text = f"""
                        Data: {date}
                        Compressore: {compressor_id}
                        Anomalie rilevate:\n"""
                        
                        for _, row in group.iterrows():
                            deviation = ((row['true_value'] - row['predicted_value']) / row['predicted_value']) * 100
                            anomaly_text += f"""
                            - Timestamp: {row['timestamp']}
                              Parametro: {row['parameter']}
                              Valore attuale: {row['true_value']:.3f}
                              Valore previsto: {row['predicted_value']:.3f}
                              Deviazione: {deviation:.2f}%
                            """
                        
                        # Create Document for each day's anomalies
                        anomaly_doc = Document(
                            page_content=anomaly_text,
                            metadata={
                                'source': f'anomalies_{date}_{compressor_id}',
                                'type': 'anomaly_records',
                                'date': str(date),
                                'compressor_id': compressor_id
                            }
                        )
                        documents.append(anomaly_doc)
                    
                    logger.info(f"Added anomaly records for {len(grouped)} days to documents")
        
        except Exception as e:
            logger.error(f"Error loading data from database: {str(e)}")
        
        return documents

    def create_vector_store(self, chunks):
        """Create FAISS vector store from document chunks"""
        logger.info("Creating vector store...")
        
        # Create FAISS vector store
        vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )
        
        # Create a timestamped folder to save the index
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        index_path = self.vector_store_path / f"faiss_index_{timestamp}"
        index_path.mkdir(exist_ok=True)
        
        # Save vector store
        vector_store.save_local(
            folder_path=str(index_path),
            index_name="index"
        )
        
        # Create symbolic link to latest version
        latest_path = self.vector_store_path / "faiss_index"
        
        # If it exists and is a symbolic link, remove it first
        if latest_path.exists():
            if latest_path.is_symlink() or latest_path.is_dir():
                # For Windows, remove directory forcefully
                if os.name == 'nt':
                    shutil.rmtree(latest_path)
                else:
                    # For Unix-like systems, remove symlink
                    latest_path.unlink()
        
        # Create symlink or copy directory based on OS
        if os.name == 'nt':
            # Windows doesn't support symlinks easily, so copy the directory
            shutil.copytree(index_path, latest_path)
        else:
            # Unix-like systems support symlinks
            os.symlink(index_path, latest_path)
        
        logger.info(f"Vector store saved to {index_path}")
        logger.info(f"Latest vector store accessible at {latest_path}")
        
        return vector_store, index_path

    def load_existing_vector_store(self):
        """Load an existing vector store"""
        index_path = self.vector_store_path / "faiss_index"
        if not (index_path / "index.faiss").exists():
            raise FileNotFoundError(
                f"Vector store index not found at {index_path}"
            )
        
        logger.info(f"Loading vector store from {index_path}")
        return FAISS.load_local(
            folder_path=str(index_path),
            embeddings=self.embeddings,
            index_name="index",
            allow_dangerous_deserialization=True
        )

    def update_vector_store(self):
        """Update the vector store with new data"""
        try:
            logger.info("Starting vector store update...")
            
            # Load existing vector store to preserve old data
            existing_store = None
            try:
                existing_store = self.load_existing_vector_store()
                logger.info("Successfully loaded existing vector store")
            except FileNotFoundError:
                logger.info("No existing vector store found. Will create a new one.")
            
            # Load new documents and split them
            chunks = self.load_and_split_documents()
            
            if existing_store:
                # Add new chunks to existing store
                existing_store.add_documents(chunks)
                vector_store = existing_store
                logger.info(f"Added {len(chunks)} new chunks to existing vector store")
            else:
                # Create new vector store
                vector_store, _ = self.create_vector_store(chunks)
                logger.info(f"Created new vector store with {len(chunks)} chunks")
            
            # Save metadata about the update
            metadata = {
                "last_updated": datetime.now().isoformat(),
                "num_documents": len(chunks),
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
            }
            
            with open(self.vector_store_path / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            
            logger.info("Vector store update completed successfully")
            return vector_store
            
        except Exception as e:
            logger.error(f"Error during vector store update: {str(e)}")
            raise

    def setup(self):
        """Run the complete vector store setup process"""
        try:
            logger.info("Starting vector store setup...")
            
            # Check if processed documents exist
            if not self.processed_docs_path.exists():
                raise FileNotFoundError(
                    f"Processed documents directory not found at {self.processed_docs_path}. "
                    "Run data_preprocessing.py first."
                )
            
            # Load and split documents
            chunks = self.load_and_split_documents()
            
            # Create and save vector store
            vector_store, _ = self.create_vector_store(chunks)
            
            logger.info("Vector store setup completed successfully")
            return vector_store
            
        except Exception as e:
            logger.error(f"Error during vector store setup: {str(e)}")
            raise

def main():
    setup = VectorStoreSetup()
    
    try:
        # First try to update existing vector store
        try:
            vector_store = setup.update_vector_store()
            logger.info("Vector store updated successfully")
        except FileNotFoundError:
            # If not found, create new vector store
            logger.info("No existing vector store found. Creating new one...")
            vector_store = setup.setup()
            logger.info("Vector store created successfully")
        
    except Exception as e:
        logger.error(f"Failed to setup/update vector store: {str(e)}")
        raise

if __name__ == "__main__":
    main()