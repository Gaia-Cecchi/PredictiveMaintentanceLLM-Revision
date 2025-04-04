from data_preprocessing import DataPreprocessor
from vector_store_setup import VectorStoreSetup
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def reset_and_setup():
    """Reset database and rebuild vector store"""
    try:
        # Reset and rebuild database
        logger.info("Resetting and rebuilding database...")
        preprocessor = DataPreprocessor()
        preprocessor.process_all()
        
        # Setup vector store
        logger.info("Setting up vector store...")
        setup = VectorStoreSetup()
        setup.setup()
        
        logger.info("Reset and setup completed successfully")
        
    except Exception as e:
        logger.error(f"Error during reset and setup: {str(e)}")
        raise

if __name__ == "__main__":
    reset_and_setup()
