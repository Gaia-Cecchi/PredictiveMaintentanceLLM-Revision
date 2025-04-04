import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
DATA_BASE_PATH = Path(os.getenv('DATA_BASE_PATH'))
COMPRESSOR_1_PATH = Path(os.getenv('COMPRESSOR_1_PATH'))
COMPRESSOR_2_PATH = Path(os.getenv('COMPRESSOR_2_PATH'))
COMPRESSOR_3_PATH = Path(os.getenv('COMPRESSOR_3_PATH'))
COMPRESSOR_4_PATH = Path(os.getenv('COMPRESSOR_4_PATH'))
WEATHER_DATA_PATH = Path(os.getenv('WEATHER_DATA_PATH'))
TECH_SPECS_PATH = Path(os.getenv('TECH_SPECS_PATH'))

# File paths for compressor 1
COMPRESSOR_1_FILES = {
    'predictions': COMPRESSOR_1_PATH / 'compressore1_predictions.xlsx',
    'daily_data': COMPRESSOR_1_PATH / 'compressore1_dolmengiornalieri.xlsx',
    'hourly_data': COMPRESSOR_1_PATH / 'compressore1_dolmenorari.xlsx',
    'manual': COMPRESSOR_1_PATH / 'CSD102.pdf'
}

# File paths for compressor 2
COMPRESSOR_2_FILES = {
    'manual': COMPRESSOR_2_PATH / 'BSD72.pdf',
    'daily_data': COMPRESSOR_2_PATH / 'compressore2_dolmengiornalieri.xlsx',
    'hourly_data': COMPRESSOR_2_PATH / 'compressore2_dolmenorari.xlsx'
}

# File paths for compressor 3
COMPRESSOR_3_FILES = {
    'manual': COMPRESSOR_3_PATH / 'BS61.pdf',
    'daily_data': COMPRESSOR_3_PATH / 'compressore3_dolmengiornalieri.xlsx',
    'hourly_data': COMPRESSOR_3_PATH / 'compressore3_dolmenorari.xlsx'
}

# File paths for compressor 4
COMPRESSOR_4_FILES = {
    'manual': COMPRESSOR_4_PATH / 'SK21.pdf',
    'daily_data': COMPRESSOR_4_PATH / 'compressore4_dolmengiornalieri.xlsx',
    'hourly_data': COMPRESSOR_4_PATH / 'compressore4_dolmenorari.xlsx'
}

# Weather data files
WEATHER_FILES = {
    'march': WEATHER_DATA_PATH / 'marzo_2024_56029.xlsx',
    'april': WEATHER_DATA_PATH / 'aprile_2024_56029.xlsx',
    'may': WEATHER_DATA_PATH / 'maggio_2024_56029.xlsx',
    'june': WEATHER_DATA_PATH / 'giugno_2024_56029.xlsx'
}

# Technical specifications
TECH_SPECS_FILES = {
    'climate_interference': TECH_SPECS_PATH / 'interferenze_clima.pdf',
    'meeting_report': TECH_SPECS_PATH / 'report_riunione_090125.pdf'
}

# Materials files
MATERIALS_PATH = DATA_BASE_PATH / 'materiali'
MATERIALS_FILES = {
    'guasti': MATERIALS_PATH / 'guasti.csv',
    'manutenzioni': MATERIALS_PATH / 'manutenzioni.csv',
    'feedback': MATERIALS_PATH / 'feedback.csv',
    'carico_operativo': MATERIALS_PATH / 'carico_operativo.csv',
    'politiche': MATERIALS_PATH / 'politiche_aziendali.txt'
}

# LLM Configuration
LLM_MODEL = "llama-3.3-70b-versatile"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Database configuration
SQLITE_DB_PATH = DATA_BASE_PATH / "compressor_data.db"
VECTOR_STORE_PATH = DATA_BASE_PATH / "vectorstore.faiss"

# Monitoring parameters
PARAMETERS = [
    'Current (A)',
    'CosPhi (Units)',
    'Energy Consumption (kWh)',
    'Reactive Energy (VARh)',
    'Voltage (V)'
]

# Weather parameters
WEATHER_COLUMNS = [
    'LOCALITA', 'DATA', 'TMEDIA °C', 'TMIN °C', 'TMAX °C',
    'PUNTORUGIADA °C', 'UMIDITA %', 'VISIBILITA km',
    'VENTOMEDIA km/h', 'VENTOMAX km/h', 'RAFFICA km/h',
    'PRESSIONESLM mb', 'PRESSIONEMEDIA mb', 'PIOGGIA mm',
    'FENOMENI'
]