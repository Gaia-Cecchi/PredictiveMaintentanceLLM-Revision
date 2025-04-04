# File contenente tutti gli stili CSS usati nell'applicazione

def get_dashboard_styles():
    """Stile delle card e dei pulsanti nella dashboard di monitoraggio"""
    return """
        <style>
        .compressor-card {
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid #ddd;
            margin-bottom: 1rem;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .compressor-card:hover {
            border-color: #0066cc;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .card-header {
            font-size: 1.2rem;
            font-weight: 500;
            margin-bottom: 0.5rem;
        }
        .card-status {
            color: #666;
            margin-top: 0.5rem;
        }
        .anomaly-details {
            margin-top: 1rem;
            padding-top: 1rem;
            border-top: 1px solid #eee;
        }
        .parameter-list {
            margin: 0.5rem 0;
            padding-left: 1rem;
        }
        .analyze-button {
            margin-top: 1rem;
        }
        
        .stButton button {
            text-align: left !important;
            justify-content: flex-start !important;
            padding: 0.9rem 0.9rem 0.9rem 0.9rem;
            margin: 1rem 0 0 0 !important;
        }
        .stButton button p {
            text-align: left !important;
            width: 100% !important;
        }
        </style>
    """

def get_weather_card_styles():
    """Stile della card meteo"""
    return """
        <style>
        /* Rimuovi spazi extra dai container di Streamlit */
        [data-testid="column"] > div:has(> .weather-card) {
            padding: 0 !important;
            margin: 0 !important;
        }
        
        [data-testid="column"] > div {
            padding-top: 0 !important;
            margin-top: 0 !important;
        }
        
        /* Stile della weather card */
        .weather-card {
            padding: 1.2rem;
            border: 1px solid rgb(49, 51, 63);
            border-radius: 0.5rem;
            background-color: transparent;
            margin: 0;
            width: 100%;
            white-space: pre-wrap;
            text-align: left !important;
            display: block;
            line-height: 1.6;
            font-size: 16px;
        }

        .weather-title {
            font-size: 18px;
            font-weight: bold;
            display: block;
        }
        </style>
    """

def get_chat_styles():
    """Stile della chat nella sidebar"""
    return """
        <style>
        .chat-message {
            margin-bottom: 10px;
            padding: 10px;
            border-radius: 6px;
            color: white;
        }
        .user-message {
            background-color: #2a4865;
            border-left: 4px solid #0078ff;
        }
        .assistant-message {
            background-color: #353b48;
            border-left: 4px solid #8e44ad;
        }
        .source-label {
            margin-top: 5px;
            font-size: 12px;
            color: #cccccc;
        }
        </style>
    """
