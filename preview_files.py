import pandas as pd
from pathlib import Path
import PyPDF2

def preview_excel(file_path, n_rows=5):
    """
    Preview the first n rows of an Excel file.
    If the file has multiple sheets, preview each sheet.
    """
    print(f"\n=== Preview of {file_path.name} ===")
    try:
        xl = pd.ExcelFile(file_path)
        for sheet_name in xl.sheet_names:
            print(f"\nSheet: {sheet_name}")
            df = pd.read_excel(file_path, sheet_name=sheet_name, nrows=n_rows)
            print(df.head())
    except Exception as e:
        print(f"Error reading {file_path.name}: {str(e)}")

def preview_pdf(file_path, n_pages=1):
    """
    Preview the first n pages of a PDF file.
    """
    print(f"\n=== Preview of {file_path.name} ===")
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for i in range(min(n_pages, len(pdf_reader.pages))):
                text = pdf_reader.pages[i].extract_text()
                print(f"\nFirst {len(text.split())} words of page {i+1}:")
                print(" ".join(text.split()[:50]) + "...")
    except Exception as e:
        print(f"Error reading {file_path.name}: {str(e)}")

def main():
    # Base directory
    base_dir = Path("/teamspace/studios/this_studio")
    
    # Preview files for Compressor 1
    print("\n=== COMPRESSOR 1 ===")
    comp1_files = [
        "compressore1_predictions.xlsx",
        "compressore1_dolmengiornalieri.xlsx",
        "compressore1_dolmenorari.xlsx",
        "CSD102.pdf"
    ]
    for file in comp1_files:
        path = base_dir / "compressore_1" / file
        if file.endswith('.xlsx'):
            preview_excel(path)
        elif file.endswith('.pdf'):
            preview_pdf(path)

    # Preview weather data
    print("\n=== WEATHER DATA ===")
    weather_files = [
        "marzo_2024_56029.xlsx",
        "aprile_2024_56029.xlsx",
        "maggio_2024_56029.xlsx",
        "giugno_2024_56029.xlsx"
    ]
    for file in weather_files:
        preview_excel(base_dir / "weather_data" / file)

    # Preview technical specifications
    print("\n=== TECHNICAL SPECIFICATIONS ===")
    tech_specs = [
        "interferenze_clima.pdf",
        "report_riunione_090125.pdf"
    ]
    for file in tech_specs:
        preview_pdf(base_dir / "specifiche_tecniche" / file)

if __name__ == "__main__":
    main()