# Weather Data Collection and Processing

This directory contains scripts and data for collecting and processing weather data from Weather Underground.

## Files Overview

- `scraping_weather_data/` - Directory containing web scraping scripts
- `weather_data.csv` - Main CSV file with raw weather data
- `weather_data.xlsx` - Excel version of the raw weather data
- `halfhourly_weather_data_supabase.csv` - Version optimized for Supabase with "celsius degrees" terminology
- `weather_data_db_optimized.csv` - Version optimized for database/LLM usage
- `weather_data_metadata.json` - Metadata describing the optimized data format

## Database-Optimized Version

The `weather_data_db_optimized.csv` file has been specifically designed for:

1. **Database Storage Efficiency**: Clean, consistent column names and appropriate data types
2. **LLM Readability**: Well-structured data with logical organization
3. **Analysis Ready**: Numeric values extracted from text, categorized conditions
4. **Time Intelligence**: Date components extracted into separate fields for easier querying

### Features

- ISO 8601 timestamps (`YYYY-MM-DDTHH:MM:SS`)
- Numeric values without units (e.g., `23.5` instead of `23.5°C`)
- Extracted date components (year, month, day, hour, minute)
- Wind direction in both cardinal abbreviations and degrees
- Categorized weather conditions
- Comprehensive metadata in a separate JSON file

## Using the Data with LLMs

When using this data with LLMs, refer to the `weather_data_metadata.json` file which provides:

- Data source information
- Location coordinates
- Detailed column descriptions
- Value ranges and units
- Schema version

This metadata helps LLMs understand the structure and meaning of the data, enabling more accurate analysis and responses.

## Data Processing Steps

1. **Collection**: Web scraping from Weather Underground historical data
2. **Cleaning**: Removal of special characters, standardizing formats
3. **Transformation**: Extracting numeric values, categorizing conditions
4. **Enrichment**: Adding computed fields like weekend indicators
5. **Standardization**: Ensuring consistent formats and units
6. **Validation**: Removing invalid or sparse records
