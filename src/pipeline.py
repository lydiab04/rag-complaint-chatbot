import pandas as pd
import re
from sklearn.model_selection import train_test_split

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Remove CFPB boilerplate text if present
    text = re.sub(re.escape("i am writing to file a complaint"), "", text)
    # Remove special characters but keep punctuation markers for sentence boundaries
    text = re.sub(r'[^a-zA-Z0-9\s\.\,\?\!]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def process_data(file_path):
    # Load data (assumes CSV or Parquet format)
    df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_parquet(file_path)
    
    # Map target categories to align with instructions
    target_products = ['Credit card', 'Personal loan', 'Savings account', 'Money transfer']
    # Flexible matching for variations in CFPB naming conventions
    df = df[df['product'].str.contains('|'.join(target_products), case=False, na=False)]
    
    # Drop rows without narratives
    df = df.dropna(subset=['consumer_complaint_narrative'])
    df = df[df['consumer_complaint_narrative'].str.strip() != ""]
    
    # Clean narratives
    df['cleaned_narrative'] = df['consumer_complaint_narrative'].apply(clean_text)
    
    # Save full clean data
    df.to_csv("data/processed/filtered_complaints.csv", index=False)
    print(f"Filtered dataset size: {df.shape[0]} rows")
    
    # Task 2: Stratified Sampling (10,000 - 15,000 rows)
    # Group by the 'product' field to ensure equal proportional distribution
    strat_sample, _ = train_test_split(
        df, 
        train_size=12000, 
        stratify=df['product'], 
        random_state=42
    )
    strat_sample.to_csv("data/processed/stratified_sample.csv", index=False)
    print("Stratified sampling completed successfully.")
    return strat_sample