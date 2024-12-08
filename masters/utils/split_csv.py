import pandas as pd
import numpy as np

from tqdm import tqdm

def split_csv_in_batches(input_file, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1, batch_size=10000, output_prefix='output'):
    chunks = pd.read_csv(input_file, chunksize=batch_size)
    
    train_data = []
    valid_data = []
    test_data = []
    
    for chunk in tqdm(chunks):
        chunk = chunk.sample(frac=1, random_state=42).reset_index(drop=True)
        
        total_rows = len(chunk)
        train_size = int(train_ratio * total_rows)
        valid_size = int(valid_ratio * total_rows)
        test_size = total_rows - train_size - valid_size
        
        train_chunk = chunk[:train_size]
        valid_chunk = chunk[train_size:train_size+valid_size]
        test_chunk = chunk[train_size+valid_size:]
        
        train_data.append(train_chunk)
        valid_data.append(valid_chunk)
        test_data.append(test_chunk)
    
    train_df = pd.concat(train_data)
    valid_df = pd.concat(valid_data)
    test_df = pd.concat(test_data)
    
    train_df.to_csv(f'{output_prefix}_train.csv', index=False)
    valid_df.to_csv(f'{output_prefix}_valid.csv', index=False)
    test_df.to_csv(f'{output_prefix}_test.csv', index=False)

if __name__ == "__main__":
    split_csv_in_batches(r'C:\Users\PanSt\master_thesis\masters\datasets\diorisis.csv', batch_size=10000, output_prefix='diorisis')
