import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def load_data(data_dir):
    """
    Loads train_source1, train_source2, train_source3, and train_ground_truth from TSV files.
    """
    try:
        source1_path = os.path.join(data_dir, 'train_source1.tsv')
        source2_path = os.path.join(data_dir, 'train_source2.tsv')
        source3_path = os.path.join(data_dir, 'train_source3.tsv')
        ground_truth_path = os.path.join(data_dir, 'train_ground_truth.tsv')

        # CRITICAL: Use sep='\t' as commas exist within the data fields
        train_source1 = pd.read_csv(source1_path, sep='\t')
        train_source2 = pd.read_csv(source2_path, sep='\t')
        train_source3 = pd.read_csv(source3_path, sep='\t')
        train_ground_truth = pd.read_csv(ground_truth_path, sep='\t')

        print("Successfully loaded datasets.")
        return train_source1, train_source2, train_source3, train_ground_truth
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return None, None, None, None

def create_validation_split(train_source1, output_dir, test_size=0.2, random_state=42):
    """
    Creates a 20% local validation split based on train_source1.tsv and saves it locally.
    """
    if train_source1 is None:
        print("Error: train_source1 is None. Cannot create validation split.")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Create the 20% validation split
    train_split, val_split = train_test_split(train_source1, test_size=test_size, random_state=random_state)
    
    train_split_path = os.path.join(output_dir, 'train_source1_train_split.tsv')
    val_split_path = os.path.join(output_dir, 'train_source1_val_split.tsv')
    
    # Save splits locally
    train_split.to_csv(train_split_path, sep='\t', index=False)
    val_split.to_csv(val_split_path, sep='\t', index=False)
    
    print(f"Successfully created validation split.")
    print(f"Training split saved to {train_split_path} (shape: {train_split.shape})")
    print(f"Validation split saved to {val_split_path} (shape: {val_split.shape})")
    
    return train_split, val_split

if __name__ == "__main__":
    # Define paths
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
    data_dir = os.path.join(base_dir, 'dataset', 'train')
    output_dir = os.path.join(base_dir, 'output')
    
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load data
    # Note: If the files do not exist yet, this will print an error, which is expected.
    # To test logic, make sure the files are present in the dataset/train folder.
    src1, src2, src3, gt = load_data(data_dir)
    
    # Create validation split if source1 is loaded
    if src1 is not None:
        create_validation_split(src1, output_dir)
