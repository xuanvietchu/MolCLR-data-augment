import os
import random
import pandas as pd

# Function to sample 0.2% of the content
def sample_data(file_path, sample_percentage=0.2):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    sample_size = int(len(lines) * sample_percentage / 100)    
    sampled_lines = random.sample(lines, sample_size)
    
    return sampled_lines


if __name__ == '__main__':
    # random seed
    random.seed(42)
    
    pubchem_file = 'pubchem-10m-clean.txt'
    data_folder = 'data'

    pubchem_file = os.path.join('data', pubchem_file)

    sampled_pubchem_data = sample_data(pubchem_file, 0.2*10)

    # Save the sampled data
    sampled_pubchem_file = 'pubchem-200k-sample.txt'
    sampled_pubchem_file = os.path.join('data', sampled_pubchem_file)
    with open(sampled_pubchem_file, 'w') as file:
        file.writelines(sampled_pubchem_data)
    

