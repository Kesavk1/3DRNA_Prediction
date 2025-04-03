# Initialize seed to control randomness
np.random.seed(0)

# Directories and files adjusted for the new competition
DATA_DIR = os.getenv('DATA_DIR', '/kaggle/input/stanford-rna-3d-folding/')
main_files = [
   "train_sequences.csv", 
   "train_labels.csv", 
   "validation_sequences.csv", 
   "validation_labels.csv", 
   "test_sequences.csv",
   "sample_submission.csv"
]

DEFAULT_THRESHOLD = 0.4  # Default threshold after analysis

def optimize_dataframe(df, inplace=False, category_threshold=DEFAULT_THRESHOLD):
   """
   Optimizes the DataFrame to save memory.
   """
   if category_threshold < 0 or category_threshold > 1:
       raise ValueError("category_threshold must be between 0 and 1.")
   
   if not inplace:
       df = df.copy()
   
   for col in df.columns:
       col_type = df[col].dtype
       if np.issubdtype(col_type, np.integer):
           c_min, c_max = df[col].min(), df[col].max()
           if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
               df[col] = df[col].astype(np.int8)
           elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
               df[col] = df[col].astype(np.int16)
           elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
               df[col] = df[col].astype(np.int32)
       elif np.issubdtype(col_type, np.floating):
           # First check if it's not the special value -1.0e+18
           if df[col].min() > np.finfo(np.float32).min and df[col].max() < np.finfo(np.float32).max:
               df[col] = df[col].astype(np.float32)
       if col_type == object:
           unique_vals = len(df[col].unique())
           if unique_vals / len(df) < category_threshold:
               df[col] = df[col].astype('category')
   
   return df

def load_main_data(chunksize=50000):
   """
   Loads the main files.
   """
   data = {}
   for file_name in main_files:
       file_path = os.path.join(DATA_DIR, file_name)
       if os.path.exists(file_path):
           chunks = pd.read_csv(file_path, on_bad_lines='skip', low_memory=False, chunksize=chunksize)
           dataframes = [optimize_dataframe(chunk, category_threshold=DEFAULT_THRESHOLD) for chunk in chunks]
           data[file_name] = pd.concat(dataframes, ignore_index=True)
           print(f"File {file_name} loaded successfully. Shape: {data[file_name].shape}")
       else:
           print(f"File {file_path} not found!")
           data[file_name] = None
   return data

def filter_columns_by_prefix(df, prefix="x_"):
   """
   Filters and counts the number of columns in a DataFrame based on a provided prefix.
   
   :param df: DataFrame where filtering will be applied.
   :param prefix: Prefix to be used for filtering. Ex: "x_", "y_", "z_".
   :return: List of filtered columns.
   """
   filtered_columns = [col for col in df.columns if col.startswith(prefix)]
   return filtered_columns

def count_nucleotides(df, column_name='sequence'):
   """
   Counts the frequency of each nucleotide in a specific column of a DataFrame.
   
   :param df: DataFrame containing the sequences.
   :param column_name: Name of the column containing the sequences. Default is 'sequence'.
   :return: Counter object with the nucleotide counts.
   """
   from collections import Counter

   # Check if the column exists in the DataFrame
   if column_name not in df.columns:
       raise ValueError(f"Column '{column_name}' not found in DataFrame.")
   
   # Concatenate all sequences and count nucleotides
   all_sequences = ''.join(df[column_name].tolist())
   nucleotide_counts = Counter(all_sequences)
   
   return nucleotide_counts

def get_columns_without_missing_values(df):
   """
   Returns columns without any missing values in the DataFrame.
   
   :param df: DataFrame to be checked.
   :return: List of columns without missing values.
   """
   missing_values = df.isnull().sum()
   return missing_values[missing_values == 0].index.tolist()

def get_empty_columns(df):
   """
   Returns columns that are completely empty in the DataFrame.
   
   :param df: DataFrame to be checked.
   :return: List of empty columns.
   """
   missing_values = df.isnull().sum()
   return missing_values[missing_values == df.shape[0]].index.tolist()

def plot_coord_distributions(df_labels, prefix='x_', max_structures=5):
   """
   Plots the distribution of coordinates (x, y, or z) for up to max_structures structures.
   
   :param df_labels: DataFrame containing the coordinates.
   :param prefix: Prefix of columns to be plotted ('x_', 'y_', or 'z_').
   :param max_structures: Maximum number of structures to show.
   """
   # Find coordinate columns with the specified prefix
   coord_cols = filter_columns_by_prefix(df_labels, prefix)
   
   # Limit to the maximum number of structures
   coord_cols = sorted(coord_cols)[:max_structures]
   
   if not coord_cols:
       print(f"No column with prefix '{prefix}' found.")
       return
   
   # Set up the plot
   fig, axes = plt.subplots(1, len(coord_cols), figsize=(16, 4))
   if len(coord_cols) == 1:
       axes = [axes]  # Ensure axes is iterable even with a single subplot
   
   # Plot histograms for each column
   for i, col in enumerate(coord_cols):
       # Filter special values (-1.0e+18) if present
       values = df_labels[col]
       filtered_values = values[values > -1.0e+17]  # Cutoff value to filter -1.0e+18
       
       axes[i].hist(filtered_values, bins=30, alpha=0.7)
       axes[i].set_title(f'Distribution of {col}')
       axes[i].set_xlabel('Value')
       axes[i].set_ylabel('Frequency')
   
   plt.tight_layout()
   plt.show()

def analyze_3d_structure(df_labels):
   """
   Analyzes the 3D coordinates of RNA structures.
   
   :param df_labels: DataFrame containing 3D coordinates.
   """
   # Find all coordinate columns
   x_cols = filter_columns_by_prefix(df_labels, 'x_')
   y_cols = filter_columns_by_prefix(df_labels, 'y_')
   z_cols = filter_columns_by_prefix(df_labels, 'z_')
   
   print(f"Number of x columns: {len(x_cols)}")
   print(f"Number of y columns: {len(y_cols)}")
   print(f"Number of z columns: {len(z_cols)}")
   
   # Check for missing or special values in coordinates
   special_value = -1.0e+18  # Special value observed in the data
   
   for i, (x_col, y_col, z_col) in enumerate(zip(x_cols, y_cols, z_cols), 1):
       # Count missing or special values
       x_special = (df_labels[x_col] == special_value).sum()
       y_special = (df_labels[y_col] == special_value).sum()
       z_special = (df_labels[z_col] == special_value).sum()
       
       x_null = df_labels[x_col].isnull().sum()
       y_null = df_labels[y_col].isnull().sum()
       z_null = df_labels[z_col].isnull().sum()
       
       # Count how many complete structures exist (all x, y, z are neither special nor null)
       valid_structures = ((df_labels[x_col] != special_value) & 
                          (df_labels[y_col] != special_value) & 
                          (df_labels[z_col] != special_value) &
                          df_labels[x_col].notnull() & 
                          df_labels[y_col].notnull() & 
                          df_labels[z_col].notnull()).sum()
       
       total_rows = len(df_labels)
       
       print(f"\nStructure {i}:")
       print(f"  Special values: x={x_special} ({x_special/total_rows*100:.2f}%), y={y_special} ({y_special/total_rows*100:.2f}%), z={z_special} ({z_special/total_rows*100:.2f}%)")
       print(f"  Null values: x={x_null} ({x_null/total_rows*100:.2f}%), y={y_null} ({y_null/total_rows*100:.2f}%), z={z_null} ({z_null/total_rows*100:.2f}%)")
       print(f"  Complete structures: {valid_structures} ({valid_structures/total_rows*100:.2f}%)")
       
       # Limit analysis to the first 5 structures
       if i >= 5:
           print("\nAnalysis limited to the first 5 structures.")
           break

def analyze_sequences(df_sequences):
   """
   Analyzes RNA sequences.
   
   :param df_sequences: DataFrame containing the 'sequence' column.
   """
   # Basic statistics of the sequence column
   print("\nBasic statistics of the 'sequence' column:")
   print(df_sequences['sequence'].describe())
   
   # Sequence lengths
   seq_lengths = df_sequences['sequence'].apply(len)
   print("\nSequence length statistics:")
   print(f"Minimum: {seq_lengths.min()}")
   print(f"Maximum: {seq_lengths.max()}")
   print(f"Mean: {seq_lengths.mean():.2f}")
   print(f"Median: {seq_lengths.median()}")
   
   # Nucleotide counts
   nucleotide_counts = count_nucleotides(df_sequences)
   total_nucleotides = sum(nucleotide_counts.values())
   
   print("\nNucleotide distribution:")
   for nucleotide, count in sorted(nucleotide_counts.items()):
       print(f"{nucleotide}: {count} ({count/total_nucleotides*100:.2f}%)")
   
   # Plot length distribution
   plt.figure(figsize=(10, 6))
   plt.hist(seq_lengths, bins=30, alpha=0.7)
   plt.title('Sequence Length Distribution')
   plt.xlabel('Length')
   plt.ylabel('Frequency')
   plt.grid(True, alpha=0.3)
   plt.show()

def main():
   # Load main data
   main_data = load_main_data()

   # Check which files were loaded
   print("\nLoaded files:")
   for file_name, df in main_data.items():
       if df is not None:
           print(f"- {file_name}: {df.shape}")
   
   # Analyze 3D structures in validation_labels.csv
   if "validation_labels.csv" in main_data and main_data["validation_labels.csv"] is not None:
       print("\n===== Analysis of 3D Structures (validation_labels.csv) =====")
       df_labels = main_data["validation_labels.csv"]
       
       # Count coordinate columns
       x_cols = filter_columns_by_prefix(df_labels, 'x_')
       y_cols = filter_columns_by_prefix(df_labels, 'y_')
       z_cols = filter_columns_by_prefix(df_labels, 'z_')
       
       print(f"There are {len(x_cols)} x_ columns in the DataFrame.")
       print(f"There are {len(y_cols)} y_ columns in the DataFrame.")
       print(f"There are {len(z_cols)} z_ columns in the DataFrame.")
       
       # Identify columns without missing values
       columns_without_missing = get_columns_without_missing_values(df_labels)
       print(f"\nColumns without missing values: {len(columns_without_missing)}")
       
       # Identify completely empty columns
       empty_columns = get_empty_columns(df_labels)
       print(f"Completely empty columns: {len(empty_columns)}")
       
       # Analyze 3D coordinates in detail
       analyze_3d_structure(df_labels)
       
       # Plot distribution of x, y, z coordinates for the first structures
       print("\nDistribution of X coordinates:")
       plot_coord_distributions(df_labels, 'x_', max_structures=3)
       print("\nDistribution of Y coordinates:")
       plot_coord_distributions(df_labels, 'y_', max_structures=3)
       print("\nDistribution of Z coordinates:")
       plot_coord_distributions(df_labels, 'z_', max_structures=3)
   
   # Analyze sequences in train_sequences.csv
   if "train_sequences.csv" in main_data and main_data["train_sequences.csv"] is not None:
       print("\n===== Analysis of Sequences (train_sequences.csv) =====")
       df_sequences = main_data["train_sequences.csv"]
       
       # First few rows of the sequence column
       print("\nFirst few rows of the 'sequence' column:")
       print(df_sequences['sequence'].head())
       
       # Data type of the sequence column
       print("\nData type of the 'sequence' column:")
       print(df_sequences['sequence'].dtype)
       
       # Complete sequence analysis
       analyze_sequences(df_sequences)
   
   return main_data

if __name__ == '__main__':
   main_data = main()
