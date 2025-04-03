def visualize_rna_heatmap_from_processed_data(processed_data, num_samples=12):
    """
    Visualizes a heatmap for RNA sequences using processed data.
    
    Parameters:
    processed_data: Dictionary with processed data returned by the main() function
    num_samples: Number of sequences to visualize
    """
    try:
        # Check if we have the necessary data
        if 'X_valid' not in processed_data or processed_data['X_valid'] is None:
            print("Validation data not found in processed_data object")
            return None
        
        # Get the data
        X_valid = processed_data['X_valid']
        print(f"Data found with format: {X_valid.shape}")
        
        # Limit to the number of samples
        X_valid_subset = X_valid[:num_samples]
        
        # If we have IDs, use them
        if 'valid_ids' in processed_data and processed_data['valid_ids']:
            valid_ids = processed_data['valid_ids'][:num_samples]
        else:
            valid_ids = [f"Seq_{i+1}" for i in range(X_valid_subset.shape[0])]
        
        # Convert one-hot encoding to nucleotide indices
        # Expected format: A=[1,0,0,0,0], C=[0,1,0,0,0], G=[0,0,1,0,0], U=[0,0,0,1,0], N=[0,0,0,0,1]
        sequences_matrix = np.argmax(X_valid_subset, axis=2)
        
        # Replace zeros (padding) with 4 (N/Unknown) when all values are zero
        is_padding = np.all(X_valid_subset == 0, axis=2)
        sequences_matrix[is_padding] = 4
        
        # Define a categorical colormap (distinct colors per nucleotide)
        cmap = mcolors.ListedColormap(['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#95a5a6'])
        bounds = [0, 1, 2, 3, 4, 5]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        
        # Create figure
        plt.figure(figsize=(20, 10))
        im = plt.imshow(sequences_matrix, cmap=cmap, norm=norm, aspect='auto')
        
        # Add color bar
        cbar = plt.colorbar(im, ticks=[0.5, 1.5, 2.5, 3.5, 4.5])
        cbar.set_label('Nucleotides', fontsize=14)
        cbar.set_ticklabels(['A', 'C', 'G', 'U', 'N/Padding'])
        
        # Add axis labels
        plt.xlabel("Position in Sequence", fontsize=14)
        plt.ylabel("RNA Sequences", fontsize=14)
        
        # Add title
        plt.title("RNA Sequences Heatmap", fontsize=16)
        
        # Add sequence IDs as y-axis labels
        plt.yticks(range(len(valid_ids)), valid_ids, fontsize=10)
        
        # Show only some labels on x-axis to avoid crowding
        sequence_length = sequences_matrix.shape[1]
        step = max(1, sequence_length // 20)  # Show at most 20 labels
        plt.xticks(range(0, sequence_length, step), range(1, sequence_length + 1, step))
        
        # Add grid
        plt.grid(False)
        
        # Add information about nucleotide distribution
        all_nucleotides = sequences_matrix.flatten()
        nucleotide_counts = {
            'A': np.sum(all_nucleotides == 0),
            'C': np.sum(all_nucleotides == 1),
            'G': np.sum(all_nucleotides == 2),
            'U': np.sum(all_nucleotides == 3),
            'N': np.sum(all_nucleotides == 4)
        }
        
        total_nucleotides = sum(nucleotide_counts.values())
        nucleotide_percentages = {k: (v / total_nucleotides) * 100 for k, v in nucleotide_counts.items()}
        
        # Add text with statistics
        info_text = "\n".join([
            f"Total sequences visualized: {num_samples}",
            f"Maximum length: {sequence_length}",
            f"A: {nucleotide_percentages['A']:.1f}%",
            f"C: {nucleotide_percentages['C']:.1f}%",
            f"G: {nucleotide_percentages['G']:.1f}%",
            f"U: {nucleotide_percentages['U']:.1f}%",
            f"N/Padding: {nucleotide_percentages['N']:.1f}%"
        ])
        
        plt.figtext(0.02, 0.02, info_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
        
        # Show the plot
        plt.tight_layout()
        plt.show()
        
        # Optionally, save the plot
        output_dir = '/kaggle/working/'
        plt.savefig(os.path.join(output_dir, 'rna_heatmap.png'), dpi=300)
        print(f"Heatmap saved to {os.path.join(output_dir, 'rna_heatmap.png')}")
        
        return sequences_matrix
    except Exception as e:
        print(f"Error processing data: {e}")
        return None

# Use the function (assuming processed_data is available)
visualize_rna_heatmap_from_processed_data(processed_data)
