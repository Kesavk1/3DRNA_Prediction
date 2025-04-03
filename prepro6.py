ef prepare_test_features(test_seq_df, max_length=720):
    """
    Prepares test features (one-hot encoding of the sequence).
    """
    X_test = []
    for _, row in test_seq_df.iterrows():
        seq = row['sequence']
        features = []
        for nucleotide in seq:
            if nucleotide == 'A':
                features.append([1, 0, 0, 0, 0])
            elif nucleotide == 'C':
                features.append([0, 1, 0, 0, 0])
            elif nucleotide == 'G':
                features.append([0, 0, 1, 0, 0])
            elif nucleotide == 'U':
                features.append([0, 0, 0, 1, 0])
            else:
                features.append([0, 0, 0, 0, 1])
        if len(features) < max_length:
            padding = [[0, 0, 0, 0, 0]] * (max_length - len(features))
            features.extend(padding)
        else:
            features = features[:max_length]
        X_test.append(features)
    return np.array(X_test)

def extract_sequence_features(seq_features):
    """
    Extract relevant sequence features from one-hot encoding.
    """
    # Get valid rows (non-padding)
    valid_mask = ~np.all(seq_features == 0, axis=1)
    valid_features = seq_features[valid_mask]
    
    # Calculate nucleotide composition
    a_content = np.mean(valid_features[:, 0])
    c_content = np.mean(valid_features[:, 1])
    g_content = np.mean(valid_features[:, 2])
    u_content = np.mean(valid_features[:, 3])
    gc_content = c_content + g_content
    
    return {
        'length': np.sum(valid_mask),
        'a_content': a_content,
        'c_content': c_content,
        'g_content': g_content, 
        'u_content': u_content,
        'gc_content': gc_content,
        'au_content': a_content + u_content
    }

def visualize_3d_structure(true_coords, pred_coords, sample_idx=0, title="3D Structure Comparison", show_plot=False):
    """
    Visualizes the true and predicted 3D structures for a sample.
    Only shows the plot if explicitly requested.
    """
    true = true_coords[sample_idx]
    pred = pred_coords[sample_idx]
    mask = ~np.all(true == 0, axis=1)
    true = true[mask]
    pred = pred[mask]
    
    fig = plt.figure(figsize=(15, 7))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(true[:, 0], true[:, 1], true[:, 2], 'b-', label='True')
    ax1.scatter(true[:, 0], true[:, 1], true[:, 2], c='b', s=20, alpha=0.5)
    ax1.set_title('True Structure')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.grid(True)
    
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(pred[:, 0], pred[:, 1], pred[:, 2], 'r-', label='Predicted')
    ax2.scatter(pred[:, 0], pred[:, 1], pred[:, 2], c='r', s=20, alpha=0.5)
    ax2.set_title('Predicted Structure')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    # Always save the figure
    filename = f'structure_comparison_{sample_idx}.png'
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    
    # Only show the plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
        
    return filename  # Return the filename for reference
