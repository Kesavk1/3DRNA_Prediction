def generate_base_structures_with_golden_seeds(
    X_test, 
    test_seq_df, 
    golden_seeds, 
    optimal_params, 
    X_valid, 
    y_valid
):
    """
    Generate base structures using golden seeds with RNA-specific optimizations.
    
    Parameters:
    -----------
    X_test: Test features
    test_seq_df: DataFrame with test sequences
    golden_seeds: List of golden seed information
    optimal_params: Model parameters
    X_valid, y_valid: Validation data for model training
    
    Returns:
    --------
    Dictionary mapping sequence IDs to lists of base structures
    """
    print("Generating base structures with golden seeds and RNA-specific optimizations...")
    
    # Dictionary to store base structures for each sequence
    seq_to_base_structures = {}
    
    # Initialize empty base structures list for each sequence
    for _, row in test_seq_df.iterrows():
        target_id = row['target_id']
        seq_to_base_structures[target_id] = []
    
    # Sort golden seeds by TM-score for best-first approach
    sorted_seeds = sorted(golden_seeds, key=lambda x: x['tm_score'], reverse=True)
    
    # For very small RNAs, different seeds may not add much diversity
    # For large RNAs, different seeds could capture different folding patterns
    small_rna_threshold = 50  # Nucleotides
    large_rna_threshold = 200  # Nucleotides
    
    # RNA-specific parameters based on sequence properties
    for i, (_, row) in enumerate(test_seq_df.iterrows()):
        target_id = row['target_id']
        seq = row['sequence']
        seq_length = len(seq)
        
        print(f"Processing sequence {i+1}/{len(test_seq_df)}, ID: {target_id}, length: {seq_length}")
        
        # Extract sequence features
        seq_features = extract_sequence_features(X_test[i])
        
        # Analyze sequence to determine RNA-specific parameters
        gc_content = seq_features['gc_content']
        au_content = seq_features['au_content']
        
        # Adjust parameters based on RNA properties
        if seq_length < small_rna_threshold:
            print(f"Small RNA detected (length={seq_length}). Using specialized parameters.")
            num_seeds_to_use = min(3, len(sorted_seeds))  # Use fewer seeds for small RNAs
            noise_scaling = 0.7  # Lower noise for small RNAs (more stable)
            use_global_movement = False  # Less global movement for small RNAs
            
            # Small RNAs with high GC content are more stable
            if gc_content > 0.6:
                noise_scaling *= 0.8  # Further reduce noise for GC-rich small RNAs
            
        elif seq_length < large_rna_threshold:
            print(f"Medium RNA detected (length={seq_length}).")
            num_seeds_to_use = min(4, len(sorted_seeds))
            noise_scaling = 1.0  # Standard noise level
            use_global_movement = True
            
            # For medium RNAs, GC content indicates stability regions
            if gc_content > 0.6:
                noise_scaling *= 0.9
            elif au_content > 0.6:
                noise_scaling *= 1.1  # AU-rich regions are more flexible
            
        else:
            print(f"Large RNA detected (length={seq_length}). Using specialized parameters.")
            num_seeds_to_use = min(5, len(sorted_seeds))  # Use more seeds for large RNAs
            noise_scaling = 0.5  # Lower noise for large RNAs (prevent unrealistic structures)
            use_global_movement = True  # Use global movement for large RNAs (domain flexibility)
            
            # Large RNAs tend to have distinct domains
            # Adjust parameters to reflect domain structure
            if seq_length > 300:
                num_seeds_to_use = min(5, len(sorted_seeds))  # Maximum diversity for very large RNAs
        
        # Process with selected seeds
        base_structures = []
        for seed_idx in range(num_seeds_to_use):
            if seed_idx < len(sorted_seeds):
                seed_info = sorted_seeds[seed_idx]
                print(f"  Generating with seed {seed_info['seed']} (TM-score: {seed_info['tm_score']:.4f})")
                
                # Set the random seed
                np.random.seed(seed_info['seed'])
                
                # Create model with adjusted parameters
                adjusted_noise = optimal_params['noise'] * noise_scaling
                
                # Create model with RNA-specific adjustments
                model = reference_based_approach(
                    X_valid, 
                    y_valid,
                    geometric_sampling=True,  # Always use geometric sampling for better structures
                    noise_level=adjusted_noise,
                    correlation=optimal_params['corr']
                )
                
                if model is None:
                    print(f"  Failed to create model with seed {seed_info['seed']}")
                    continue
                
                # Generate prediction
                try:
                    # Get basic prediction for this sequence
                    base_pred = model.predict(X_test[i:i+1])[0][:seq_length]
                    
                    # Apply RNA-specific post-processing
                    processed_pred = post_process_rna_structure(
                        base_pred, 
                        seq, 
                        gc_content, 
                        use_global_movement=use_global_movement
                    )
                    
                    # Normalize the structure
                    normalized_pred = normalize_structure(processed_pred)
                    
                    # Verify the structure meets basic validation criteria
                    if check_structure_validity(normalized_pred):
                        base_structures.append(normalized_pred)
                    else:
                        print(f"  Structure from seed {seed_info['seed']} failed validation. Attempting repair.")
                        
                        # Try to repair the structure
                        repaired_structure = repair_invalid_structure(normalized_pred)
                        if check_structure_validity(repaired_structure):
                            base_structures.append(repaired_structure)
                            print(f"  Successfully repaired structure from seed {seed_info['seed']}")
                        else:
                            print(f"  Could not repair structure from seed {seed_info['seed']}")
                    
                except Exception as e:
                    print(f"  Error generating prediction with seed {seed_info['seed']}: {str(e)}")
                    continue
        
        # If we didn't get any valid structures, create an emergency structure
        if not base_structures:
            print(f"Warning: No valid structures generated for {target_id}. Creating emergency structure.")
            emergency_structure = create_emergency_structure(seq_length)
            base_structures.append(emergency_structure)
        
        # Store the structures
        seq_to_base_structures[target_id] = base_structures
        print(f"  Generated {len(base_structures)} base structures for {target_id}")
    
    return seq_to_base_structures
