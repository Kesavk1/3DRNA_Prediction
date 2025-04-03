def generate_diverse_candidates(base_structures, seq_length, num_per_base=5):
    """
    Generate diverse candidate structures from a set of base structures.
    Adapts variation parameters based on RNA size.
    
    Parameters:
    -----------
    base_structures: List of base structures to generate variations from
    seq_length: Length of the sequence
    num_per_base: Number of variations to generate per base structure
    
    Returns:
    --------
    List of candidate structures
    """
    candidates = []
    
    # First, add all base structures
    for base in base_structures:
        candidates.append(base)
    
    # Then generate variations from each base
    for base_idx, base in enumerate(base_structures):
        print(f"  Generating variations from base structure {base_idx+1}/{len(base_structures)}...")
        
        # Determine noise levels based on sequence length
        if seq_length < 50:
            # Small RNA - can handle more variation
            noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
        elif seq_length < 120:
            # Medium RNA - moderate variation
            noise_levels = [0.05, 0.1, 0.15, 0.2, 0.25]
        else:
            # Large RNA - more conservative
            noise_levels = [0.03, 0.06, 0.09, 0.12, 0.15]
        
        # Generate variations with different parameters
        for i in range(num_per_base):
            # Use different parameters for diversity
            noise_idx = i % len(noise_levels)
            noise_level = noise_levels[noise_idx]
            preserve_distance = (i % 2 == 0)  # Alternate between preserving and not
            use_global = (i % 3 == 0)  # Occasional global movements
            
            # Add small random variation to correlation
            correlation = 0.8 + np.random.uniform(-0.1, 0.1)
            
            # Set a unique random seed for each variation
            np.random.seed(base_idx * 100 + i)
            
            variation = sample_structural_variation(
                base,
                noise_level=noise_level,
                preserve_distance=preserve_distance,
                use_global_movement=use_global,
                correlation=correlation
            )
            
            # Normalize the structure
            normalized = normalize_structure(variation)
            candidates.append(normalized)
    
    print(f"Generated {len(candidates)} candidate structures in total")
    return candidates

def generate_diverse_structures_from_bases(base_structures, seq_length, quality_model, num_per_base=5):
    """
    Generate diverse candidate structures from a set of base structures,
    with RNA-specific variations and quality filtering.
    
    Parameters:
    -----------
    base_structures: List of base structures to generate variations from
    seq_length: Length of the RNA sequence
    quality_model: Model for quality assessment
    num_per_base: Number of variations to generate per base structure
    
    Returns:
    --------
    List of diverse candidate structures
    """
    candidates = []
    
    # First, add all base structures
    for base in base_structures:
        candidates.append(base)
    
    # Then generate variations from each base
    for base_idx, base in enumerate(base_structures):
        print(f"  Generating variations from base structure {base_idx+1}/{len(base_structures)}...")
        
        # Determine variation parameters based on sequence length
        if seq_length < 50:
            # Small RNA - can handle more variation
            noise_levels = [0.05, 0.1, 0.15, 0.2, 0.25]
            preserve_distances = [True, True, True, False, False]  # Mostly preserve distances
            use_globals = [False, False, True, False, True]  # Occasional global movements
        elif seq_length < 120:
            # Medium RNA - moderate variation
            noise_levels = [0.03, 0.06, 0.1, 0.15, 0.2]
            preserve_distances = [True, True, True, True, False]  # Mostly preserve distances
            use_globals = [False, True, False, True, False]  # Mix of global and local
        else:
            # Large RNA - more conservative
            noise_levels = [0.02, 0.04, 0.06, 0.08, 0.1]
            preserve_distances = [True, True, True, True, True]  # Always preserve distances
            use_globals = [False, False, True, False, True]  # Occasional global for domains
        
        # Generate variations with different parameters
        for i in range(num_per_base):
            # Use different parameters for diversity
            noise_idx = i % len(noise_levels)
            noise_level = noise_levels[noise_idx]
            preserve_distance = preserve_distances[noise_idx]
            use_global = use_globals[noise_idx] 
            
            # Add small random variation to correlation
            correlation = 0.8 + np.random.uniform(-0.1, 0.1)
            
            # Set a unique random seed for each variation
            np.random.seed(base_idx * 100 + i)
            
            variation = sample_structural_variation(
                base,
                noise_level=noise_level,
                preserve_distance=preserve_distance,
                use_global_movement=use_global,
                correlation=correlation
            )
            
            # Apply additional RNA-specific refinements
            # For example, ensure proper backbone geometry
            variation = refine_rna_backbone(variation)
            
            # Normalize the structure
            normalized = normalize_structure(variation)
            
            # Verify the structure is valid
            if check_structure_validity(normalized):
                candidates.append(normalized)
            else:
                print(f"    Structure failed validation. Attempting repair.")
                repaired = repair_invalid_structure(normalized)
                if check_structure_validity(repaired):
                    candidates.append(repaired)
                    print(f"    Successfully repaired structure")
    
    print(f"Generated {len(candidates)} candidate structures in total")
    
    # Pre-filter candidates based on quality before detailed evaluation
    if len(candidates) > 30:  # Only pre-filter if we have many candidates
        print("Pre-filtering candidates based on basic quality metrics...")
        quality_scores = []
        
        # Simple quality assessment for pre-filtering
        for candidate in candidates:
            # Calculate basic quality score
            valid_mask = ~np.all(candidate == 0, axis=1)
            valid_coords = candidate[valid_mask]
            
            # Skip if too few valid coordinates
            if len(valid_coords) < 3:
                quality_scores.append(0.0)
                continue
            
            # Calculate bond lengths
            bond_lengths = []
            for j in range(1, len(valid_coords)):
                dist = np.linalg.norm(valid_coords[j] - valid_coords[j-1])
                bond_lengths.append(dist)
            
            # Score based on ideal bond length
            avg_bond_length = np.mean(bond_lengths)
            bond_score = 1.0 - min(1.0, abs(avg_bond_length - 3.8) / 3.8)
            
            quality_scores.append(bond_score)
        
        # Convert to numpy array
        quality_scores = np.array(quality_scores)
        
        # Take top 30 candidates based on quality score
        top_indices = np.argsort(quality_scores)[-30:]
        candidates = [candidates[idx] for idx in top_indices]
        print(f"Pre-filtered to top 30 candidates")
    
    return candidates

def evaluate_and_prune_structures(candidates, seq_features, quality_model, top_k=5):
    """
    Evaluate structure candidates and select the top-k structures.
    This function handles both NN-based and rule-based quality models.
    
    Parameters:
    -----------
    candidates: List of candidate structures
    seq_features: RNA sequence features
    quality_model: Model for quality assessment
    top_k: Number of top structures to select
    
    Returns:
    --------
    List of top-k structures
    """
    # Determine if the model is a neural network or rule-based
    is_nn_model = hasattr(quality_model, 'model')
    
    try:
        if is_nn_model:
            print("Using neural network for quality assessment...")
            return evaluate_and_prune_nn(candidates, seq_features, quality_model, top_k)
        else:
            print("Using rule-based model for quality assessment...")
            return evaluate_and_prune_rules(candidates, top_k)
        
    except Exception as e:
        print(f"Error during quality evaluation: {str(e)}")
        traceback.print_exc()
        
        # Fall back to rule-based evaluation if any error occurs
        print("Falling back to basic rule-based scoring...")
        return evaluate_and_prune_rules(candidates, top_k)
    def evaluate_and_prune_nn(candidates, seq_features, quality_model, top_k=5):
    """
    Evaluate candidates using NN model and select the top-k.
    
    Parameters:
    -----------
    candidates: List of candidate structures
    seq_features: One-hot encoded sequence features
    quality_model: Trained quality assessment model
    top_k: Number of top structures to keep
    
    Returns:
    --------
    List of top-k structures
    """
    try:
        # Extract actual sequence length (non-padding)
        valid_mask = ~np.all(seq_features == 0, axis=1)
        seq_length = np.sum(valid_mask)
        
        # Prepare batched data for prediction
        stacked_candidates = np.array(candidates)
        
        # Prepare sequence features input - deve ter o mesmo número de amostras que stacked_candidates
        batch_size = stacked_candidates.shape[0]
        
        # Expand seq_features to have batch_size samples (replicando para cada candidato)
        # Certifique-se de que seq_features tem 3 dimensões (batch, seq_len, features)
        if len(seq_features.shape) == 2:  # Se for (seq_len, features)
            seq_features = np.expand_dims(seq_features, axis=0)  # Adicionar dimensão de batch
        
        # Replicar para todos os candidatos
        stacked_seq = np.repeat(seq_features, batch_size, axis=0)
        
        # Predict quality scores
        quality_scores = quality_model.predict_quality(stacked_candidates, stacked_seq)
        quality_scores = quality_scores.flatten()
        
        # Sort by quality score
        sorted_indices = np.argsort(quality_scores)[::-1]  # Descending order
        
        # Keep top-k structures
        top_structures = [candidates[idx] for idx in sorted_indices[:top_k]]
        top_scores = quality_scores[sorted_indices[:top_k]]
        
        print(f"Selected top {top_k} structures with NN predicted qualities: {top_scores}")
        
        return top_structures
        
    except Exception as e:
        print(f"Error in NN evaluation: {str(e)}")
        traceback.print_exc()
        
        # Fall back to rule-based approach if NN fails
        print("Falling back to rule-based evaluation...")
        return evaluate_and_prune_rules(candidates, top_k)

def evaluate_and_prune_rules(candidates, top_k=5):
    """
    Evaluate candidates using rule-based metrics and select the top-k.
    
    Parameters:
    -----------
    candidates: List of candidate structures
    top_k: Number of top structures to keep
    
    Returns:
    --------
    List of top-k structures
    """
    quality_scores = []
    
    for i, candidate in enumerate(candidates):
        # Calculate a quality score based on structural features
        # 1. Check for valid coordinates
        valid_mask = ~np.all(candidate == 0, axis=1)
        valid_coords = candidate[valid_mask]
        
        # Skip if no valid coordinates
        if len(valid_coords) < 3:
            quality_scores.append(0.5)  # Neutral score
            continue
        
        # 2. Calculate bond lengths between consecutive residues
        bond_lengths = []
        for j in range(1, len(valid_coords)):
            dist = np.linalg.norm(valid_coords[j] - valid_coords[j-1])
            bond_lengths.append(dist)
        
        avg_bond_length = np.mean(bond_lengths)
        bond_std = np.std(bond_lengths)
        
        # 3. Score based on how close to ideal RNA bond length (3.8Å)
        bond_score = 1.0 - min(1.0, abs(avg_bond_length - 3.8) / 3.8)
        
        # 4. Bond consistency score (lower std deviation is better)
        consistency_score = 1.0 - min(1.0, bond_std / 2.0)
        
        # 5. Check structure validity
        is_valid = check_structure_validity(candidate)
        valid_score = 1.0 if is_valid else 0.5
        
        # 6. Combined score
        score = 0.4 * bond_score + 0.3 * consistency_score + 0.3 * valid_score
        
        # 7. Add small random component for variations
        random_component = np.random.uniform(-0.05, 0.05)
        score = min(1.0, max(0.0, score + random_component))
        
        quality_scores.append(score)
    
    # Convert to numpy array
    quality_scores = np.array(quality_scores)
    
    # Sort by quality score
    sorted_indices = np.argsort(quality_scores)[::-1]  # Descending order
    
    # Keep top-k structures
    top_structures = [candidates[idx] for idx in sorted_indices[:top_k]]
    top_scores = quality_scores[sorted_indices[:top_k]]
    
    print(f"Selected top {top_k} structures with rule-based qualities: {top_scores}")
    
    return top_structures

def generate_and_prune_structures(base_coords, seq_features, quality_model, num_candidates=20, top_k=5):
    """
    Generate multiple structure candidates and use the NN model to prune to the best ones.
    Modified to handle variable-length RNA sequences.
    """
    # Get actual sequence length (non-padding)
    valid_mask = ~np.all(base_coords == 0, axis=1)
    seq_length = np.sum(valid_mask)
    print(f"Processing structure with actual length: {seq_length}")
    
    # Generate candidate structures with different parameters
    candidates = []
    
    # Add the base structure
    candidates.append(normalize_structure(base_coords))
    
    # Generate variations with different parameters
    for i in range(num_candidates - 1):
        # Use different parameters for diversity
        noise_level = 0.1 + (i % 10) * 0.05
        preserve_distance = (i % 3 != 0)
        use_global = (i % 4 == 0)
        correlation = 0.7 + (i % 5) * 0.05
        
        variation = sample_structural_variation(
            base_coords,
            noise_level=noise_level,
            preserve_distance=preserve_distance,
            use_global_movement=use_global,
            correlation=correlation
        )
        
        # Normalize the structure
        normalized = normalize_structure(variation)
        candidates.append(normalized)
    
    # Convert to array for batch processing
    stacked_candidates = np.array(candidates)
    
    # Implement a simple rule-based quality assessment as fallback
    print("Using rule-based quality assessment...")
    quality_scores = []
    
    for i, candidate in enumerate(candidates):
        # Calculate a quality score based on structural features
        # 1. Check for unusual bond lengths
        valid_indices = np.where(valid_mask)[0]
        valid_coords = candidate[valid_indices]
        
        # Skip if no valid coordinates
        if len(valid_coords) < 3:
            quality_scores.append(0.5)
            continue
        
        # Calculate bond lengths
        bond_lengths = []
        for j in range(1, len(valid_coords)):
            dist = np.linalg.norm(valid_coords[j] - valid_coords[j-1])
            bond_lengths.append(dist)
        
        # Score based on how close to ideal RNA bond length
        avg_bond_length = np.mean(bond_lengths)
        bond_std = np.std(bond_lengths)
        
        # Ideal bond length is around 3.8Å
        bond_score = 1.0 - min(1.0, abs(avg_bond_length - 3.8) / 3.8)
        
        # Bond consistency score
        consistency_score = 1.0 - min(1.0, bond_std / 2.0)
        
        # Structural validity
        is_valid = check_structure_validity(candidate)
        valid_score = 1.0 if is_valid else 0.5
        
        # Combined score
        final_score = 0.4 * bond_score + 0.3 * consistency_score + 0.3 * valid_score
        
        # Add a small random component for variations
        random_component = np.random.uniform(-0.05, 0.05)
        final_score = min(1.0, max(0.0, final_score + random_component))
        
        quality_scores.append(final_score)
    
    quality_scores = np.array(quality_scores)
    
    # Sort candidates by quality score
    sorted_indices = np.argsort(quality_scores)[::-1]  # Descending order
    
    # Keep top-k structures
    top_structures = [candidates[idx] for idx in sorted_indices[:top_k]]
    top_scores = quality_scores[sorted_indices[:top_k]]
    
    print(f"Selected top {top_k} structures with predicted qualities: {top_scores}")
    
    return top_structures
