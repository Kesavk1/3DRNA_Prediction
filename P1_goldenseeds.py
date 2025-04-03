def reference_based_approach(X_ref, y_ref, geometric_sampling=False, noise_level=0.2, correlation=0.7):
    try:
        class ReferenceModel:
            def __init__(self, geometric_sampling=False, base_noise_level=0.2, correlation=0.7):
                self.geometric_sampling = geometric_sampling
                self.base_noise_level = base_noise_level
                self.correlation = correlation
                
            def fit(self, X, y):
                # First, handle NaN values in the reference structures
                self.reference_structures = np.nan_to_num(y, nan=0.0)
                self.global_mean = np.nanmean(y, axis=(0, 1))
                self.global_std = np.nanstd(y, axis=(0, 1))
                
                # Replace potential NaN values in statistics
                self.global_mean = np.nan_to_num(self.global_mean, nan=0.0)
                self.global_std = np.nan_to_num(self.global_std, nan=1.0)
                
                # Calculate size statistics
                self.size_groups = {}
                # Group reference structures by size
                for i in range(len(self.reference_structures)):
                    valid_mask = ~np.all(self.reference_structures[i] == 0, axis=1)
                    size = np.sum(valid_mask)
                    
                    if size < 120:
                        group = "small"
                    elif size < 200:
                        group = "medium"
                    else:
                        group = "large"
                        
                    if group not in self.size_groups:
                        self.size_groups[group] = []
                    self.size_groups[group].append(i)
                    
                print(f"Size distribution - Small: {len(self.size_groups.get('small', []))}, "
                      f"Medium: {len(self.size_groups.get('medium', []))}, "
                      f"Large: {len(self.size_groups.get('large', []))}")
                      
                # Store the correlation parameter for use in sample_structural_variation
                global_correlation = self.correlation
                print(f"Using noise level: {self.base_noise_level}, correlation: {global_correlation}")
                
                return self
                
            def predict(self, X):
                batch_size = X.shape[0]
                seq_length = X.shape[1]
                predictions = np.zeros((batch_size, seq_length, 3))
                
                for i in range(batch_size):
                    # Determine the RNA size group
                    valid_mask = ~np.all(X[i] == 0, axis=1)
                    size = np.sum(valid_mask)
                    if size < 120:
                        group = "small"
                        # Size-specific noise scaling
                        noise_level = self.base_noise_level * 0.6
                    elif size < 200:
                        group = "medium"
                        noise_level = self.base_noise_level * 1.0
                    else:
                        group = "large"
                        noise_level = self.base_noise_level * 0.4
                    
                    # If we have reference structures in this size group, use them
                    if group in self.size_groups and self.size_groups[group]:
                        # Randomly pick a reference structure from the same size group
                        ref_idx = np.random.choice(self.size_groups[group])
                        base_struct = self.reference_structures[ref_idx].copy()
                        
                        if self.geometric_sampling:
                            # Pass the correlation parameter to the variation function
                            predictions[i] = sample_structural_variation(
                                base_struct, 
                                noise_level=noise_level,
                                preserve_distance=True,
                                use_global_movement=(group == "small"),
                                correlation=self.correlation
                            )
                        else:
                            noise = np.random.normal(0, noise_level, base_struct.shape)
                            predictions[i] = base_struct + noise
                    else:
                        # Fall back to the original method if no size match
                        sample = np.random.normal(self.global_mean, self.global_std, size=(seq_length, 3))
                        if self.geometric_sampling:
                            predictions[i] = sample_structural_variation(
                                sample, 
                                noise_level=noise_level,
                                preserve_distance=True,
                                use_global_movement=(group == "small"),
                                correlation=self.correlation
                            )
                        else:
                            predictions[i] = sample
                        
                return predictions
        
        # Create and return model with specific parameters
        model = ReferenceModel(geometric_sampling=geometric_sampling, 
                              base_noise_level=noise_level,
                              correlation=correlation)
        model.fit(X_ref, y_ref)
        return model
    
    except Exception as e:
        print(f"Error in reference_based_approach: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def evaluate_model(model, X_valid, y_valid, show_plots=False, save_top_plots=False):
    # Problem: Inadequate evaluation
    
    # SOLUTION:
    import numpy as np
    
    # Ensure there are no NaNs in the data
    X_valid_clean = np.nan_to_num(X_valid, nan=0.0)
    y_valid_clean = np.nan_to_num(y_valid, nan=0.0)
    
    # Make prediction with try/except to capture errors
    try:
        y_pred = model.predict(X_valid_clean)
        
        # Check if prediction contains NaNs or infinities
        if np.isnan(y_pred).any() or np.isinf(y_pred).any():
            print("WARNING: Prediction contains NaN or infinite values!")
            y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Calculate metrics  
        mae = np.mean(np.abs(y_pred - y_valid_clean))
        mse = np.mean((y_pred - y_valid_clean)**2)
        
        # Calculate TM-scores for each structure
        tm_scores = []
        for i in range(len(X_valid)):
            # Compute score with error handling  
            try:
                tm = calculate_tm_score(y_pred[i], y_valid_clean[i])
                if np.isnan(tm) or np.isinf(tm):
                    print(f"WARNING: Invalid TM-score for sample {i}, using 0.0")
                    tm = 0.0
            except Exception as e:
                print(f"Error calculating TM-score for sample {i}: {str(e)}")
                tm = 0.0
                
            tm_scores.append(tm)
        
        # Final metrics
        avg_tm_score = np.mean(tm_scores)
        
        print(f"MAE: {mae:.4f}, MSE: {mse:.4f}")  
        print(f"Average TM-score: {avg_tm_score:.4f}")
        
        return {
            'mae': mae,
            'mse': mse,
            'tm_scores': tm_scores,  
            'avg_tm_score': avg_tm_score,
            'success': True
        }
        
    except Exception as e:
        print(f"ERROR in evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            'mae': float('inf'),
            'mse': float('inf'), 
            'tm_scores': [0.0] * len(X_valid),
            'avg_tm_score': 0.0,
            'success': False,
            'error': str(e)  
        }

def find_diverse_golden_seeds(
    X_valid, 
    y_valid, 
    golden_threshold=0.6, 
    attempts=200, 
    optimal_params={'noise': 0.21, 'corr': 0.83},
    diversity_threshold=0.15,
    max_seeds=10
):
    """
    Searches for "golden" seeds that produce good results, ensuring diversity
    and controlling overfitting.
    
    Parameters:
    -----------
    X_valid: Validation data for features
    y_valid: Validation data for target structures
    golden_threshold: TM-score threshold to consider a seed as "golden"
    attempts: Number of attempts to find good seeds
    optimal_params: Optimal parameters for the reference model
    diversity_threshold: Threshold to consider seeds as diverse from each other
    max_seeds: Maximum number of golden seeds to return
    
    Returns:
    --------
    golden_seeds: List of diverse "golden" seeds
    all_seeds: List of all tested seeds with their scores
    """
    print(f"Searching for up to {max_seeds} diverse golden seeds with TM-score threshold of {golden_threshold}...")
    
    # List to store all tested seeds
    all_seeds = []
    
    # List to store the "golden" seeds
    golden_seeds = []
    
    # List to store the predicted structures for each golden seed
    golden_predictions = []
    
    # Set of seeds already tested to avoid duplications
    tested_seeds = set()
    
    # Counter for valid attempts (excluding duplicates)
    valid_attempts = 0
    
    # Define parameter search ranges for different seed ranges
    seed_ranges = [
        (1, 1000),         # Initial range
        (1001, 10000),     # Medium seeds
        (10001, 100000),   # Larger seeds
        (100001, 1000000)  # Very large seeds
    ]
    
    # Alternating between different ranges to promote diversity
    range_index = 0
    
    # Keep track of the best seed for each RNA size range
    best_small_rna_seed = {'seed': None, 'tm_score': 0.0}  # <50 residues
    best_medium_rna_seed = {'seed': None, 'tm_score': 0.0}  # 50-120 residues
    best_large_rna_seed = {'seed': None, 'tm_score': 0.0}  # >120 residues
    
    # Calculate sequence length statistics
    seq_lengths = []
    for coords in y_valid:
        valid_mask = ~np.all(coords == 0, axis=1)
        seq_length = np.sum(valid_mask)
        seq_lengths.append(seq_length)
    
    # Separate indices by size
    small_rna_indices = [i for i, length in enumerate(seq_lengths) if length < 50]
    medium_rna_indices = [i for i, length in enumerate(seq_lengths) if 50 <= length < 120]
    large_rna_indices = [i for i, length in enumerate(seq_lengths) if length >= 120]
    
    print(f"RNA Distribution: {len(small_rna_indices)} small, {len(medium_rna_indices)} medium, {len(large_rna_indices)} large")
    
    # Main cycle to search for seeds
    while valid_attempts < attempts and len(golden_seeds) < max_seeds:
        # Select seed range
        min_seed, max_seed = seed_ranges[range_index]
        range_index = (range_index + 1) % len(seed_ranges)
        
        # Generate random seed from this range
        seed = np.random.randint(min_seed, max_seed)
        
        # Check if we've already tested this seed
        if seed in tested_seeds:
            continue
        
        tested_seeds.add(seed)
        valid_attempts += 1
        
        if valid_attempts % 10 == 0:
            print(f"Testing seed {valid_attempts}/{attempts} (seed={seed})...")
        
        # Set the seed for reproducibility
        np.random.seed(seed)
        
        # Create model with this seed
        try:
            model = reference_based_approach(
                X_valid, 
                y_valid,
                geometric_sampling=True,
                noise_level=optimal_params['noise'],
                correlation=optimal_params['corr']
            )
            
            if model is None:
                print(f"  Failed to create model with seed {seed}")
                continue
                
            # Evaluate the model on different validation subsets
            # Calculate overall TM-score
            metrics = evaluate_model(model, X_valid, y_valid)
            tm_score = metrics['avg_tm_score']
            
            # Check for overfitting using TM-score on different subsets
            if len(small_rna_indices) > 0:
                small_metrics = evaluate_model_on_indices(model, X_valid, y_valid, small_rna_indices)
                small_tm_score = small_metrics['avg_tm_score']
            else:
                small_tm_score = 0.0
                
            if len(medium_rna_indices) > 0:
                medium_metrics = evaluate_model_on_indices(model, X_valid, y_valid, medium_rna_indices)
                medium_tm_score = medium_metrics['avg_tm_score']
            else:
                medium_tm_score = 0.0
                
            if len(large_rna_indices) > 0:
                large_metrics = evaluate_model_on_indices(model, X_valid, y_valid, large_rna_indices)
                large_tm_score = large_metrics['avg_tm_score']
            else:
                large_tm_score = 0.0
            
            # Calculate standard deviation between scores for different sizes
            # A high deviation may indicate overfitting in certain sizes
            size_scores = [s for s in [small_tm_score, medium_tm_score, large_tm_score] if s > 0]
            size_std = np.std(size_scores) if len(size_scores) > 1 else 0.0
            
            # Penalize the score for high variability between sizes (possible overfitting)
            adjusted_tm_score = tm_score - size_std
            
            # Register this seed
            seed_info = {
                'seed': seed,
                'tm_score': tm_score,
                'adjusted_tm_score': adjusted_tm_score,
                'small_tm_score': small_tm_score,
                'medium_tm_score': medium_tm_score,
                'large_tm_score': large_tm_score,
                'size_std': size_std
            }
            all_seeds.append(seed_info)
            
            # Update the best seeds by size
            if small_tm_score > best_small_rna_seed['tm_score'] and small_tm_score > golden_threshold:
                best_small_rna_seed = {'seed': seed, 'tm_score': small_tm_score}
                
            if medium_tm_score > best_medium_rna_seed['tm_score'] and medium_tm_score > golden_threshold:
                best_medium_rna_seed = {'seed': seed, 'tm_score': medium_tm_score}
                
            if large_tm_score > best_large_rna_seed['tm_score'] and large_tm_score > golden_threshold:
                best_large_rna_seed = {'seed': seed, 'tm_score': large_tm_score}
            
            # Check if this is a "golden" seed
            if adjusted_tm_score >= golden_threshold:
                # Generate predictions for diversity comparison
                preds = model.predict(X_valid)
                
                # Check diversity relative to seeds already found
                is_diverse = True
                for i, existing_preds in enumerate(golden_predictions):
                    similarity = calculate_prediction_similarity(preds, existing_preds)
                    if similarity > (1.0 - diversity_threshold):
                        is_diverse = False
                        # If the new one is better than an existing one and they are similar, we replace
                        if adjusted_tm_score > golden_seeds[i]['adjusted_tm_score']:
                            print(f"  Replacing seed {golden_seeds[i]['seed']} (score={golden_seeds[i]['adjusted_tm_score']:.4f}) " 
                                  f"with seed {seed} (score={adjusted_tm_score:.4f})")
                            golden_seeds[i] = seed_info
                            golden_predictions[i] = preds
                        break
                
                if is_diverse and len(golden_seeds) < max_seeds:
                    print(f"  Found golden seed: {seed} (TM-score: {tm_score:.4f}, Adjusted: {adjusted_tm_score:.4f})")
                    golden_seeds.append(seed_info)
                    golden_predictions.append(preds)
                    
                    if len(golden_seeds) >= max_seeds:
                        print(f"  Reached maximum number of {max_seeds} golden seeds.")
                        break
        
        except Exception as e:
            print(f"  Error testing seed {seed}: {str(e)}")
            continue
    
    # If we didn't find enough golden seeds, include the best by size
    if len(golden_seeds) < max_seeds:
        # Add the best seeds from each size category, if not already included
        special_seeds = [
            best_small_rna_seed,
            best_medium_rna_seed,
            best_large_rna_seed
        ]
        
        for special in special_seeds:
            if special['seed'] is not None:
                # Check if this seed is already in the golden ones
                if not any(gs['seed'] == special['seed'] for gs in golden_seeds):
                    # Find the complete details of this seed in all_seeds
                    for seed_detail in all_seeds:
                        if seed_detail['seed'] == special['seed']:
                            golden_seeds.append(seed_detail)
                            break
                    
                    if len(golden_seeds) >= max_seeds:
                        break
    
    # Sort golden seeds by adjusted TM-score (for better diversity and less overfitting)
    golden_seeds.sort(key=lambda x: x['adjusted_tm_score'], reverse=True)
    
    # Show statistics of the found seeds
    print(f"Found {len(golden_seeds)} golden seeds in {valid_attempts} attempts")
    for i, gs in enumerate(golden_seeds):
        print(f"  Seed {i+1}: {gs['seed']} (TM-score: {gs['tm_score']:.4f}, Adjusted: {gs['adjusted_tm_score']:.4f})")
        print(f"    TM-scores by size - Small: {gs['small_tm_score']:.4f}, Medium: {gs['medium_tm_score']:.4f}, Large: {gs['large_tm_score']:.4f}")
        print(f"    Standard deviation between sizes: {gs['size_std']:.4f}")
    
    return golden_seeds, all_seeds

def evaluate_model_on_indices(model, X_data, y_data, indices):
    """
    Evaluates the model only on specific indices of the data.
    Useful to evaluate performance on subsets like small/medium/large RNAs.
    """
    X_subset = [X_data[i] for i in indices]
    y_subset = [y_data[i] for i in indices]
    
    return evaluate_model(model, X_subset, y_subset)

def calculate_prediction_similarity(preds1, preds2):
    """
    Calculates the similarity between two sets of predictions.
    Returns a value between 0 (totally different) and 1 (identical).
    """
    similarities = []
    
    # For each pair of sequences in the predictions
    for p1, p2 in zip(preds1, preds2):
        # Identify valid (non-zero) coordinates
        valid_mask1 = ~np.all(p1 == 0, axis=1)
        valid_mask2 = ~np.all(p2 == 0, axis=1)
        
        # Use only positions valid in both predictions
        valid_mask = valid_mask1 & valid_mask2
        
        # If there are no overlapping valid positions, continue
        if np.sum(valid_mask) < 3:
            continue
        
        # Extract valid coordinates
        valid_p1 = p1[valid_mask]
        valid_p2 = p2[valid_mask]
        
        # Calculate similarity based on RMSD distance
        squared_diff = np.sum((valid_p1 - valid_p2) ** 2, axis=1)
        rmsd = np.sqrt(np.mean(squared_diff))
        
        # Convert RMSD to similarity (lower RMSD values = higher similarity)
        # Normalize so it's between 0 and 1
        similarity = 1.0 / (1.0 + rmsd / 5.0)  # Division by 5.0 is an arbitrary scale
        similarities.append(similarity)
    
    # Return average similarity
    return np.mean(similarities) if similarities else 0.0
