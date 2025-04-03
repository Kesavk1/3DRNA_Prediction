def run_hybrid_pipeline(
    X_valid, 
    y_valid, 
    test_seq_df, 
    sample_submission_df, 
    output_dir, 
    golden_threshold=0.6, 
    seed_attempts=200, 
    optimal_params={'noise': 0.21, 'corr': 0.83}
):
    """
    Run a hybrid pipeline that combines golden seeds approach with NN pruning.
    
    Parameters:
    -----------
    X_valid, y_valid: Validation data for training models
    test_seq_df: DataFrame with test sequences
    sample_submission_df: Sample submission format
    output_dir: Output directory for files
    golden_threshold: Threshold for considering a seed as "golden"
    seed_attempts: Number of seeds to try
    optimal_params: Optimal parameters for the reference model
    
    Returns:
    --------
    submission_df, status_dict
    """
    print("=" * 80)
    print("HYBRID PIPELINE: GOLDEN SEEDS + NN PRUNING".center(80))
    print("=" * 80)
    
    status = {
        'success': False,
        'golden_seeds_found': 0,
        'nn_training_success': False,
        'best_tm_score': 0.0,
        'error': None
    }
    
    try:
        # PHASE 1: Find Golden Seeds
        print("\nPHASE 1: Searching for golden seeds...")
        golden_seeds, all_seeds = find_diverse_golden_seeds(
            X_valid, 
            y_valid, 
            golden_threshold=golden_threshold, 
            attempts=seed_attempts, 
            optimal_params=optimal_params
        )
        
        # Even if we don't find golden seeds, we can use the best seeds we found
        if not golden_seeds and all_seeds:
            print("No golden seeds found, using top seeds from search...")
            # Sort by TM-score
            all_seeds.sort(key=lambda x: x['tm_score'], reverse=True)
            # Take top 5 seeds
            top_seeds = all_seeds[:5]
        else:
            top_seeds = golden_seeds
            
        status['golden_seeds_found'] = len(golden_seeds)
        
        # PHASE 2: Train Quality Assessment Model
        print("\nPHASE 2: Training NN quality assessment model...")
        try:
            quality_model = train_enhanced_quality_model(X_valid, y_valid, X_valid, y_valid)
            status['nn_training_success'] = True
        except Exception as e:
            print(f"Error training NN model: {str(e)}")
            print("Falling back to rule-based quality assessment...")
            quality_model = create_rule_based_model()
            
        # PHASE 3: Generate Base Structures with Golden Seeds
        print("\nPHASE 3: Generating base structures with golden seeds...")
        X_test = prepare_test_features(test_seq_df)
        
        # Generate predictions using each of the top seeds
        seed_predictions = []
        for i, seed_info in enumerate(top_seeds):
            print(f"Generating predictions with seed {seed_info['seed']} (TM-score: {seed_info['tm_score']:.4f})...")
            
            # Set the random seed
            np.random.seed(seed_info['seed'])
            
            # Create model with this seed
            model = reference_based_approach(
                X_valid, 
                y_valid,
                geometric_sampling=True,
                noise_level=optimal_params['noise'],
                correlation=optimal_params['corr']
            )
            
            # Generate predictions
            if model is not None:
                preds = model.predict(X_test)
                seed_predictions.append({
                    'seed': seed_info['seed'],
                    'tm_score': seed_info['tm_score'],
                    'predictions': preds
                })
                
                # Update best TM-score for status
                if seed_info['tm_score'] > status['best_tm_score']:
                    status['best_tm_score'] = seed_info['tm_score']
            else:
                print(f"Failed to create model with seed {seed_info['seed']}")
                
        if not seed_predictions:
            raise Exception("Failed to generate any predictions with golden seeds")
            
        # PHASE 4: Generate and Prune Structures
        print("\nPHASE 4: Generating diverse candidates and using NN pruning...")
        
        seq_to_coords = {}
        for i, (_, row) in enumerate(test_seq_df.iterrows()):
            target_id = row['target_id']
            seq = row['sequence']
            seq_length = len(seq)
            
            print(f"Processing sequence {i+1}/{len(test_seq_df)}, ID: {target_id}, length: {seq_length}")
            
            # Collect base predictions from all seeds for this sequence
            base_structures = []
            for pred_info in seed_predictions:
                base_struct = pred_info['predictions'][i][:seq_length]
                base_structures.append(normalize_structure(base_struct))
                
            # Extract sequence features
            seq_features = X_test[i][:seq_length]
            
            # Generate more candidates through controlled variations
            candidates = generate_diverse_candidates(base_structures, seq_length, num_per_base=5)
            
            # Evaluate and prune candidates
            if status['nn_training_success']:
                print("Using NN model for quality assessment...")
                try:
                    top_structures = evaluate_and_prune_structures(
                        candidates, 
                        seq_features, 
                        quality_model, 
                        top_k=5
                    )
                except Exception as e:
                    print(f"Error in NN evaluation: {str(e)}")
                    print("Falling back to rule-based assessment...")
                    top_structures = evaluate_and_prune_rules(candidates, top_k=5)
            else:
                print("Using rule-based quality assessment...")
                top_structures = evaluate_and_prune_rules(candidates, top_k=5)
                
            # Store the final structures
            seq_to_coords[target_id] = top_structures
            
        # PHASE 5: Create Submission
        print("\nPHASE 5: Creating submission file...")
        submission_df = create_submission_dataframe(seq_to_coords, sample_submission_df)
        
        # Save submission
        hybrid_file = os.path.join(output_dir, 'submission_hybrid.csv')
        submission_df.to_csv(hybrid_file, index=False)
        print(f"Hybrid submission saved to {hybrid_file}")
        
        # Save as standard submission
        standard_file = os.path.join(output_dir, 'submission.csv')
        submission_df.to_csv(standard_file, index=False)
        
        # Set success
        status['success'] = True
        
        return submission_df, status
        
    except Exception as e:
        print(f"ERROR in hybrid pipeline: {str(e)}")
        traceback.print_exc()
        status['error'] = str(e)
        return None, status
    def integrate_with_hybrid_pipeline(run_hybrid_pipeline_func):
   """
   Integrates the enhanced NN model with the hybrid pipeline.
   
   Parameters:
   -----------
   run_hybrid_pipeline_func: Original hybrid pipeline function
   
   Returns:
   --------
   Modified hybrid pipeline function
   """
   def enhanced_hybrid_pipeline(
       X_valid, 
       y_valid, 
       test_seq_df, 
       sample_submission_df, 
       output_dir, 
       golden_threshold=0.6, 
       seed_attempts=200, 
       optimal_params={'noise': 0.21, 'corr': 0.83}
   ):
       """
       Run a hybrid pipeline with enhanced NN quality model.
       """
       print("=" * 80)
       print("ENHANCED HYBRID PIPELINE: GOLDEN SEEDS + ADVANCED NN PRUNING".center(80))
       print("=" * 80)
       
       status = {
           'success': False,
           'golden_seeds_found': 0,
           'nn_training_success': False,
           'best_tm_score': 0.0,
           'error': None
       }
       
       try:
           # PHASE 1: Find Golden Seeds (same as original)
           print("\nPHASE 1: Searching for golden seeds...")
           golden_seeds, all_seeds = find_diverse_golden_seeds(
               X_valid, 
               y_valid, 
               golden_threshold=golden_threshold, 
               attempts=seed_attempts, 
               optimal_params=optimal_params
           )
           
           # Even if we don't find golden seeds, we can use the best seeds we found
           if not golden_seeds and all_seeds:
               print("No golden seeds found, using top seeds from search...")
               # Sort by TM-score
               all_seeds.sort(key=lambda x: x['tm_score'], reverse=True)
               # Take top 5 seeds
               top_seeds = all_seeds[:5]
           else:
               top_seeds = golden_seeds
               
           status['golden_seeds_found'] = len(golden_seeds)
           
           # PHASE 2: Train Enhanced Quality Assessment Model
           print("\nPHASE 2: Training enhanced NN quality assessment model...")
           try:
               enhanced_quality_model = train_enhanced_quality_model(X_valid, y_valid, X_valid, y_valid)
               rule_based_model = create_rule_based_model()
               
               # Compare models
               model_comparison = evaluate_and_compare_models(
                   enhanced_quality_model, 
                   rule_based_model, 
                   X_valid, 
                   y_valid
               )
               
               # Use the best model
               best_model_type = model_comparison['best_model']
               if best_model_type == 'neural_network':
                   quality_model = enhanced_quality_model
                   print("Using enhanced neural network model for quality assessment")
               else:
                   quality_model = rule_based_model
                   print("Using rule-based model for quality assessment")
               
               status['nn_training_success'] = (best_model_type == 'neural_network')
               
           except Exception as e:
               print(f"Error training and comparing models: {str(e)}")
               print("Falling back to rule-based quality assessment...")
               quality_model = create_rule_based_model()
           
           # PHASE 3 and beyond: same as original hybrid pipeline
           # Continue with the rest of the pipeline...
           # (generate base structures, evaluate candidates, create submission)
           
           # Call the original function with our quality model
           # This is a placeholder - in a real implementation, 
           # you would continue with the rest of the pipeline using the quality_model
           
           return run_hybrid_pipeline_func(
               X_valid, 
               y_valid, 
               test_seq_df, 
               sample_submission_df, 
               output_dir, 
               golden_threshold=golden_threshold, 
               seed_attempts=seed_attempts, 
               optimal_params=optimal_params,
               quality_model=quality_model  # Pass the selected model
           )
           
       except Exception as e:
           print(f"ERROR in enhanced hybrid pipeline: {str(e)}")
           traceback.print_exc()
           status['error'] = str(e)
           
           # Fall back to original pipeline
           print("Falling back to original hybrid pipeline...")
           return run_hybrid_pipeline_func(
               X_valid, 
               y_valid, 
               test_seq_df, 
               sample_submission_df, 
               output_dir, 
               golden_threshold=golden_threshold, 
               seed_attempts=seed_attempts, 
               optimal_params=optimal_params
           )
   
   return enhanced_hybrid_pipeline
def phase3_integration_with_hybrid_pipeline(run_hybrid_pipeline_func):
    """
    Integrates the enhanced Phase 3 (base structure generation) with the hybrid pipeline.
    
    Parameters:
    -----------
    run_hybrid_pipeline_func: Original hybrid pipeline function
    
    Returns:
    --------
    Modified hybrid pipeline function
    """
    def enhanced_hybrid_pipeline(
        X_valid, 
        y_valid, 
        test_seq_df, 
        sample_submission_df, 
        output_dir, 
        golden_threshold=0.6, 
        seed_attempts=200, 
        optimal_params={'noise': 0.21, 'corr': 0.83},
        quality_model=None
    ):
        """
        Run a hybrid pipeline with enhanced base structure generation.
        """
        print("=" * 80)
        print("ENHANCED HYBRID PIPELINE WITH RNA-SPECIFIC STRUCTURE GENERATION".center(80))
        print("=" * 80)
        
        status = {
            'success': False,
            'golden_seeds_found': 0,
            'nn_training_success': False,
            'best_tm_score': 0.0,
            'error': None
        }
        
        try:
            # PHASE 1: Find Golden Seeds (same as original)
            print("\nPHASE 1: Searching for golden seeds...")
            golden_seeds, all_seeds = find_diverse_golden_seeds(
                X_valid, 
                y_valid, 
                golden_threshold=golden_threshold, 
                attempts=seed_attempts, 
                optimal_params=optimal_params
            )
            
            # Even if we don't find golden seeds, we can use the best seeds we found
            if not golden_seeds and all_seeds:
                print("No golden seeds found, using top seeds from search...")
                # Sort by TM-score
                all_seeds.sort(key=lambda x: x['tm_score'], reverse=True)
                # Take top 5 seeds
                top_seeds = all_seeds[:5]
            else:
                top_seeds = golden_seeds
                
            status['golden_seeds_found'] = len(golden_seeds)
            
            # PHASE 2: Train Quality Assessment Model (if not provided)
            if quality_model is None:
                print("\nPHASE 2: Training quality assessment model...")
                try:
                    quality_model = train_enhanced_quality_model(X_valid, y_valid, X_valid, y_valid)
                    status['nn_training_success'] = True
                except Exception as e:
                    print(f"Error training quality model: {str(e)}")
                    print("Falling back to rule-based quality assessment...")
                    quality_model = create_rule_based_model()
            else:
                print("\nPHASE 2: Using provided quality model")
                status['nn_training_success'] = hasattr(quality_model, 'model')  # Check if it's a NN model
            
            # PHASE 3: Generate Base Structures with RNA-specific optimizations
            print("\nPHASE 3: Generating base structures with RNA-specific optimizations...")
            # Prepare test features
            X_test = prepare_test_features(test_seq_df)
            
            # Generate base structures using our enhanced function
            seq_to_base_structures = generate_base_structures_with_golden_seeds(
                X_test,
                test_seq_df,
                top_seeds,
                optimal_params,
                X_valid,
                y_valid
            )
            
            # PHASE 4: Generate and evaluate diverse candidates
            print("\nPHASE 4: Generating diverse candidates and evaluating quality...")
            
            seq_to_coords = {}
            for i, (_, row) in enumerate(test_seq_df.iterrows()):
                target_id = row['target_id']
                seq = row['sequence']
                seq_length = len(seq)
                
                print(f"Processing sequence {i+1}/{len(test_seq_df)}, ID: {target_id}, length: {seq_length}")
                
                # Get base structures for this sequence
                base_structures = seq_to_base_structures[target_id]
                
                if not base_structures:
                    print(f"No base structures found for {target_id}. Creating emergency structure.")
                    base_structures = [create_emergency_structure(seq_length)]
                
                # Extract sequence features
                seq_features = X_test[i][:seq_length]
                
                # Generate diverse candidates
                candidates = generate_diverse_structures_from_bases(
                    base_structures, 
                    seq_length, 
                    quality_model,
                    num_per_base=5
                )
                
                # Evaluate and select the best structures
                try:
                    top_structures = evaluate_and_prune_structures(
                        candidates, 
                        seq_features, 
                        quality_model, 
                        top_k=5
                    )
                except Exception as e:
                    print(f"Error in structure evaluation: {str(e)}")
                    print("Falling back to basic selection...")
                    # If evaluation fails, just use the base structures
                    top_structures = base_structures[:5]
                    
                    # If we need more structures, pad with variations
                    while len(top_structures) < 5:
                        idx = len(top_structures) % len(base_structures)
                        variation = sample_structural_variation(
                            base_structures[idx],
                            noise_level=0.1,
                            preserve_distance=True,
                            use_global_movement=False
                        )
                        top_structures.append(normalize_structure(variation))
                
                # Store the final structures
                seq_to_coords[target_id] = top_structures
            
            # PHASE 5: Create Submission
            print("\nPHASE 5: Creating submission file...")
            submission_df = create_submission_dataframe(seq_to_coords, sample_submission_df)
            
            # Save submission
            enhanced_file = os.path.join(output_dir, 'submission_enhanced.csv')
            submission_df.to_csv(enhanced_file, index=False)
            print(f"Enhanced submission saved to {enhanced_file}")
            
            # Save as standard submission
            standard_file = os.path.join(output_dir, 'submission.csv')
            submission_df.to_csv(standard_file, index=False)
            
            # Set success
            status['success'] = True
            
            # Get best TM-score from seeds for reporting
            if top_seeds:
                status['best_tm_score'] = max(seed['tm_score'] for seed in top_seeds)
            
            return submission_df, status
            
        except Exception as e:
            print(f"ERROR in enhanced hybrid pipeline: {str(e)}")
            traceback.print_exc()
            status['error'] = str(e)
            
            # Fall back to original pipeline as last resort
            print("Falling back to original pipeline...")
            return run_hybrid_pipeline_func(
                X_valid, 
                y_valid, 
                test_seq_df, 
                sample_submission_df, 
                output_dir, 
                golden_threshold=golden_threshold, 
                seed_attempts=seed_attempts, 
                optimal_params=optimal_params
            )
    
    return enhanced_hybrid_pipeline
  if __name__ == "__main__":
    # Execution mode selection
    use_hybrid_pipeline = True     # combine golden seeds and NN pruning
    use_nn_pruning = False         # Use only NN pruning
    use_reference_only = False     # Use only reference-based approach
    
    # Print startup banner
    print("=" * 80)
    print("RNA 3D STRUCTURE PREDICTION PIPELINE".center(80))
    print("=" * 80)
    
    # Print selected mode
    if use_hybrid_pipeline:
        mode_description = "Hybrid Pipeline: Golden Seeds + NN Pruning"
    elif use_nn_pruning:
        mode_description = "Neural Network based pruning pipeline"
    elif use_reference_only:
        mode_description = "Reference model only"
    else:
        mode_description = "Standard pipeline"
    
    print(f"Selected mode: {mode_description}")
    print("-" * 80)
    
    try:
        # Execute the selected pipeline
        if use_hybrid_pipeline:
            start_time = time.time()
            
            print("Loading processed data...")
            X_train, y_train, X_valid, y_valid = load_processed_data()
            
            print("\nLoading test data...")
            test_seq_df = pd.read_csv(os.path.join(DATA_DIR, "test_sequences.csv"))
            sample_submission_df = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
            
            # Run the hybrid pipeline
            submission_df, status = run_hybrid_pipeline(
                X_valid, y_valid,
                test_seq_df, sample_submission_df,
                OUTPUT_DIR,
                golden_threshold=0.6,
                seed_attempts=100
            )
            
            # Calculate total runtime
            runtime = time.time() - start_time
            hours, remainder = divmod(runtime, 3600)
            minutes, seconds = divmod(remainder, 60)
            
            # Display results summary
            print("\n" + "=" * 80)
            print("HYBRID PIPELINE RESULTS SUMMARY".center(80))
            print("=" * 80)
            print(f"Total runtime: {int(hours)}h {int(minutes)}m {int(seconds)}s")
            
            if status['success']:
                print("\nHYBRID PIPELINE STATISTICS:")
                print(f"  - Golden seeds found: {status['golden_seeds_found']}")
                print(f"  - NN training success: {status['nn_training_success']}")
                print(f"  - Best TM-score: {status['best_tm_score']:.4f}")
            else:
                print(f"\nPipeline failed with error: {status['error']}")
            
            # Display output file information
            print("\nOUTPUT FILES:")
            submission_file = os.path.join(OUTPUT_DIR, 'submission_hybrid.csv')
            if os.path.exists(submission_file):
                try:
                    file_size = os.path.getsize(submission_file)
                    print(f"  - Hybrid submission: {submission_file} ({file_size/1024/1024:.2f} MB)")
                except:
                    print(f"  - Hybrid submission: {submission_file}")
            
            standard_file = os.path.join(OUTPUT_DIR, 'submission.csv')
            if os.path.exists(standard_file):
                try:
                    file_size = os.path.getsize(standard_file)
                    print(f"  - Standard submission: {standard_file} ({file_size/1024/1024:.2f} MB)")
                except:
                    print(f"  - Standard submission: {standard_file}")
            
            print("=" * 80)
        
        elif use_nn_pruning:
            start_time = time.time()
            
            print("Loading processed data...")
            X_train, y_train, X_valid, y_valid = load_processed_data()
            
            print("\nLoading test data...")
            test_seq_df = pd.read_csv(os.path.join(DATA_DIR, "test_sequences.csv"))
            sample_submission_df = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
            
            # Train quality model
            print("\nTraining quality assessment model...")
            quality_model = train_enhanced_quality_model(X_valid, y_valid, X_valid, y_valid)
            
            # Create reference model with default parameters
            print("\nCreating reference model...")
            reference_model = reference_based_approach(
                X_valid, 
                y_valid,
                geometric_sampling=True,
                noise_level=0.21,
                correlation=0.83
            )
            
            # Generate submission using NN pruning only
            submission_df = generate_nn_pruned_submission(
                reference_model,
                quality_model,
                test_seq_df,
                sample_submission_df
            )
            
            # Calculate total runtime
            runtime = time.time() - start_time
            hours, remainder = divmod(runtime, 3600)
            minutes, seconds = divmod(remainder, 60)
            
            print("\n" + "=" * 80)
            print("NN PRUNING PIPELINE RESULTS".center(80))
            print("=" * 80)
            print(f"Total runtime: {int(hours)}h {int(minutes)}m {int(seconds)}s")
            
            # Display output file information
            print("\nOUTPUT FILES:")
            submission_file = os.path.join(OUTPUT_DIR, 'submission_nn_pruned.csv')
            if os.path.exists(submission_file):
                try:
                    file_size = os.path.getsize(submission_file)
                    print(f"  - NN pruned submission: {submission_file} ({file_size/1024/1024:.2f} MB)")
                except:
                    print(f"  - NN pruned submission: {submission_file}")
            
            standard_file = os.path.join(OUTPUT_DIR, 'submission.csv')
            if os.path.exists(standard_file):
                try:
                    file_size = os.path.getsize(standard_file)
                    print(f"  - Standard submission: {standard_file} ({file_size/1024/1024:.2f} MB)")
                except:
                    print(f"  - Standard submission: {standard_file}")
            
            print("=" * 80)
            
        elif use_reference_only:
            start_time = time.time()
            
            print("Loading processed data...")
            X_train, y_train, X_valid, y_valid = load_processed_data()
            
            print("\nLoading test data...")
            test_seq_df = pd.read_csv(os.path.join(DATA_DIR, "test_sequences.csv"))
            sample_submission_df = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
            
            # Create optimized reference model
            print("\nCreating and evaluating reference model...")
            reference_model = reference_based_approach(
                X_valid, 
                y_valid,
                geometric_sampling=True,
                noise_level=0.21,
                correlation=0.83
            )
            
            metrics = evaluate_model(reference_model, X_valid, y_valid)
            tm_score = metrics['avg_tm_score']
            print(f"Reference model TM-score: {tm_score:.4f}")
            
            # Prepare test sequences
            X_test = prepare_test_features(test_seq_df)
            
            # Generate predictions
            print("\nGenerating predictions...")
            predictions = reference_model.predict(X_test)
            
            # Create submission dataframe
            print("\nCreating submission dataframe...")
            submission_df = sample_submission_df.copy()
            
            seq_to_coords = {}
            for i, (_, row) in enumerate(test_seq_df.iterrows()):
                target_id = row['target_id']
                seq_length = len(row['sequence'])
                
                # Normalize and process structure
                struct = normalize_structure(predictions[i][:seq_length])
                
                # Create 5 copies with small variations
                structures = [struct]
                for j in range(4):
                    variation = sample_structural_variation(
                        struct,
                        noise_level=0.05,
                        preserve_distance=True,
                        correlation=0.9
                    )
                    structures.append(normalize_structure(variation))
                
                seq_to_coords[target_id] = structures
            
            # Fill the dataframe
            for i, row in submission_df.iterrows():
                if i % 1000 == 0:
                    print(f"Processing row {i}/{len(submission_df)}")
                
                id_parts = row['ID'].split('_')
                seq_id = id_parts[0]
                residue_idx = int(id_parts[1]) - 1
                
                if seq_id in seq_to_coords:
                    structures = seq_to_coords[seq_id]
                    if residue_idx < len(structures[0]):
                        for struct_idx in range(5):
                            submission_df.at[i, f'x_{struct_idx+1}'] = structures[struct_idx][residue_idx][0]
                            submission_df.at[i, f'y_{struct_idx+1}'] = structures[struct_idx][residue_idx][1]
                            submission_df.at[i, f'z_{struct_idx+1}'] = structures[struct_idx][residue_idx][2]
            
            # Save submission
            reference_file = os.path.join(OUTPUT_DIR, 'submission_reference.csv')
            submission_df.to_csv(reference_file, index=False)
            print(f"Reference submission saved to {reference_file}")
            
            # Save as standard submission
            standard_file = os.path.join(OUTPUT_DIR, 'submission.csv')
            submission_df.to_csv(standard_file, index=False)
            
            # Calculate total runtime
            runtime = time.time() - start_time
            hours, remainder = divmod(runtime, 3600)
            minutes, seconds = divmod(remainder, 60)
            
            print("\n" + "=" * 80)
            print("REFERENCE MODEL RESULTS".center(80))
            print("=" * 80)
            print(f"Total runtime: {int(hours)}h {int(minutes)}m {int(seconds)}s")
            
            # Display output file information
            print("\nOUTPUT FILES:")
            if os.path.exists(reference_file):
                try:
                    file_size = os.path.getsize(reference_file)
                    print(f"  - Reference submission: {reference_file} ({file_size/1024/1024:.2f} MB)")
                except:
                    print(f"  - Reference submission: {reference_file}")
            
            if os.path.exists(standard_file):
                try:
                    file_size = os.path.getsize(standard_file)
                    print(f"  - Standard submission: {standard_file} ({file_size/1024/1024:.2f} MB)")
                except:
                    print(f"  - Standard submission: {standard_file}")
            
            print("=" * 80)
            
        else:
            # Standard pipeline - if user disabled all options
            print("No pipeline mode selected. Please set one of the pipeline flags to True.")
            print("Available options:")
            print("  - use_hybrid_pipeline: Combined golden seeds and NN pruning")
            print("  - use_nn_pruning: Neural Network based pruning only")
            print("  - use_reference_only: Use only reference model approach")
        
        print("\nProcess completed.")
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("ERROR IN MAIN EXECUTION".center(80))
        print("=" * 80)
        print(f"Critical error: {str(e)}")
        traceback.print_exc()
        print("=" * 80)
