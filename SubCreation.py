def create_submission_dataframe(seq_to_coords, sample_submission_df):
   """
   Create a submission DataFrame from the final structures.
   
   Parameters:
   -----------
   seq_to_coords: Dictionary mapping sequence IDs to lists of structures
   sample_submission_df: Sample submission format
   
   Returns:
   --------
   Submission DataFrame
   """
   # Create a copy of the sample submission
   submission_df = sample_submission_df.copy()
   
   # Fill in the coordinates for each structure
   for i, row in submission_df.iterrows():
       if i % 1000 == 0:
           print(f"Processing row {i}/{len(submission_df)}")
       
       # Parse the ID to get sequence ID and residue index
       id_parts = row['ID'].split('_')
       seq_id = id_parts[0]
       residue_idx = int(id_parts[1]) - 1  # Convert to 0-based indexing
       
       # Check if we have structures for this sequence
       if seq_id in seq_to_coords:
           structures = seq_to_coords[seq_id]
           
           # Check if the residue index is valid
           if residue_idx < len(structures[0]):
               # Fill in coordinates for all 5 structures
               for struct_idx in range(5):
                   if struct_idx < len(structures):
                       submission_df.at[i, f'x_{struct_idx+1}'] = structures[struct_idx][residue_idx][0]
                       submission_df.at[i, f'y_{struct_idx+1}'] = structures[struct_idx][residue_idx][1]
                       submission_df.at[i, f'z_{struct_idx+1}'] = structures[struct_idx][residue_idx][2]
                   else:
                       # If we have fewer than 5 structures, duplicate the last one
                       last_idx = len(structures) - 1
                       submission_df.at[i, f'x_{struct_idx+1}'] = structures[last_idx][residue_idx][0]
                       submission_df.at[i, f'y_{struct_idx+1}'] = structures[last_idx][residue_idx][1]
                       submission_df.at[i, f'z_{struct_idx+1}'] = structures[last_idx][residue_idx][2]
   
   return submission_df

def generate_nn_pruned_submission(model, quality_model, test_seq_df, sample_submission_df):
    """
    Enhanced submission generation that uses NN-based pruning for structure selection.
    """
    print("Generating submission with Neural Network pruning...")
    
    # Prepare test features
    X_test = prepare_test_features(test_seq_df)
    
    # Generate multiple predictions for ensemble diversity
    print("Generating base predictions...")
    base_predictions = model.predict(X_test)
    
    seq_to_coords = {}
    for i, (_, row) in enumerate(test_seq_df.iterrows()):
        target_id = row['target_id']
        seq = row['sequence']
        seq_length = len(seq)
        
        print(f"Processing sequence {i+1}/{len(test_seq_df)}, ID: {target_id}, length: {seq_length}")
        
        # Get base coordinates
        base_coords = base_predictions[i][:seq_length]
        
        # Extract sequence features for this RNA
        seq_features = X_test[i][:seq_length]
        
        # Generate and prune structures using the NN model
        structures = generate_and_prune_structures(
            base_coords, 
            seq_features, 
            quality_model,
            num_candidates=30,  # Generate more candidates
            top_k=5             # Keep top 5 for submission
        )
        
        # Store the structures
        seq_to_coords[target_id] = structures
    
    # Create submission DataFrame
    print("Creating submission file...")
    submission_df = sample_submission_df.copy()
    
    for i, row in submission_df.iterrows():
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
    
    submission_file = os.path.join(OUTPUT_DIR, 'submission_nn_pruned.csv')
    submission_df.to_csv(submission_file, index=False)
    print(f"NN-pruned submission file saved to {submission_file}")
    
    # Also save as standard submission
    standard_file = os.path.join(OUTPUT_DIR, 'submission.csv')
    submission_df.to_csv(standard_file, index=False)
    
    return submission_df
