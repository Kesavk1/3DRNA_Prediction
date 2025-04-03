class EnhancedRNAQualityNN:
    """
    Enhanced Neural Network model for RNA structure quality assessment.
    Features:
    - Handles variable-length RNA sequences
    - Incorporates RNA-specific features
    - Attention mechanism for capturing long-range interactions
    - Multiple evaluation metrics for robust quality assessment
    """
    def __init__(self, max_length=720):
        self.max_length = max_length
        self.is_trained = False
        self.model = None
        self.build_model()
        
    def build_model(self):
        """
        Build an enhanced model architecture for RNA quality assessment.
        """
        # Define the masking layer to handle variable-length sequences
        coord_input = layers.Input(shape=(self.max_length, 3), name='coordinates')
        
        # Create a mask for zero-padded coordinates
        mask_layer = layers.Lambda(
            lambda x: tf.cast(tf.reduce_sum(tf.abs(x), axis=-1) > 0.0, tf.float32),
            output_shape=lambda shape: (shape[0], shape[1])
        )
        mask = mask_layer(coord_input)
        
        # Expandir dimensões
        mask_expanded_layer = layers.Lambda(
            lambda x: tf.expand_dims(x, axis=-1),
            output_shape=lambda shape: (shape[0], shape[1], 1)
        )
        mask_expanded = mask_expanded_layer(mask)  # Shape: (batch, seq_len, 1)
        
        # Optional sequence features input
        seq_input = layers.Input(shape=(self.max_length, 5), name='sequence')
        
        # Definição da função de distância pareada
        def create_pairwise_dist_layer():
            def masked_pairwise_dist_fn(inputs):
                coords, m = inputs
                # Expand dims for broadcasting
                coords1 = tf.expand_dims(coords, 2)
                coords2 = tf.expand_dims(coords, 1)
                
                # Calculate Euclidean distance
                diff = coords1 - coords2
                squared_diff = tf.reduce_sum(tf.square(diff), axis=-1)
                dist = tf.sqrt(squared_diff + 1e-8)
                
                # Create mask for valid pairs
                mask1 = tf.expand_dims(m, 2)
                mask2 = tf.expand_dims(m, 1)
                pair_mask = mask1 * mask2
                
                # Apply mask
                masked_dist = dist * pair_mask
                return masked_dist
            
            return layers.Lambda(
                masked_pairwise_dist_fn,
                output_shape=lambda shape: (shape[0][0], shape[0][1], shape[0][1])
            )
        
        # Aplicar a camada de distância pareada
        pairwise_dist_layer = create_pairwise_dist_layer()
        distances = pairwise_dist_layer([coord_input, mask])
        
        # 1.2 Process distances with 2D convolutions
        dist_features = layers.Reshape((self.max_length, self.max_length, 1))(distances)
        dist_features = layers.Conv2D(16, 3, activation='relu', padding='same')(dist_features)
        dist_features = layers.BatchNormalization()(dist_features)
        dist_features = layers.MaxPooling2D(2)(dist_features)
        
        dist_features = layers.Conv2D(32, 3, activation='relu', padding='same')(dist_features)
        dist_features = layers.BatchNormalization()(dist_features)
        dist_features = layers.MaxPooling2D(2)(dist_features)
        
        # Flatten with adaptive pooling to handle variable lengths
        dist_features = layers.GlobalAveragePooling2D()(dist_features)
        
        # 1.3 Process direct 3D coordinates with 1D convolutions
        # Apply mask to zero out padded positions
        masked_coords = layers.Multiply()([coord_input, mask_expanded])
        
        coord_features = layers.Conv1D(32, 3, activation='relu', padding='same')(masked_coords)
        coord_features = layers.BatchNormalization()(coord_features)
        
        # Para o mecanismo de auto-atenção, criamos as camadas Dense fora da função Lambda
        query_dense = layers.Dense(32)
        key_dense = layers.Dense(32)
        value_dense = layers.Dense(32)
        
        # Função de auto-atenção agora usa camadas pré-definidas
        def create_self_attention_layer(query_dense, key_dense, value_dense):
            def self_attention_fn(inputs):
                x, m = inputs
                # Simple self-attention usando camadas pré-definidas
                query = query_dense(x)
                key = key_dense(x)
                value = value_dense(x)
                
                # Calculate attention scores
                scores = tf.matmul(query, key, transpose_b=True)
                scores = scores / tf.sqrt(32.0)
                
                # Apply mask
                mask1 = tf.expand_dims(m, 2)
                mask2 = tf.expand_dims(m, 1)
                mask_2d = mask1 * mask2
                
                # Very negative number for masked positions (-1e9)
                scores = scores * mask_2d + (1.0 - mask_2d) * (-1e9)
                
                # Apply softmax
                attention_weights = tf.nn.softmax(scores, axis=-1)
                
                # Apply attention
                output = tf.matmul(attention_weights, value)
                
                return output
            
            return layers.Lambda(
                self_attention_fn,
                output_shape=lambda shape: (shape[0][0], shape[0][1], 32)
            )
            
        # Aplicar a camada de auto-atenção
        self_attention_layer = create_self_attention_layer(query_dense, key_dense, value_dense)
        attention_output = self_attention_layer([coord_features, mask])
        
        # Continue processing coordinates
        coord_features = layers.Add()([coord_features, attention_output])  # Residual connection
        coord_features = layers.Conv1D(64, 3, activation='relu', padding='same')(coord_features)
        coord_features = layers.BatchNormalization()(coord_features)
        
        # Global pooling for variable length
        coord_features = layers.GlobalAveragePooling1D()(coord_features)
        
        # 2. Process sequence information (if provided)
        seq_features = layers.Conv1D(32, 3, activation='relu', padding='same')(seq_input)
        seq_features = layers.BatchNormalization()(seq_features)
        seq_features = layers.GlobalAveragePooling1D()(seq_features)
        
        # 3. Calculate RNA-specific features
        
        # 3.1 Extract GC content and other sequence composition features
        def create_sequence_composition_layer():
            def sequence_composition_fn(inputs):
                seq, m = inputs
                # One-hot encoded sequence: (batch, len, 5) [A,C,G,U,N]
                # Calculate GC content
                c_base = seq[:, :, 1]  # C base (index 1)
                g_base = seq[:, :, 2]  # G base (index 2)
                
                # Sum up G and C bases and divide by sequence length
                gc_sum = tf.reduce_sum(c_base * m + g_base * m, axis=1)
                seq_length = tf.reduce_sum(m, axis=1)
                
                # Avoid division by zero
                gc_content = gc_sum / (seq_length + 1e-8)
                
                # Calculate other base contents
                a_base = seq[:, :, 0]  # A base
                u_base = seq[:, :, 3]  # U base
                a_content = tf.reduce_sum(a_base * m, axis=1) / (seq_length + 1e-8)
                u_content = tf.reduce_sum(u_base * m, axis=1) / (seq_length + 1e-8)
                
                # Combine features
                composition = tf.stack([gc_content, a_content, u_content], axis=1)
                
                return composition
            
            return layers.Lambda(
                sequence_composition_fn,
                output_shape=lambda shape: (shape[0][0], 3)
            )
        
        # Aplicar a camada de composição de sequência
        seq_composition_layer = create_sequence_composition_layer()
        seq_composition = seq_composition_layer([seq_input, mask])
        
        # 3.2 Calculate basic structural features
        def create_structural_features_layer():
            def structural_features_fn(inputs):
                coords, m = inputs
                # Calculate average bond length
                coords1 = coords[:, :-1, :]
                coords2 = coords[:, 1:, :]
                
                # Create mask for valid pairs
                mask_bonds = m[:, :-1] * m[:, 1:]
                mask_bonds_expanded = tf.expand_dims(mask_bonds, -1)
                
                # Calculate bond vectors and lengths
                bonds = coords2 - coords1
                masked_bonds = bonds * mask_bonds_expanded
                
                # Euclidean distance
                bond_lengths = tf.sqrt(tf.reduce_sum(tf.square(masked_bonds), axis=-1) + 1e-8)
                
                # Average bond length
                total_bonds = tf.reduce_sum(mask_bonds, axis=1)
                avg_bond_length = tf.reduce_sum(bond_lengths, axis=1) / (total_bonds + 1e-8)
                
                # Bond length consistency (std dev)
                mean_bond = tf.expand_dims(avg_bond_length, -1)
                squared_diff = tf.square(bond_lengths - mean_bond) * mask_bonds
                bond_var = tf.reduce_sum(squared_diff, axis=1) / (total_bonds + 1e-8)
                bond_std = tf.sqrt(bond_var + 1e-8)
                
                # Combine features
                struct_features = tf.stack([avg_bond_length, bond_std], axis=1)
                
                return struct_features
            
            return layers.Lambda(
                structural_features_fn,
                output_shape=lambda shape: (shape[0][0], 2)
            )
        
        # Aplicar a camada de características estruturais
        struct_features_layer = create_structural_features_layer()
        struct_features = struct_features_layer([coord_input, mask])
        
        # 4. Combine all features
        combined = layers.Concatenate()([
            dist_features,      # Pairwise distance features
            coord_features,     # Direct coordinate features
            seq_features,       # Sequence features
            seq_composition,    # GC content, etc.
            struct_features     # Basic structural features
        ])
        
        # 5. Final processing with dense layers
        x = layers.Dense(128, activation='relu')(combined)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # 6. Multiple output heads for different aspects of quality
        quality_score = layers.Dense(1, activation='sigmoid', name='quality_score')(x)
        bond_score = layers.Dense(1, activation='sigmoid', name='bond_score')(x)
        valid_score = layers.Dense(1, activation='sigmoid', name='valid_score')(x)
        
        # Create the model
        self.model = models.Model(
            inputs=[coord_input, seq_input],
            outputs=[quality_score, bond_score, valid_score]
        )
        
        # Compile with weighted losses to emphasize the overall quality score
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=1e-4, clipnorm=1.0),  # Add gradient clipping
            loss={
                'quality_score': 'mean_squared_error',
                'bond_score': 'mean_squared_error',
                'valid_score': 'binary_crossentropy'
            },
            loss_weights={
                'quality_score': 1.0,     # Primary loss
                'bond_score': 0.3,        # Secondary loss
                'valid_score': 0.3        # Secondary loss
            },
            metrics={
                'quality_score': ['mae', 'mse'],
                'bond_score': ['mae'],
                'valid_score': ['accuracy']
            }
        )
    
    # Os métodos train, predict_quality, save_model e load_model permanecem os mesmos
    def train(self, X_train_coords, X_train_seq, y_train, 
              validation_data=None, epochs=50, batch_size=16):
        """
        Train the model with multiple outputs.
    
        Parameters:
        -----------
        X_train_coords: Coordinate inputs (batch, seq_len, 3)
        X_train_seq: Sequence inputs (batch, seq_len, 5)
        y_train: Dictionary with 'quality_score', 'bond_score', and 'valid_score' outputs
        validation_data: Optional validation data in the same format
        """
        # Define callbacks
        callbacks = [
            # Early stopping on the primary output - com mode='min' para métricas de perda
            EarlyStopping(
                monitor='val_quality_score_loss' if validation_data else 'quality_score_loss',
                mode='min',  # Explicitamente indica que queremos minimizar a perda
                patience=10,
                restore_best_weights=True
            ),
            # Custom callback to detect and handle NaN values
            tf.keras.callbacks.TerminateOnNaN()
        ]
    
        # Train the model
        history = self.model.fit(
            x=[X_train_coords, X_train_seq],
            y=y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
    
        self.is_trained = True
        return history
    
    def predict_quality(self, X_coords, X_seq):
        """
        Predict quality scores for RNA structures.
    
        Parameters:
        -----------
        X_coords: Coordinate inputs (batch, seq_len, 3)
        X_seq: Sequence inputs (batch, seq_len, 5) ou (seq_len, 5) que será expandido
    
        Returns:
        --------
        Primary quality score predictions (0-1)
        """
        if not self.is_trained:
            print("WARNING: Model has not been trained yet!")
            return None
    
        # Handle potential shape issues
        batch_size = X_coords.shape[0]
        seq_len = X_coords.shape[1]
    
        # Ensure X_seq has 3 dimensions (batch, seq_len, features)
        if len(X_seq.shape) == 2:  # Se for (seq_len, features)
            X_seq = np.expand_dims(X_seq, axis=0)  # Adicionar dimensão de batch
            X_seq = np.repeat(X_seq, batch_size, axis=0)  # Replicar para todos os exemplos de batch
    
        # Ensure correct format for coordinates
        if seq_len > self.max_length:
            print(f"WARNING: Input sequence length ({seq_len}) exceeds model's maximum length ({self.max_length}).")
            print("Truncating input sequence to maximum length.")
            X_coords = X_coords[:, :self.max_length, :]
        elif seq_len < self.max_length:
            print(f"Padding input sequence from length {seq_len} to {self.max_length}")
            padding = np.zeros((batch_size, self.max_length - seq_len, 3))
            X_coords = np.concatenate([X_coords, padding], axis=1)
    
        # Ensure correct format for sequence
        if X_seq is None:
            # If no sequence provided, create zero array
            X_seq = np.zeros((batch_size, self.max_length, 5))
        else:
            seq_shape = X_seq.shape
            if seq_shape[1] > self.max_length:
                X_seq = X_seq[:, :self.max_length, :]
            elif seq_shape[1] < self.max_length:
                padding = np.zeros((batch_size, self.max_length - seq_shape[1], 5))
                X_seq = np.concatenate([X_seq, padding], axis=1)
    
        # Predict all outputs
        outputs = self.model.predict([X_coords, X_seq])
    
        # Return the primary quality score
        return outputs[0]  # quality_score output
    
    def save_model(self, filepath):
        """Save the model to disk"""
        if self.is_trained:
            self.model.save(filepath)
        else:
            print("WARNING: Cannot save untrained model")
    
    def load_model(self, filepath):
        """Load a pre-trained model from disk"""
        self.model = models.load_model(filepath)
        self.is_trained = True
def prepare_multi_output_targets(train_coords, train_scores):
    """
    Prepare multi-output target values from TM-scores.
    
    Parameters:
    -----------
    train_coords: Training coordinate data
    train_scores: TM-score values (overall quality)
    
    Returns:
    --------
    Dictionary with multiple output targets
    """
    batch_size = len(train_scores)
    
    # Initialize targets dictionary
    targets = {
        'quality_score': train_scores,
        'bond_score': np.zeros((batch_size, 1)),
        'valid_score': np.zeros((batch_size, 1))
    }
    
    # Calculate bond scores and validity scores for each structure
    for i in range(batch_size):
        coords = train_coords[i]
        
        # Calculate bond score (based on ideal bond length)
        valid_mask = ~np.all(coords == 0, axis=1)
        valid_coords = coords[valid_mask]
        
        # Skip if no valid coordinates
        if len(valid_coords) < 3:
            targets['bond_score'][i] = 0.5  # Neutral score
            targets['valid_score'][i] = 0  # Invalid
            continue
        
        # Calculate bond lengths
        bond_lengths = []
        for j in range(1, len(valid_coords)):
            dist = np.linalg.norm(valid_coords[j] - valid_coords[j-1])
            bond_lengths.append(dist)
        
        avg_bond_length = np.mean(bond_lengths)
        bond_std = np.std(bond_lengths)
        
        # Score based on how close to ideal RNA bond length (3.8Å)
        bond_score = 1.0 - min(1.0, abs(avg_bond_length - 3.8) / 3.8)
        targets['bond_score'][i] = bond_score
        
        # Validity score (binary)
        is_valid = check_structure_validity(coords)
        targets['valid_score'][i] = 1 if is_valid else 0
    
    return targets

def train_enhanced_quality_model(X_train, y_train, X_valid, y_valid):
    """
    Train an enhanced RNA quality assessment model.
    
    Parameters:
    -----------
    X_train, X_valid: One-hot encoded RNA sequences
    y_train, y_valid: True 3D coordinates
    
    Returns:
    --------
    Trained EnhancedRNAQualityNN model
    """
    print("Training enhanced RNA quality assessment model...")
    
    # First, determine maximum sequence length in the data
    max_train_len = max(np.sum(~np.all(X_train[i] == 0, axis=1)) for i in range(len(X_train)))
    max_valid_len = max(np.sum(~np.all(X_valid[i] == 0, axis=1)) for i in range(len(X_valid)))
    max_length = max(max_train_len, max_valid_len)
    
    print(f"Maximum sequence length in data: {max_length}")
    
    # Adjust max_length to a reasonable value (for memory efficiency)
    max_length = min(max_length, 720)  # Cap at 720 if larger
    
    # Generate training data with structure variations
    print("Generating training data with structure variations...")
    
    # Parameters for data generation
    num_variations = 10  # Generate 10 variations for each structure
    
    # Containers for training data
    train_seqs = []
    train_coords = []
    train_scores = []
    
    # Process training structures
    for i in range(min(len(X_train), 50)):  # Limit to 50 training examples
        print(f"Processing training structure {i+1}/{min(len(X_train), 50)}")
        seq_features = X_train[i]
        true_coords = y_train[i]
        
        # Check for NaN in true coordinates
        if np.isnan(true_coords).any():
            print(f"Skipping structure {i} due to NaN in true coordinates")
            continue
        
        # Add the true structure (highest quality)
        train_seqs.append(seq_features)
        train_coords.append(true_coords)
        train_scores.append(1.0)  # Perfect score for true structure
        
        # Generate variations with different qualities
        for j in range(num_variations):
            # Vary noise level to get different quality structures
            noise_level = 0.05 + (j * 0.05)  # Smaller steps for better distribution
            try:
                variation = sample_structural_variation(
                    true_coords, 
                    noise_level=noise_level,
                    preserve_distance=True,  # Always preserve distances for stability
                    use_global_movement=(j % 3 == 0)  # Mix of global and local movements
                )
                
                # Check for NaN or Inf in variation
                if np.isnan(variation).any() or np.isinf(variation).any():
                    print(f"Skipping variation {j} for structure {i} due to NaN/Inf")
                    continue
                
                # Calculate TM-score as ground truth quality
                tm_score = calculate_tm_score(variation, true_coords)
                
                # Check if score is valid
                if np.isnan(tm_score) or np.isinf(tm_score) or tm_score <= 0:
                    print(f"Skipping variation {j} for structure {i} due to invalid TM-score: {tm_score}")
                    continue
                
                # Apply additional normalization for stability
                normalized_variation = normalize_coordinates(variation.reshape(1, -1, 3))[0]
                
                train_seqs.append(seq_features)
                train_coords.append(normalized_variation)
                train_scores.append(tm_score)
            except Exception as e:
                print(f"Error generating variation {j} for structure {i}: {str(e)}")
                continue
    
    # Create a smaller validation set for speed and stability
    valid_seqs = []
    valid_coords = []
    valid_scores = []
    
    for i in range(min(len(X_valid), 10)):  # Use only 10 validation examples
        print(f"Processing validation structure {i+1}/{min(len(X_valid), 10)}")
        seq_features = X_valid[i]
        true_coords = y_valid[i]
        
        # Check for NaN in true coordinates
        if np.isnan(true_coords).any():
            print(f"Skipping validation structure {i} due to NaN in true coordinates")
            continue
        
        # Add the true structure
        valid_seqs.append(seq_features)
        valid_coords.append(true_coords)
        valid_scores.append(1.0)
        
        # Generate just 3 variations for validation
        for j in range(3):
            noise_level = 0.05 + (j * 0.1)
            try:
                variation = sample_structural_variation(
                    true_coords, 
                    noise_level=noise_level,
                    preserve_distance=True,
                    use_global_movement=(j % 2 == 0)
                )
                
                # Check for NaN or Inf
                if np.isnan(variation).any() or np.isinf(variation).any():
                    print(f"Skipping validation variation {j} for structure {i} due to NaN/Inf")
                    continue
                
                tm_score = calculate_tm_score(variation, true_coords)
                
                # Check if score is valid
                if np.isnan(tm_score) or np.isinf(tm_score) or tm_score <= 0:
                    print(f"Skipping validation variation {j} for structure {i} due to invalid TM-score: {tm_score}")
                    continue
                
                # Apply additional normalization
                normalized_variation = normalize_coordinates(variation.reshape(1, -1, 3))[0]
                
                valid_seqs.append(seq_features)
                valid_coords.append(normalized_variation)
                valid_scores.append(tm_score)
            except Exception as e:
                print(f"Error generating validation variation {j} for structure {i}: {str(e)}")
                continue
    
    # Convert to numpy arrays and handle potential issues
    train_seqs = np.array(train_seqs)
    train_coords = np.array(train_coords)
    train_scores = np.array(train_scores).reshape(-1, 1)  # Reshape to (n, 1)
    
    valid_seqs = np.array(valid_seqs)
    valid_coords = np.array(valid_coords)
    valid_scores = np.array(valid_scores).reshape(-1, 1)  # Reshape to (n, 1)
    
    # Verify data quality and apply additional cleaning
    train_coords = np.nan_to_num(train_coords, nan=0.0, posinf=0.0, neginf=0.0)
    train_scores = np.clip(train_scores, 0.0, 1.0)  # Ensure scores are in [0, 1]
    
    valid_coords = np.nan_to_num(valid_coords, nan=0.0, posinf=0.0, neginf=0.0)
    valid_scores = np.clip(valid_scores, 0.0, 1.0)
    
    # Log data statistics for debugging
    print(f"Training data: {len(train_scores)} structures")
    print(f"Train coords shape: {train_coords.shape}, train scores shape: {train_scores.shape}")
    print(f"Train coords range: [{np.min(train_coords)}, {np.max(train_coords)}]")
    print(f"Train scores range: [{np.min(train_scores)}, {np.max(train_scores)}]")
    
    print(f"Validation data: {len(valid_scores)} structures")
    
    try:
        # Prepare multi-output targets
        print("Preparing multi-output training targets...")
        train_targets = prepare_multi_output_targets(train_coords, train_scores)
        valid_targets = prepare_multi_output_targets(valid_coords, valid_scores)
        
        # Create and train the enhanced model
        print("Creating and training enhanced model...")
        model = EnhancedRNAQualityNN(max_length=max_length)
        
        # Train the model
        history = model.train(
            X_train_coords=train_coords,
            X_train_seq=train_seqs,
            y_train=train_targets,
            validation_data=([valid_coords, valid_seqs], valid_targets),
            epochs=30,
            batch_size=16
        )
        
        # Validate the model
        print("Validating model...")
        val_predictions = model.predict_quality(valid_coords, valid_seqs)
        val_predictions = val_predictions.flatten()
        
        # Calculate correlation between predicted and true scores
        correlation = np.corrcoef(val_predictions, valid_scores.flatten())[0, 1]
        mae = np.mean(np.abs(val_predictions - valid_scores.flatten()))
        
        print(f"Validation results:")
        print(f"Correlation: {correlation:.4f}")
        print(f"MAE: {mae:.4f}")
        
        # Save the model
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        model.save_model(os.path.join(OUTPUT_DIR, 'enhanced_rna_quality_model.h5'))
        
        return model
        
    except Exception as e:
        print(f"Error training enhanced model: {str(e)}")
        traceback.print_exc()
        
        # Fall back to a simpler model or rule-based approach
        print("Falling back to a simplified model due to training error...")
        return create_rule_based_model()

def create_rule_based_model():
   """
   Create a rule-based quality assessment model as fallback.
   """
   class RuleBasedQualityModel:
       def __init__(self):
           self.is_trained = True
           
       def predict_quality(self, X_coords, X_seq=None):
           batch_size = X_coords.shape[0]
           
           # Implement a comprehensive rule-based quality metric
           scores = []
           
           for i in range(batch_size):
               # Check for valid coordinates
               valid_mask = ~np.all(X_coords[i] == 0, axis=1)
               coords = X_coords[i][valid_mask]
               
               if len(coords) < 3:
                   scores.append(0.5)  # Default score for very short structures
                   continue
               
               # 1. Calculate bond lengths
               bond_lengths = []
               for j in range(1, len(coords)):
                   dist = np.linalg.norm(coords[j] - coords[j-1])
                   bond_lengths.append(dist)
               
               avg_bond_length = np.mean(bond_lengths)
               bond_std = np.std(bond_lengths)
               
               # 2. Score based on how close to ideal RNA bond length
               bond_score = 1.0 - min(1.0, abs(avg_bond_length - 3.8) / 3.8)
               
               # 3. Bond consistency score
               consistency_score = 1.0 - min(1.0, bond_std / 1.5)
               
               # 4. Check structure validity
               is_valid = check_structure_validity(coords)
               valid_score = 1.0 if is_valid else 0.5
               
               # 5. Check for extreme compression or expansion
               min_bond = min(bond_lengths) if bond_lengths else 0
               max_bond = max(bond_lengths) if bond_lengths else 0
               compression_score = 1.0
               if min_bond < 1.0 or max_bond > 10.0:  # Physical constraints for RNA
                   compression_score = 0.7
               
               # 6. Analyze radius of gyration (compactness)
               center = np.mean(coords, axis=0)
               distances = np.sqrt(np.sum((coords - center) ** 2, axis=1))
               radius_gyration = np.mean(distances)
               
               # Typical radius of gyration for RNA scales with sequence length (approximate)
               expected_radius = 3.0 * np.power(len(coords), 1/3)  # Simple scaling law
               compactness_score = 1.0 - min(1.0, abs(radius_gyration - expected_radius) / expected_radius)
               
               # 7. Combined score
               final_score = (
                   0.3 * bond_score + 
                   0.2 * consistency_score + 
                   0.2 * valid_score + 
                   0.15 * compression_score + 
                   0.15 * compactness_score
               )
               
               # Ensure score is in range [0, 1]
               final_score = min(1.0, max(0.0, final_score))
               
               scores.append(final_score)
           
           return np.array(scores).reshape(-1, 1)
       
       def save_model(self, filepath):
           # Nothing to save for rule-based model
           pass
   
   return RuleBasedQualityModel()

def evaluate_and_compare_models(quality_model, rule_model, X_valid, y_valid):
   """
   Evaluate and compare different quality assessment models.
   
   Parameters:
   -----------
   quality_model: Trained neural network model
   rule_model: Rule-based model
   X_valid, y_valid: Validation data
   
   Returns:
   --------
   Dictionary with evaluation metrics
   """
   print("Evaluating and comparing quality assessment models...")
   
   # Create validation data with multiple quality levels
   print("Generating validation structures with different quality levels...")
   
   # Containers for validation data
   val_seqs = []
   val_coords = []
   val_scores = []
   
   # Number of samples to generate per structure
   num_samples = 5
   
   # Generate validation data
   for i in range(min(10, len(X_valid))):
       seq_features = X_valid[i]
       true_coords = y_valid[i]
       
       # Skip structures with NaN
       if np.isnan(true_coords).any():
           continue
           
       # Add the true structure
       val_seqs.append(seq_features)
       val_coords.append(true_coords)
       val_scores.append(1.0)
       
       # Generate variations with different quality levels
       for j in range(num_samples):
           noise_level = 0.1 * (j + 1)  # Increasing noise
           
           try:
               variation = sample_structural_variation(
                   true_coords,
                   noise_level=noise_level,
                   preserve_distance=(j % 2 == 0),
                   use_global_movement=(j % 3 == 0)
               )
               
               # Skip invalid variations
               if np.isnan(variation).any() or np.isinf(variation).any():
                   continue
                   
               # Calculate TM-score
               tm_score = calculate_tm_score(variation, true_coords)
               
               # Skip invalid scores
               if np.isnan(tm_score) or np.isinf(tm_score) or tm_score <= 0:
                   continue
                   
               val_seqs.append(seq_features)
               val_coords.append(variation)
               val_scores.append(tm_score)
               
           except Exception as e:
               print(f"Error generating validation variation: {str(e)}")
               continue
   
   # Convert to numpy arrays
   val_coords = np.array(val_coords)
   val_seqs = np.array(val_seqs)
   val_scores = np.array(val_scores).reshape(-1, 1)
   
   print(f"Validation data: {len(val_scores)} structures")
   
   # Evaluate neural network model
   nn_predictions = None
   try:
       print("Evaluating neural network model...")
       nn_predictions = quality_model.predict_quality(val_coords, val_seqs)
       nn_correlation = np.corrcoef(nn_predictions.flatten(), val_scores.flatten())[0, 1]
       nn_mae = np.mean(np.abs(nn_predictions.flatten() - val_scores.flatten()))
       
       print(f"Neural network model - Correlation: {nn_correlation:.4f}, MAE: {nn_mae:.4f}")
   except Exception as e:
       print(f"Error evaluating neural network model: {str(e)}")
       nn_correlation = 0.0
       nn_mae = float('inf')
   
   # Evaluate rule-based model
   rule_predictions = None
   try:
       print("Evaluating rule-based model...")
       rule_predictions = rule_model.predict_quality(val_coords)
       rule_correlation = np.corrcoef(rule_predictions.flatten(), val_scores.flatten())[0, 1]
       rule_mae = np.mean(np.abs(rule_predictions.flatten() - val_scores.flatten()))
       
       print(f"Rule-based model - Correlation: {rule_correlation:.4f}, MAE: {rule_mae:.4f}")
   except Exception as e:
       print(f"Error evaluating rule-based model: {str(e)}")
       rule_correlation = 0.0
       rule_mae = float('inf')
   
   # Determine the best model
   if nn_correlation > rule_correlation:
       print("Neural network model performs better")
       best_model = "neural_network"
   else:
       print("Rule-based model performs better")
       best_model = "rule_based"
   
   return {
       'neural_network': {
           'correlation': nn_correlation,
           'mae': nn_mae,
           'predictions': nn_predictions
       },
       'rule_based': {
           'correlation': rule_correlation,
           'mae': rule_mae,
           'predictions': rule_predictions
       },
       'best_model': best_model,
       'validation_data': {
           'coords': val_coords,
           'scores': val_scores
       }
   }
