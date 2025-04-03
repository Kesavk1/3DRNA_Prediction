ef repair_invalid_structure(structure):
    """
    Attempt to repair an invalid RNA structure.
    
    Parameters:
    -----------
    structure: Potentially invalid RNA structure
    
    Returns:
    --------
    Repaired structure
    """
    # Create a copy to repair
    repaired = structure.copy()
    
    # Check for valid residues
    valid_mask = ~np.all(repaired == 0, axis=1)
    
    # Fix bond lengths
    for i in range(1, len(repaired)):
        if valid_mask[i] and valid_mask[i-1]:
            # Get current bond
            bond_vector = repaired[i] - repaired[i-1]
            bond_length = np.linalg.norm(bond_vector)
            
            # Check if bond is too short or too long
            if bond_length < 1.0 or bond_length > 7.0:
                # Fix bond to ideal length
                ideal_length = 3.8
                if bond_length > 0:
                    repaired[i] = repaired[i-1] + (bond_vector / bond_length) * ideal_length
                else:
                    # Generate a random direction if bond length is zero
                    random_direction = np.random.randn(3)
                    random_direction = random_direction / np.linalg.norm(random_direction)
                    repaired[i] = repaired[i-1] + random_direction * ideal_length
    
    # Check for clashes (atoms too close to each other)
    for i in range(len(repaired)):
        if valid_mask[i]:
            for j in range(i+3, len(repaired)):  # Skip adjacent residues
                if valid_mask[j]:
                    # Calculate distance
                    distance = np.linalg.norm(repaired[j] - repaired[i])
                    
                    # If atoms are too close
                    if distance < 1.0:
                        # Move one atom away slightly in a random direction
                        random_direction = np.random.randn(3)
                        random_direction = random_direction / np.linalg.norm(random_direction)
                        repaired[j] = repaired[i] + random_direction * 4.0  # Place at safe distance
    
    # Final normalization
    repaired = normalize_structure(repaired)
    
    return repaired

def create_emergency_structure(seq_length):
    """
    Create an emergency structure when all else fails.
    Generates a physically plausible RNA structure.
    
    Parameters:
    -----------
    seq_length: Length of the RNA sequence
    
    Returns:
    --------
    Basic RNA structure
    """
    # Create a simple linear structure as fallback
    emergency_structure = np.zeros((seq_length, 3))
    
    # Define canonical nucleotide step (3.8Å)
    step = np.array([3.8, 0.0, 0.0])
    
    # Generate a straight chain with some randomness
    for i in range(seq_length):
        if i == 0:
            emergency_structure[i] = np.zeros(3)
        else:
            # Add slight random deviation to prevent perfect linearity
            random_noise = np.random.normal(0, 0.2, 3)
            emergency_structure[i] = emergency_structure[i-1] + step + random_noise
    
    # Add a slight curve to make it more RNA-like
    # Apply a gentle curve in the y-z plane
    for i in range(seq_length):
        angle = i * 0.1  # Gradual rotation
        emergency_structure[i, 1] += 2 * np.sin(angle)  # Y-component
        emergency_structure[i, 2] += 2 * np.cos(angle)  # Z-component
    
    # Normalize
    emergency_structure = normalize_structure(emergency_structure)
    
    return emergency_structure
