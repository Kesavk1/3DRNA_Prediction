def sample_structural_variation(coords, noise_level=0.5, preserve_distance=True, 
                               use_global_movement=False, correlation=0.7):
    """
    Enhanced version of structural variation sampling with better
    handling of large RNAs and improved noise distribution.
    """
    new_coords = coords.copy()
    valid_mask = ~np.all(coords == 0, axis=1)
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) < 3:
        return new_coords
    
    # Parameters optimized for RNA structure
    typical_bond_length = 3.8  # Angstroms - typical RNA backbone distance
    
    # Add global domain movements if requested
    if use_global_movement and len(valid_indices) > 20:
        # More natural domain identification - try to find natural hinge points
        # For RNA, these often occur at junctions between helices
        
        # Calculate distance between consecutive residues as a heuristic
        # for finding potential hinge points (larger distances often indicate junctions)
        distances = []
        for i in range(1, len(valid_indices)):
            idx1 = valid_indices[i-1]
            idx2 = valid_indices[i]
            dist = np.linalg.norm(coords[idx1] - coords[idx2])
            distances.append((i, dist))
        
        # Sort by distance to find potential hinges
        distances.sort(key=lambda x: x[1], reverse=True)
        
        # Take top 2 potential hinge points (if we have enough points)
        num_hinges = min(2, len(distances)//3)
        
        for h in range(num_hinges):
            if h < len(distances):
                hinge_point = distances[h][0]
                if hinge_point < 5 or hinge_point > len(valid_indices) - 5:
                    continue
                    
                hinge_idx = valid_indices[hinge_point]
                
                # Angle of rotation with natural distribution
                # More small movements than large ones
                angle = np.random.exponential(0.2)  # Mostly small angles with occasional larger ones
                if np.random.random() < 0.5:
                    angle = -angle  # Allow both directions
                
                # Create a more natural rotation matrix with slight 3D component
                # RNAs often bend and twist in 3D
                sin_a, cos_a = np.sin(angle), np.cos(angle)
                tilt = np.random.normal(0, 0.1)  # Small tilt in 3D
                rotation_matrix = np.array([
                    [cos_a, -sin_a, 0],
                    [sin_a, cos_a, tilt],
                    [0, -tilt, 1]
                ])
                
                # Apply rotation around hinge point
                ref_point = new_coords[hinge_idx]
                for i in valid_indices[hinge_point+1:]:
                    vector = new_coords[i] - ref_point
                    rotated = np.dot(vector, rotation_matrix)
                    new_coords[i] = ref_point + rotated
    
    # Propagate variation residue by residue, with correlation
    # RNA has strong local correlations in structure
    prev_noise = np.zeros(3)
    
    correlation = 0.5  # High correlation for smoother variations
    
    for i in range(1, len(coords)):
        if not valid_mask[i] or not valid_mask[i-1]:
            continue
            
        vec = new_coords[i-1] - new_coords[i]
        vec_length = np.linalg.norm(vec)
        
        # Generate correlated noise (smoother transitions)
        new_noise = np.random.normal(0, noise_level, size=3)
        noise_vec = correlation * prev_noise + (1 - correlation) * new_noise
        prev_noise = noise_vec.copy()
        
        noise_norm = np.linalg.norm(noise_vec)
        if noise_norm > 0:
            # Scale noise proportionally
            noise_vec = noise_vec / noise_norm * (noise_level * vec_length)
        
        # Add noise to the direction
        new_vec = vec + noise_vec
        
        # Preserve distance if requested
        if preserve_distance:
            current_length = np.linalg.norm(new_vec)
            if current_length > 0:
                # Allow slight variation in bond length (RNA is not rigid)
                target_length = typical_bond_length * (1 + np.random.normal(0, 0.05))
                new_vec = new_vec / current_length * target_length
        
        new_coords[i] = new_coords[i-1] - new_vec
    
    return new_coords

def get_rotation_matrix(axis, theta):
    """
    Return the rotation matrix for rotation around an arbitrary axis.
    
    Parameters:
    -----------
    axis: Unit vector defining the rotation axis
    theta: Rotation angle in radians
    
    Returns:
    --------
    3x3 rotation matrix
    """
    # Ensure axis is a unit vector
    axis = axis / np.linalg.norm(axis)
    
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    
    return np.array([
        [a*a + b*b - c*c - d*d, 2*(b*c - a*d), 2*(b*d + a*c)],
        [2*(b*c + a*d), a*a + c*c - b*b - d*d, 2*(c*d - a*b)],
        [2*(b*d - a*c), 2*(c*d + a*b), a*a + d*d - b*b - c*c]
    ])

def refine_rna_backbone(structure):
    """
    Refine the RNA backbone geometry to match known constraints.
    
    Parameters:
    -----------
    structure: RNA 3D structure
    
    Returns:
    --------
    Refined structure
    """
    # Create a copy to refine
    refined = structure.copy()
    
    # Check for valid residues
    valid_mask = ~np.all(refined == 0, axis=1)
    
    # Apply RNA-specific backbone constraints
    for i in range(2, len(refined)):
        if valid_mask[i] and valid_mask[i-1] and valid_mask[i-2]:
            # In RNA, there are constraints on three consecutive backbone atoms
            
            # Get the two backbone vectors
            vec1 = refined[i-1] - refined[i-2]
            vec2 = refined[i] - refined[i-1]
            
            # Calculate current angle between vectors
            vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-6)
            vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-6)
            cos_angle = np.dot(vec1_norm, vec2_norm)
            
            # Clamp to valid range for numerical stability
            cos_angle = max(-1.0, min(1.0, cos_angle))
            angle = np.arccos(cos_angle)
            
            # In RNA, the typical backbone angle is around 100-120 degrees
            ideal_angle = np.radians(110)
            
            # If the angle is too far from ideal, adjust it
            if abs(angle - ideal_angle) > np.radians(30):
                # Create a rotation to adjust the angle
                # Get the rotation axis (perpendicular to the plane of vec1 and vec2)
                axis = np.cross(vec1_norm, vec2_norm)
                axis_norm = axis / (np.linalg.norm(axis) + 1e-6)
                
                # Determine rotation angle to reach ideal angle
                angle_diff = ideal_angle - angle
                
                # Apply rotation to vec2
                rotation_matrix = get_rotation_matrix(axis_norm, angle_diff)
                new_vec2 = np.dot(rotation_matrix, vec2_norm) * np.linalg.norm(vec2)
                
                # Update the position
                refined[i] = refined[i-1] + new_vec2
    
    return refined
