def identify_stem_loops(sequence):
    """
    Simple function to identify potential stem-loop regions in RNA.
    
    Parameters:
    -----------
    sequence: RNA sequence
    
    Returns:
    --------
    List of (start, end) indices for potential stem loops
    """
    # This is a simplified implementation
    # A real implementation would use a more sophisticated algorithm
    
    stem_loops = []
    min_stem_length = 3
    
    # Look for complementary regions that could form stems
    for i in range(len(sequence) - 2*min_stem_length - 3):
        for j in range(i + min_stem_length + 3, len(sequence) - min_stem_length):
            # Check if regions could form a stem
            potential_stem = True
            for k in range(min_stem_length):
                if not are_complementary(sequence[i+k], sequence[j+min_stem_length-1-k]):
                    potential_stem = False
                    break
            
            if potential_stem:
                # Potential stem-loop found
                stem_loops.append((i, j + min_stem_length))
                break
    
    return stem_loops

def are_complementary(base1, base2):
    """Check if two bases are complementary in RNA."""
    return (base1 == 'A' and base2 == 'U') or \
           (base1 == 'U' and base2 == 'A') or \
           (base1 == 'G' and base2 == 'C') or \
           (base1 == 'C' and base2 == 'G') or \
           (base1 == 'G' and base2 == 'U') or \
           (base1 == 'U' and base2 == 'G')  # G-U wobble pairs are valid in RNA

def apply_stem_loop_template(structure, start, end):
    """
    Apply a stem-loop template to a specific region of the structure.
    
    Parameters:
    -----------
    structure: RNA 3D structure
    start, end: Indices of the stem-loop region
    
    Returns:
    --------
    Modified structure with stem-loop template applied
    """
    # Create a copy to modify
    result = structure.copy()
    
    # Length of the region
    region_length = end - start + 1
    
    # Not enough residues to form a proper stem-loop
    if region_length < 7:
        return result
    
    # Calculate stem length (approximately 1/3 of the region on each side)
    stem_length = max(2, region_length // 6)
    loop_start = start + stem_length
    loop_end = end - stem_length
    
    # Loop length
    loop_length = loop_end - loop_start + 1
    
    # Apply stem template (roughly parallel strands)
    for i in range(stem_length):
        # Base positions in the two stems
        pos1 = start + i
        pos2 = end - i
        
        if pos1 < len(result) and pos2 < len(result):
            # Create roughly parallel strands
            if i > 0:
                # Base the position on the previous nucleotide in the strand
                result[pos1] = result[pos1-1] + np.array([0.0, 3.8, 0.0])
                result[pos2] = result[pos2+1] + np.array([0.0, -3.8, 0.0])
    
    # Apply loop template (roughly circular)
    if loop_length > 0:
        # Calculate center of the loop
        if loop_start < len(result) and loop_end < len(result):
            center = (result[loop_start-1] + result[loop_end+1]) / 2
            center[1] += 4.0  # Offset in y direction
            
            # Create a circular loop
            radius = 3.8  # approximately nucleotide distance
            for i in range(loop_length):
                idx = loop_start + i
                if idx < len(result):
                    angle = np.pi * i / (loop_length - 1)
                    result[idx] = center + np.array([
                        radius * np.cos(angle),
                        0.0,
                        radius * np.sin(angle)
                    ])
    
    return result

def post_process_rna_structure(structure, sequence, gc_content, use_global_movement=True):
    """
    Apply RNA-specific post-processing to refine a structure.
    
    Parameters:
    -----------
    structure: Predicted 3D coordinates
    sequence: RNA sequence
    gc_content: GC content of the sequence
    use_global_movement: Whether to apply global movement transformations
    
    Returns:
    --------
    Refined structure
    """
    # Create a new structure for modifications
    result = structure.copy()
    
    # 1. Apply mild refinement based on sequence composition
    noise_level = 0.1
    if gc_content > 0.6:
        # GC-rich regions tend to form more stable structures
        noise_level = 0.05  # Lower noise for more stable structures
    elif gc_content < 0.4:
        # AT-rich regions tend to be more flexible
        noise_level = 0.15  # Higher noise for more flexible regions
    
    # Apply noise proportional to sequence characteristics
    result = sample_structural_variation(
        result,
        noise_level=noise_level,
        preserve_distance=True,  # Always preserve distances for realistic structures
        use_global_movement=use_global_movement,
        correlation=0.85  # High correlation for smoother changes
    )
    
    # 2. Look for motifs in the sequence and apply structure templates
    # This is a simplified example - a complete implementation would include more motifs
    stem_loops = identify_stem_loops(sequence)
    if stem_loops:
        for start, end in stem_loops:
            # Apply stem-loop template to these regions
            result = apply_stem_loop_template(result, start, end)
    
    # 3. Normalize bond lengths to ideal values for RNA
    valid_mask = ~np.all(result == 0, axis=1)
    for i in range(1, len(result)):
        if valid_mask[i] and valid_mask[i-1]:
            # Get the current bond vector
            bond_vector = result[i] - result[i-1]
            bond_length = np.linalg.norm(bond_vector)
            
            if bond_length > 0:
                # Normalize to ideal RNA backbone distance with small variation
                ideal_length = 3.8 * (1 + np.random.normal(0, 0.03))
                result[i] = result[i-1] + (bond_vector / bond_length) * ideal_length
    
    return result
