import numpy as np
from typing import Dict, List
import torch
import pandas as pd
import pickle

def build_full_prompt(prompt: str, prompt_prefix: str, prompt_suffix: str) -> str:
        """
        Build the full prompt with instructions
        Args:
            prompt: Original financial statement content (without instructions)
        Returns:
            Full prompt with instructions
        """
        return f"{prompt_prefix}{prompt}{prompt_suffix}"

def check_gpu_utilization():
    """Print detailed GPU utilization information"""
    if not torch.cuda.is_available():
        print("❌ CUDA is not available. Running on CPU.")
        return False
   
    # Print GPU device information
    device_count = torch.cuda.device_count()
    print(f"✅ Found {device_count} CUDA device(s):")
   
    for i in range(torch.cuda.device_count()):
        device_props = torch.cuda.get_device_properties(i)
        print(f"  Device {i}: {device_props.name}")
        print(f"    Memory: {device_props.total_memory / 1024**3:.2f} GB")
       
    # Print current GPU usage
    current_device = torch.cuda.current_device()
    print(f"\nCurrent device: {current_device} ({torch.cuda.get_device_name(current_device)})")
    print(f"  Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"  Memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
   
    # Try using nvidia-smi command for more detailed information
    try:
        import subprocess
        print("\nnvidia-smi output:")
        subprocess.run(['nvidia-smi'], check=True)
    except:
        print("Failed to run nvidia-smi command")
   
    return True

def jensen_shannon_distance(p: Dict[str, float], q: Dict[str, float]) -> float:
    """
    Calculate Jensen-Shannon distance between two probability distributions
   
    Args:
        p: First probability distribution as dictionary
        q: Second probability distribution as dictionary
       
    Returns:
        Jensen-Shannon distance (0 = identical, 1 = maximally different)
    """
    # Ensure all keys are in both distributions
    all_keys = set(p.keys()) | set(q.keys())
    p_vec = np.array([p.get(k, 0.0) for k in all_keys])
    q_vec = np.array([q.get(k, 0.0) for k in all_keys])
   
    # Normalize distributions
    p_vec = p_vec / np.sum(p_vec) if np.sum(p_vec) > 0 else p_vec
    q_vec = q_vec / np.sum(q_vec) if np.sum(q_vec) > 0 else q_vec   
    # Calculate midpoint distribution
    m_vec = 0.5 * (p_vec + q_vec)
    # Calculate KL divergences and add a small epsilon to avoid log(0)
    eps = 1e-10
    p_vec = np.maximum(p_vec, eps)
    q_vec = np.maximum(q_vec, eps)
    m_vec = np.maximum(m_vec, eps)
   
    kl_p_m = np.sum(p_vec * np.log(p_vec / m_vec))
    kl_q_m = np.sum(q_vec * np.log(q_vec / m_vec))
   
    # Jensen-Shannon divergence
    js_divergence = 0.5 * (kl_p_m + kl_q_m)
   
    # Convert to distance
    return np.sqrt(js_divergence)

def load_dataset(file_path: str) -> List[str]:

    """
    Load a mutant dataset from a pickle file.

    The file is expected to be a pickle containing a tuple/list:
        (metadata, mutants)

    - `metadata` is a dict with at least a "header" key.
      `metadata["header"]` is a list of column names describing each field
      in the mutant rows.

      For the default processing path in `data_loader.py`, the header must
      contain at least the following column names (in any order):

          [
              "Word 1", "Replacement 1",
              "Word 2", "Replacement 2",
              "Mutant_Word_1", "Mutant_Word_2",
              "Mutant_Intersectional",
              "Index", "Original", "Similarity",
          ]

      These names are used by `_resolve_mutant_indices(...)` to locate the
      original and mutant sentences dynamically (no hard-coded indices).

    - `mutants` is a list of rows. Each row is a list whose elements are
      aligned with `metadata["header"]`.

    Args:
        file_path: Path to the `.pkl` file produced by the mutation pipeline.

    Returns:
        metadata (dict): Metadata dictionary, including the "header".
        mutants (list[list[Any]]): List of mutant rows aligned with the header.
    """
    
    with open(file_path, 'rb') as f:
        content = pickle.load(f)
        print("Loaded mutant data of type:", type(content))
        # Expecting a two-element list: [metadata, mutants]
        metadata = content[0]  # e.g., a dictionary including the header info
        mutants = content[1]   # list of rows (each row is a list)
    return [metadata, mutants]

def store_mutant_results(results_data, output_file):
    """Store results to Excel file"""
    header = results_data['header']
    results = results_data['results']
   
    # Create and save DataFrame
    df = pd.DataFrame(results, columns=header)
    df.to_excel(output_file, index=False)
    print('Results stored in', output_file)
