#!/usr/bin/env python3
"""
Script to extract timing data from CAPOOnly_Logs files.
Extracts strings with pattern "{number}s/it", calculates statistics, and creates a histogram.
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob

def extract_timing_values(log_directory):
    """
    Extract all timing values (in seconds per iteration) from log files.
    
    Args:
        log_directory (str): Path to the directory containing log files
        
    Returns:
        list: List of extracted timing values as floats
    """
    timing_values = []
    pattern = r'(\d+\.\d+)s/it'
    
    # Get all .out files in the directory
    log_files = glob.glob(os.path.join(log_directory, "*.out"))
    
    print(f"Found {len(log_files)} log files to process:")
    for file in log_files:
        print(f"  - {os.path.basename(file)}")
    
    # Process each log file
    for log_file in log_files:
        print(f"\nProcessing {os.path.basename(log_file)}...")
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                matches = re.findall(pattern, content)
                file_values = [float(match) for match in matches]
                timing_values.extend(file_values)
                print(f"  Found {len(file_values)} timing values")
        except Exception as e:
            print(f"  Error reading {log_file}: {e}")
    
    return timing_values

def save_timing_values(timing_values, output_file):
    """
    Save extracted timing values to a text file.
    
    Args:
        timing_values (list): List of timing values
        output_file (str): Path to output file
    """
    with open(output_file, 'w') as f:
        f.write("Extracted timing values (seconds per iteration):\n")
        f.write("=" * 50 + "\n\n")
        for i, value in enumerate(timing_values, 1):
            f.write(f"{i:4d}: {value:.2f}s/it\n")
        f.write(f"\nTotal values extracted: {len(timing_values)}\n")

def calculate_statistics(timing_values):
    """
    Calculate mean and standard deviation of timing values.
    
    Args:
        timing_values (list): List of timing values
        
    Returns:
        tuple: (mean, std_dev, min_val, max_val)
    """
    if not timing_values:
        return 0, 0, 0, 0
    
    values_array = np.array(timing_values)
    mean_val = np.mean(values_array)
    std_val = np.std(values_array)
    min_val = np.min(values_array)
    max_val = np.max(values_array)
    
    return mean_val, std_val, min_val, max_val

def plot_histogram(timing_values, output_file):
    """
    Create and save a histogram of timing values.
    
    Args:
        timing_values (list): List of timing values
        output_file (str): Path to save the histogram plot
    """
    if not timing_values:
        print("No timing values to plot!")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Create histogram
    n, bins, patches = plt.hist(timing_values, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    
    # Add statistics to the plot
    mean_val, std_val, min_val, max_val = calculate_statistics(timing_values)
    
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}s/it')
    plt.axvline(mean_val + std_val, color='orange', linestyle=':', linewidth=2, label=f'Mean + Std: {mean_val + std_val:.2f}s/it')
    plt.axvline(mean_val - std_val, color='orange', linestyle=':', linewidth=2, label=f'Mean - Std: {mean_val - std_val:.2f}s/it')
    
    plt.xlabel('Time per iteration (seconds)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Training Iteration Times\n(Extracted from CAPOOnly_Logs)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add text box with statistics
    stats_text = f'Statistics:\n'
    stats_text += f'Count: {len(timing_values)}\n'
    stats_text += f'Mean: {mean_val:.2f}s/it\n'
    stats_text += f'Std Dev: {std_val:.2f}s/it\n'
    stats_text += f'Min: {min_val:.2f}s/it\n'
    stats_text += f'Max: {max_val:.2f}s/it'
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Histogram saved to: {output_file}")

def main():
    """Main function to orchestrate the extraction and analysis."""
    # Set up paths
    log_directory = "/users/lucelo/maxent-rl-r1/plot_data/CAPOOnly_Logs"
    values_output_file = "/users/lucelo/maxent-rl-r1/extracted_timing_values.txt"
    histogram_output_file = "/users/lucelo/maxent-rl-r1/timing_histogram.png"
    
    print("=" * 60)
    print("CAPOOnly Logs Timing Data Extraction and Analysis")
    print("=" * 60)
    
    # Check if log directory exists
    if not os.path.exists(log_directory):
        print(f"Error: Log directory '{log_directory}' does not exist!")
        return
    
    # Extract timing values
    print("\n1. Extracting timing values from log files...")
    timing_values = extract_timing_values(log_directory)
    
    if not timing_values:
        print("No timing values found in the log files!")
        return
    
    print(f"\nTotal timing values extracted: {len(timing_values)}")
    
    # Save values to file
    print("\n2. Saving extracted values to file...")
    save_timing_values(timing_values, values_output_file)
    print(f"Values saved to: {values_output_file}")
    
    # Calculate and display statistics
    print("\n3. Calculating statistics...")
    mean_val, std_val, min_val, max_val = calculate_statistics(timing_values)
    
    print(f"\nStatistics Summary:")
    print(f"  Count: {len(timing_values)}")
    print(f"  Mean: {mean_val:.2f} seconds per iteration")
    print(f"  Standard Deviation: {std_val:.2f} seconds per iteration")
    print(f"  Minimum: {min_val:.2f} seconds per iteration")
    print(f"  Maximum: {max_val:.2f} seconds per iteration")
    
    # Create histogram
    print("\n4. Creating histogram...")
    plot_histogram(timing_values, histogram_output_file)
    
    print(f"\n" + "=" * 60)
    print("Analysis complete!")
    print(f"  - Extracted values: {values_output_file}")
    print(f"  - Histogram plot: {histogram_output_file}")
    print("=" * 60)

if __name__ == "__main__":
    main()
