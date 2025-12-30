#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import os

def plot_cooling_from_history(hist_file, output_dir='.'):
    """
    Plot cooling rate evolution from Parthenon history file
    
    Parameters:
    hist_file: path to parthenon.out1.hst file
    output_dir: directory to save plots
    """
    
    # Read the history file
    print(f"Reading history file: {hist_file}")
    
    # Skip comment lines and read data
    data = []
    with open(hist_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            data.append([float(x) for x in line.split()])
    
    data = np.array(data)
    
    # Extract columns based on header:
    # [1]=time [2]=dt [3]=cycle [4]=nbtotal [5]=mass [6-8]=momentum [9]=KE [10]=tot-E [11]=Ms
    time = data[:, 0]      # Column 1: time
    cooling_power = data[:, 10]  # Column 11: Ms (cooling power/rate)
    total_energy = data[:, 9]    # Column 10: tot-E
    kinetic_energy = data[:, 8]  # Column 9: KE
    
    # Convert time to more readable units (e.g., Myr if needed)
    # Assuming time is in code units, you may need to adjust this
    time_myr = time / 1e15  # Convert to some meaningful time unit
    
    print(f"Found {len(time)} time steps")
    print(f"Time range: {time_myr[0]:.2f} - {time_myr[-1]:.2f} time units")
    print(f"Cooling power range: {cooling_power.min():.2e} - {cooling_power.max():.2e}")
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Cooling Power Evolution
    ax1.plot(time_myr, cooling_power, 'b-', linewidth=2, label='Cooling Power')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Cooling Power (Ms)', fontsize=12)
    ax1.set_title('Cooling Power Evolution Over Time', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add some statistics as text
    mean_cooling = np.mean(cooling_power)
    max_cooling = np.max(cooling_power)
    min_cooling = np.min(cooling_power)
    
    stats_text = f'Mean: {mean_cooling:.2e}\nMax: {max_cooling:.2e}\nMin: {min_cooling:.2e}'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 2: Energy Evolution for context
    ax2.plot(time_myr, total_energy, 'r-', linewidth=2, label='Total Energy')
    ax2.plot(time_myr, kinetic_energy, 'g-', linewidth=2, label='Kinetic Energy')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Energy', fontsize=12)
    ax2.set_title('Energy Evolution Over Time', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    # Save the plot
    output_file = os.path.join(output_dir, 'cooling_rate_evolution.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {output_file}")
    
    # Create a focused cooling plot
    plt.figure(figsize=(12, 8))
    plt.plot(time_myr, cooling_power, 'b-', linewidth=2)
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Cooling Rate (Ms)', fontsize=14)
    plt.title('Cooling Rate Evolution Over Time', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Highlight interesting features
    # Find peaks and valleys
    from scipy.signal import find_peaks
    try:
        peaks, _ = find_peaks(cooling_power, height=np.percentile(cooling_power, 80))
        valleys, _ = find_peaks(-cooling_power, height=-np.percentile(cooling_power, 20))
        
        if len(peaks) > 0:
            plt.scatter(time_myr[peaks], cooling_power[peaks], color='red', s=50, 
                       zorder=5, label=f'Peaks ({len(peaks)})')
        if len(valleys) > 0:
            plt.scatter(time_myr[valleys], cooling_power[valleys], color='orange', s=50, 
                       zorder=5, label=f'Valleys ({len(valleys)})')
        
        if len(peaks) > 0 or len(valleys) > 0:
            plt.legend()
            
    except ImportError:
        print("scipy not available for peak detection")
    
    # Add trend analysis
    # Calculate running average
    window_size = max(1, len(cooling_power) // 20)
    if window_size > 1:
        running_avg = np.convolve(cooling_power, np.ones(window_size)/window_size, mode='same')
        plt.plot(time_myr, running_avg, 'r--', linewidth=2, alpha=0.7, 
                label=f'Running Average (window={window_size})')
        plt.legend()
    
    plt.tight_layout()
    
    # Save focused plot
    focused_output = os.path.join(output_dir, 'cooling_rate_focused.png')
    plt.savefig(focused_output, dpi=300, bbox_inches='tight')
    print(f"Focused cooling plot saved as: {focused_output}")
    
    plt.show()
    
    return time_myr, cooling_power

def analyze_cooling_trends(time, cooling_power):
    """Analyze trends in the cooling evolution"""
    
    print("\n=== Cooling Rate Analysis ===")
    print(f"Total simulation time: {time[-1] - time[0]:.2f}")
    print(f"Average cooling rate: {np.mean(cooling_power):.3e}")
    print(f"Standard deviation: {np.std(cooling_power):.3e}")
    print(f"Coefficient of variation: {np.std(cooling_power)/np.mean(cooling_power)*100:.1f}%")
    
    # Detect overall trend
    coeffs = np.polyfit(time, cooling_power, 1)
    trend_slope = coeffs[0]
    
    if abs(trend_slope) < 1e-10:
        trend = "approximately constant"
    elif trend_slope > 0:
        trend = "increasing"
    else:
        trend = "decreasing"
    
    print(f"Overall trend: {trend} (slope: {trend_slope:.3e})")
    
    # Find periods of high/low cooling
    high_threshold = np.percentile(cooling_power, 75)
    low_threshold = np.percentile(cooling_power, 25)
    
    high_periods = np.where(cooling_power > high_threshold)[0]
    low_periods = np.where(cooling_power < low_threshold)[0]
    
    print(f"High cooling periods: {len(high_periods)} time steps ({len(high_periods)/len(cooling_power)*100:.1f}%)")
    print(f"Low cooling periods: {len(low_periods)} time steps ({len(low_periods)/len(cooling_power)*100:.1f}%)")

def plot_all_cooling_comparison(simulation_dirs, output_file='combined_cooling_evolution.png'):
    """
    Create a comprehensive comparison plot of all cooling simulations
    """
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Colors and styles for different simulations
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    styles = ['-', '--', '-.', ':', '-']
    
    all_data = {}
    
    # Read data from all simulations
    for i, sim_dir in enumerate(simulation_dirs):
        hist_file = os.path.join(sim_dir, 'parthenon.out1.hst')
        if os.path.exists(hist_file):
            print(f"Reading {sim_dir}...")
            
            # Read the history file
            data = []
            with open(hist_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('#') or not line:
                        continue
                    data.append([float(x) for x in line.split()])
            
            if len(data) > 0:
                data = np.array(data)
                time = data[:, 0] / 1e15  # Convert to readable time units
                cooling_power = data[:, 10]  # Ms column
                total_energy = data[:, 9]
                kinetic_energy = data[:, 8]
                
                all_data[sim_dir] = {
                    'time': time,
                    'cooling': cooling_power,
                    'total_energy': total_energy,
                    'kinetic_energy': kinetic_energy,
                    'color': colors[i % len(colors)],
                    'style': styles[i % len(styles)]
                }
    
    # Plot 1: Cooling Power Comparison
    ax1.set_title('Cooling Power Evolution - All Simulations', fontsize=14, fontweight='bold')
    for sim_name, data in all_data.items():
        label = sim_name.replace('simulation_', '').replace('_', ' ').title()
        ax1.plot(data['time'], data['cooling'], 
                color=data['color'], linestyle=data['style'], linewidth=2, 
                label=label, alpha=0.8)
    
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Cooling Power (Ms)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Cooling Power (Log Scale)
    ax2.set_title('Cooling Power Evolution - Log Scale', fontsize=14, fontweight='bold')
    for sim_name, data in all_data.items():
        label = sim_name.replace('simulation_', '').replace('_', ' ').title()
        # Only plot non-zero values for log scale
        mask = data['cooling'] > 0
        if np.any(mask):
            ax2.semilogy(data['time'][mask], data['cooling'][mask], 
                        color=data['color'], linestyle=data['style'], linewidth=2, 
                        label=label, alpha=0.8)
    
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Cooling Power (Ms) - Log Scale')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Normalized Cooling Rates (for easier comparison)
    ax3.set_title('Normalized Cooling Rates', fontsize=14, fontweight='bold')
    for sim_name, data in all_data.items():
        label = sim_name.replace('simulation_', '').replace('_', ' ').title()
        # Normalize by the maximum value for each simulation
        if data['cooling'].max() > 0:
            normalized_cooling = data['cooling'] / data['cooling'].max()
            ax3.plot(data['time'], normalized_cooling, 
                    color=data['color'], linestyle=data['style'], linewidth=2, 
                    label=label, alpha=0.8)
    
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Normalized Cooling Power')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Running Averages
    ax4.set_title('Cooling Power - Running Averages', fontsize=14, fontweight='bold')
    for sim_name, data in all_data.items():
        label = sim_name.replace('simulation_', '').replace('_', ' ').title()
        
        # Calculate running average
        window_size = max(1, len(data['cooling']) // 20)
        if window_size > 1:
            running_avg = np.convolve(data['cooling'], np.ones(window_size)/window_size, mode='same')
            ax4.plot(data['time'], running_avg, 
                    color=data['color'], linestyle=data['style'], linewidth=3, 
                    label=label, alpha=0.8)
    
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Cooling Power (Ms) - Smoothed')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Combined comparison plot saved as: {output_file}")
    
    # Create a second comprehensive figure with additional analysis
    fig2, ((ax5, ax6), (ax7, ax8)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 5: Energy Evolution Comparison
    ax5.set_title('Total Energy Evolution', fontsize=14, fontweight='bold')
    for sim_name, data in all_data.items():
        label = sim_name.replace('simulation_', '').replace('_', ' ').title()
        ax5.plot(data['time'], data['total_energy'], 
                color=data['color'], linestyle=data['style'], linewidth=2, 
                label=label, alpha=0.8)
    
    ax5.set_xlabel('Time')
    ax5.set_ylabel('Total Energy')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    # Plot 6: Kinetic Energy Evolution
    ax6.set_title('Kinetic Energy Evolution', fontsize=14, fontweight='bold')
    for sim_name, data in all_data.items():
        label = sim_name.replace('simulation_', '').replace('_', ' ').title()
        ax6.plot(data['time'], data['kinetic_energy'], 
                color=data['color'], linestyle=data['style'], linewidth=2, 
                label=label, alpha=0.8)
    
    ax6.set_xlabel('Time')
    ax6.set_ylabel('Kinetic Energy')
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    
    # Plot 7: Cooling Efficiency (Cooling/Total Energy)
    ax7.set_title('Cooling Efficiency (Cooling Power / Total Energy)', fontsize=14, fontweight='bold')
    for sim_name, data in all_data.items():
        label = sim_name.replace('simulation_', '').replace('_', ' ').title()
        mask = data['total_energy'] > 0
        if np.any(mask):
            efficiency = data['cooling'][mask] / data['total_energy'][mask]
            ax7.plot(data['time'][mask], efficiency, 
                    color=data['color'], linestyle=data['style'], linewidth=2, 
                    label=label, alpha=0.8)
    
    ax7.set_xlabel('Time')
    ax7.set_ylabel('Cooling Efficiency')
    ax7.grid(True, alpha=0.3)
    ax7.legend()
    
    # Plot 8: Statistics Summary
    ax8.axis('off')
    stats_text = "Simulation Statistics Summary\n" + "="*40 + "\n\n"
    
    for sim_name, data in all_data.items():
        label = sim_name.replace('simulation_', '').replace('_', ' ').title()
        mean_cooling = np.mean(data['cooling'])
        max_cooling = np.max(data['cooling'])
        std_cooling = np.std(data['cooling'])
        cv = (std_cooling / mean_cooling * 100) if mean_cooling > 0 else 0
        
        stats_text += f"{label}:\n"
        stats_text += f"  Mean: {mean_cooling:.2e}\n"
        stats_text += f"  Max:  {max_cooling:.2e}\n"
        stats_text += f"  CV:   {cv:.1f}%\n"
        stats_text += f"  Points: {len(data['time'])}\n\n"
    
    ax8.text(0.05, 0.95, stats_text, transform=ax8.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    analysis_file = output_file.replace('.png', '_analysis.png')
    plt.savefig(analysis_file, dpi=300, bbox_inches='tight')
    print(f"Additional analysis plot saved as: {analysis_file}")
    
    plt.show()
    
    return all_data

def create_ultimate_comparison_plot():
    """Create the ultimate comparison plot with all simulations"""
    
    print("\n" + "="*60)
    print("CREATING ULTIMATE COOLING COMPARISON PLOT")
    print("="*60)
    
    simulation_dirs = ['simulation_coolstrong', 'simulation_cool', 'simulation_nocool']
    
    # Filter to only existing directories
    existing_dirs = [d for d in simulation_dirs if os.path.exists(os.path.join(d, 'parthenon.out1.hst'))]
    
    if len(existing_dirs) == 0:
        print("No simulation directories with history files found!")
        return
    
    print(f"Found {len(existing_dirs)} simulations to compare:")
    for d in existing_dirs:
        print(f"  - {d}")
    
    # Create the comprehensive comparison
    all_data = plot_all_cooling_comparison(existing_dirs)
    
    # Print comparison summary
    print("\n" + "="*40)
    print("COMPARISON SUMMARY")
    print("="*40)
    
    for sim_name, data in all_data.items():
        label = sim_name.replace('simulation_', '').replace('_', ' ').title()
        time_span = data['time'][-1] - data['time'][0]
        mean_cooling = np.mean(data['cooling'])
        
        # Calculate trend
        if len(data['time']) > 1:
            coeffs = np.polyfit(data['time'], data['cooling'], 1)
            trend = "increasing" if coeffs[0] > 0 else "decreasing"
        else:
            trend = "unknown"
        
        print(f"{label}:")
        print(f"  Time span: {time_span:.1f} time units")
        print(f"  Data points: {len(data['time'])}")
        print(f"  Average cooling: {mean_cooling:.2e}")
        print(f"  Trend: {trend}")
        print()

if __name__ == "__main__":
    # Look for history files in different simulation directories
    simulation_dirs = ['simulation_coolstrong', 'simulation_cool', 'simulation_nocool']
    
    for sim_dir in simulation_dirs:
        hist_file = os.path.join(sim_dir, 'parthenon.out1.hst')
        if os.path.exists(hist_file):
            print(f"\n{'='*50}")
            print(f"Processing {sim_dir}")
            print(f"{'='*50}")
            
            try:
                time, cooling = plot_cooling_from_history(hist_file, sim_dir)
                analyze_cooling_trends(time, cooling)
            except Exception as e:
                print(f"Error processing {sim_dir}: {e}")
        else:
            print(f"History file not found: {hist_file}")
    
    # Create the ultimate comparison plot
    create_ultimate_comparison_plot()
    
    print("\nDone! Check the generated PNG files for your cooling rate evolution plots.")
    print("  - combined_cooling_evolution.png (main comparison)")
    print("  - combined_cooling_evolution_analysis.png (detailed analysis)")