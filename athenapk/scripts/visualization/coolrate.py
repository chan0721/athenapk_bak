import yt
import numpy as np
import matplotlib.pyplot as plt
import time

# Global variables for cooling table
Tk = None
Lambdak = None
alphak = None

def load_cooling_table(cool_filename):
    """Load cooling table and compute alpha coefficients"""
    global Tk, Lambdak, alphak
    
    data = np.loadtxt(cool_filename)
    Tk = data[0,:]  # Temperature array
    Lambdak = data[1,:]  # Lambda cooling function
    
    # Compute alpha coefficients (power law slopes)
    alphak = np.zeros_like(Tk)
    for i in range(len(Tk)-1):
        if Tk[i+1] > Tk[i] and Lambdak[i+1] > 0 and Lambdak[i] > 0:
            alphak[i] = np.log(Lambdak[i+1]/Lambdak[i]) / np.log(Tk[i+1]/Tk[i])
        else:
            alphak[i] = 0.0
    # Use the last calculated alpha for the final point
    alphak[-1] = alphak[-2] if len(alphak) > 1 else 0.0

def get_Lambda(T):
    """Get cooling function Lambda(T) using piecewise power law interpolation"""
    if T < Tk.min():
        return 0.
    else:
        idx = np.where(T >= Tk)[0][-1]
        return Lambdak[idx] * (T / Tk[idx])**alphak[idx]

# Vectorize the function for array operations
get_Lambda_v = np.vectorize(get_Lambda)

@yt.derived_field(name=("gas","Lambda"), units="erg*cm**3/s", sampling_type="cell")
def _Lambda(field, data):
    """Cooling function Lambda(T)"""
    Lambda = get_Lambda_v(data["temperature"].d)
    return data.ds.arr(Lambda, "erg*cm**3/s")

@yt.derived_field(name=("gas","number_density"), units="1/cm**3", sampling_type="cell")
def _number_density(field, data):
    """Number density (assuming mean molecular weight)"""
    amu = data.ds.quan(1.6605402e-24, "g")
    mu = 0.6  # Mean molecular weight for ionized gas
    return data["density"] / (mu * amu)

@yt.derived_field(name=("gas","cooling_rate"), units="erg/(cm**3*s)", sampling_type="cell")
def _cooling_rate(field, data):
    """Cooling rate per unit volume: n^2 * Lambda(T)"""
    return data["number_density"]**2 * data["Lambda"]

@yt.derived_field(name=("gas","cooling_time"), units="s", sampling_type="cell")
def _cooling_time(field, data):
    """Cooling time: (3/2) * k_B * T / (n * Lambda)"""
    k_B = data.ds.quan(1.380658e-16, "erg/K")
    return 3./2. * k_B * data["temperature"] / (data["number_density"] * data["Lambda"])

@yt.derived_field(name=("gas","specific_cooling_rate"), units="erg/(g*s)", sampling_type="cell")
def _specific_cooling_rate(field, data):
    """Specific cooling rate: cooling rate per unit mass"""
    return data["cooling_rate"] / data["density"]

def analyze_cooling_rate(filename, cool_table_filename, output_dir="./"):
    """
    Analyze cooling rate from simulation output file
    
    Parameters:
    -----------
    filename : str
        Path to simulation output file (.phdf)
    cool_table_filename : str  
        Path to cooling table file
    output_dir : str
        Directory to save plots and analysis
    """
    
    print(f"Analyzing file: {filename}")
    print(f"Using cooling table: {cool_table_filename}")
    
    # Load cooling table
    load_cooling_table(cool_table_filename)
    
    # Load simulation data
    ds = yt.load(filename)
    dd = ds.all_data()
    
    # Get basic statistics
    print("\n=== Simulation Statistics ===")
    print(f"Simulation time: {ds.current_time:.2e}")
    print(f"Domain size: {ds.domain_width}")
    print(f"Grid resolution: {ds.domain_dimensions}")
    
    # Temperature statistics
    temp = dd["temperature"]
    print(f"\nTemperature range: {temp.min():.2e} - {temp.max():.2e} K")
    print(f"Mean temperature: {temp.mean():.2e} K")
    
    # Density statistics  
    rho = dd["density"]
    print(f"\nDensity range: {rho.min():.2e} - {rho.max():.2e} g/cm³")
    print(f"Mean density: {rho.mean():.2e} g/cm³")
    
    # Cooling rate statistics
    cool_rate = dd["cooling_rate"]
    print(f"\nCooling rate range: {cool_rate.min():.2e} - {cool_rate.max():.2e} erg/(cm³·s)")
    print(f"Mean cooling rate: {cool_rate.mean():.2e} erg/(cm³·s)")
    
    # Total cooling power
    total_cooling = (cool_rate * dd["cell_volume"]).sum()
    print(f"Total cooling power: {total_cooling:.2e} erg/s")
    
    # Cooling time statistics
    cool_time = dd["cooling_time"]
    print(f"\nCooling time range: {cool_time.min().to('Myr'):.2e} - {cool_time.max().to('Myr'):.2e}")
    print(f"Mean cooling time: {cool_time.mean().to('Myr'):.2e}")
    
    return ds, dd

def plot_cooling_analysis(filename, cool_table_filename, output_dir="./"):
    """Create plots of cooling analysis"""
    
    ds, dd = analyze_cooling_rate(filename, cool_table_filename, output_dir)
    
    # Create slice plots
    slice_fields = ["temperature", "cooling_rate", "cooling_time", "density"]
    
    for field in slice_fields:
        slc = yt.SlicePlot(ds, "z", field, origin="native")
        
        if field == "cooling_time":
            slc.set_unit(field, "Myr")
        elif field == "cooling_rate":
            slc.set_log(field, True)
            
        slc.save(f"{output_dir}/{ds.basename}_{field}")
    
    # Create cooling rate vs temperature scatter plot
    plt.figure(figsize=(10, 8))
    
    temp_data = dd["temperature"].d
    cool_data = dd["cooling_rate"].d
    
    # Create 2D histogram
    plt.hist2d(np.log10(temp_data), np.log10(cool_data), 
               bins=100, cmap='viridis')
    plt.colorbar(label='Number of cells')
    plt.xlabel('log₁₀(Temperature [K])')
    plt.ylabel('log₁₀(Cooling Rate [erg cm⁻³ s⁻¹])')
    plt.title(f'Cooling Rate vs Temperature\n{ds.basename}')
    
    # Overplot theoretical curve
    temp_theory = np.logspace(4, 8, 100)
    lambda_theory = get_Lambda_v(temp_theory)
    n_mean = dd["number_density"].mean().d
    cool_theory = n_mean**2 * lambda_theory
    
    plt.plot(np.log10(temp_theory), np.log10(cool_theory), 
             'r-', linewidth=2, label=f'Theory (n={n_mean:.1e} cm⁻³)')
    plt.legend()
    
    plt.savefig(f"{output_dir}/{ds.basename}_cooling_vs_temp.png", dpi=150)
    plt.close()

def compare_cooling_scenarios(file_patterns, cool_table_filename, output_dir="./"):
    """Compare cooling rates across different scenarios"""
    
    load_cooling_table(cool_table_filename)
    
    plt.figure(figsize=(12, 8))
    
    for i, pattern in enumerate(file_patterns):
        ds = yt.load(pattern)
        dd = ds.all_data()
        
        cool_rate = dd["cooling_rate"]
        temp = dd["temperature"]
        
        # Create scatter plot
        plt.scatter(temp.d, cool_rate.d, alpha=0.1, s=1, 
                   label=f'Simulation {i+1}')
    
    plt.xlabel('Temperature [K]')
    plt.ylabel('Cooling Rate [erg cm⁻³ s⁻¹]')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.title('Cooling Rate Comparison')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(f"{output_dir}/cooling_rate_comparison.png", dpi=150)
    plt.close()

def plot_cooling_time_evolution(simulation_dir, file_pattern, timesteps, cool_table_filename, output_dir="./"):
    """
    Plot cooling rate evolution over time
    
    Parameters:
    -----------
    simulation_dir : str
        Directory containing simulation files
    file_pattern : str
        Pattern for file names (e.g., "parthenon.prim.{:05d}.phdf")
    timesteps : list
        List of timestep numbers to analyze
    cool_table_filename : str
        Path to cooling table file
    output_dir : str
        Directory to save plots
    """
    
    print("=== Cooling Rate Time Evolution Analysis ===")
    
    # Load cooling table
    load_cooling_table(cool_table_filename)
    
    times = []
    total_cooling_rates = []
    mean_cooling_rates = []
    max_cooling_rates = []
    mean_temperatures = []
    
    for n in timesteps:
        filename = f"{simulation_dir}/{file_pattern.format(n)}"
        try:
            ds = yt.load(filename)
            dd = ds.all_data()
            
            # Get simulation time
            sim_time = ds.current_time.to('Myr').d
            times.append(sim_time)
            
            # Calculate cooling statistics
            cool_rate = dd["cooling_rate"]
            total_cool = (cool_rate * dd["cell_volume"]).sum().d
            mean_cool = cool_rate.mean().d
            max_cool = cool_rate.max().d
            mean_temp = dd["temperature"].mean().d
            
            total_cooling_rates.append(total_cool)
            mean_cooling_rates.append(mean_cool)
            max_cooling_rates.append(max_cool)
            mean_temperatures.append(mean_temp)
            
            print(f"t={sim_time:6.2f} Myr: Total={total_cool:.2e}, Mean={mean_cool:.2e}, Max={max_cool:.2e} erg/s")
            
        except Exception as e:
            print(f"Error with timestep {n}: {e}")
            continue
    
    # Convert to numpy arrays
    times = np.array(times)
    total_cooling_rates = np.array(total_cooling_rates)
    mean_cooling_rates = np.array(mean_cooling_rates)
    max_cooling_rates = np.array(max_cooling_rates)
    mean_temperatures = np.array(mean_temperatures)
    
    # Create the time evolution plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Total cooling power vs time
    ax1.plot(times, total_cooling_rates, 'b-o', linewidth=2, markersize=6)
    ax1.set_xlabel('Time [Myr]')
    ax1.set_ylabel('Total Cooling Power [erg/s]')
    ax1.set_yscale('log')
    ax1.set_title('Total Cooling Power Evolution')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Mean cooling rate vs time
    ax2.plot(times, mean_cooling_rates, 'r-o', linewidth=2, markersize=6)
    ax2.set_xlabel('Time [Myr]')
    ax2.set_ylabel('Mean Cooling Rate [erg cm⁻³ s⁻¹]')
    ax2.set_yscale('log')
    ax2.set_title('Mean Cooling Rate Evolution')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Max cooling rate vs time
    ax3.plot(times, max_cooling_rates, 'g-o', linewidth=2, markersize=6)
    ax3.set_xlabel('Time [Myr]')
    ax3.set_ylabel('Max Cooling Rate [erg cm⁻³ s⁻¹]')
    ax3.set_yscale('log')
    ax3.set_title('Maximum Cooling Rate Evolution')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Mean temperature vs time
    ax4.plot(times, mean_temperatures, 'm-o', linewidth=2, markersize=6)
    ax4.set_xlabel('Time [Myr]')
    ax4.set_ylabel('Mean Temperature [K]')
    ax4.set_yscale('log')
    ax4.set_title('Mean Temperature Evolution')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/cooling_time_evolution.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    # Also create a simple plot focusing just on total cooling power
    plt.figure(figsize=(10, 6))
    plt.plot(times, total_cooling_rates, 'b-o', linewidth=3, markersize=8)
    plt.xlabel('Time [Myr]', fontsize=14)
    plt.ylabel('Total Cooling Power [erg/s]', fontsize=14)
    plt.yscale('log')
    plt.title('Cooling Rate Evolution Over Time', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tick_params(labelsize=12)
    
    # Add some annotations
    if len(times) > 1:
        initial_rate = total_cooling_rates[0]
        final_rate = total_cooling_rates[-1]
        change_factor = final_rate / initial_rate
        plt.text(0.05, 0.95, f'Change factor: {change_factor:.2f}x', 
                transform=plt.gca().transAxes, fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/total_cooling_evolution.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    return times, total_cooling_rates, mean_cooling_rates, max_cooling_rates, mean_temperatures

if __name__ == "__main__":
    # Example usage
    
    # Set up file paths
    cool_table_2Z = "/root/athenapk_bak/athenapk/inputs/cooling_tables/gnat-sternberg.cooling_2Z"
    cool_table_01Z = "/root/athenapk_bak/athenapk/inputs/cooling_tables/gnat-sternberg.cooling_0.1Z"
    
    # Analyze a single file
    filename = "simulation_coolstrong/parthenon.prim.00050.phdf"
    if False:  # Set to True to run single file analysis
        try:
            plot_cooling_analysis(filename, cool_table_2Z)
            print(f"Analysis complete for {filename}")
        except Exception as e:
            print(f"Error analyzing {filename}: {e}")
    
    # Plot cooling rate evolution over time
    print("\n=== Cooling Rate Time Evolution ===")
    simulation_dirs = ["simulation_coolstrong", "simulation_cool", "simulation_nocool"]
    
    for sim_dir in simulation_dirs:
        print(f"\nAnalyzing {sim_dir}...")
        # Generate timestep list (adjust range as needed based on available files)
        timesteps = list(range(0, 280, 10))  # Every 10 timesteps from 0 to 270
        
        try:
            # Determine which cooling table to use based on simulation type
            if "cool" in sim_dir and "nocool" not in sim_dir:
                cooling_table = cool_table_2Z
            else:
                cooling_table = cool_table_01Z if "0.1Z" in sim_dir else cool_table_2Z
            
            times, total_rates, mean_rates, max_rates, temps = plot_cooling_time_evolution(
                sim_dir, "parthenon.prim.{:05d}.phdf", timesteps, cooling_table, output_dir="./"
            )
            
            print(f"Successfully analyzed {len(times)} timesteps for {sim_dir}")
            
        except Exception as e:
            print(f"Error analyzing {sim_dir}: {e}")
    
    # Compare multiple time steps (original code)
    print("\n=== Quick Time Evolution Check ===")
    for n in [0, 50, 100, 150, 200, 250]:
        filename = f"simulation_coolstrong/parthenon.prim.{n:05d}.phdf"
        try:
            ds, dd = analyze_cooling_rate(filename, cool_table_2Z)
            total_cool = (dd["cooling_rate"] * dd["cell_volume"]).sum()
            print(f"t={n:3d}: Total cooling = {total_cool:.2e} erg/s")
        except Exception as e:
            print(f"Error with file {filename}: {e}")