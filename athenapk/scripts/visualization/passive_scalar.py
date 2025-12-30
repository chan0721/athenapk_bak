import yt
import numpy as np
import matplotlib.pyplot as plt

def display_passive_scalars():
    """Display passive scalar fields from AthenaPK simulation output"""
    
    # Parameters
    output_dir = "simulation_cool/"
    file_prefix = "parthenon.prim."
    file_suffix = ".phdf"
    
    # Load a sample file to check available fields
    sample_file = f"{output_dir}{file_prefix}00000{file_suffix}"
    ds = yt.load(sample_file)
    
    print("=== Available fields ===")
    for field in ds.field_list:
        print(field)
    
    # Look for passive scalar fields
    scalar_fields = []
    for field in ds.field_list:
        field_name = str(field[1]) if isinstance(field, tuple) else str(field)
        # Common names for passive scalars in AthenaPK
        if any(x in field_name.lower() for x in ['scalar', 'r0', 'r1', 'r2', 'passive']):
            scalar_fields.append(field)
    
    print(f"\n=== Found {len(scalar_fields)} potential scalar fields ===")
    for field in scalar_fields:
        print(field)
    
    if not scalar_fields:
        print("No passive scalar fields found. Checking for common AthenaPK scalar names...")
        # Try common AthenaPK passive scalar field names
        potential_names = [
            ("athena", "r0"), ("gas", "r0"), ("parthenon", "r0"),
            ("athena", "scalar_0"), ("gas", "scalar_0"), 
            ("cons", "scalar_0"), ("prim", "scalar_0")
        ]
        
        for name in potential_names:
            try:
                test_data = ds.all_data()[name]
                scalar_fields.append(name)
                print(f"Found scalar field: {name}")
                break
            except:
                continue
    
    if not scalar_fields:
        print("No passive scalar fields detected. The simulation may not have passive scalars enabled.")
        return
    
    # Create plots for each scalar field
    for scalar_field in scalar_fields:
        print(f"\nProcessing scalar field: {scalar_field}")
        
        # Create slice plots
        for n in range(99, 1480, 50):  # Adjust range as needed
            try:
                filename = f"{output_dir}{file_prefix}{n:05d}{file_suffix}"
                ds = yt.load(filename)
                
                # Create slice plot
                slc = yt.SlicePlot(ds, "z", scalar_field, origin="native")
                slc.set_log(scalar_field, False)  # Passive scalars often aren't log-scaled
                
                # Set color limits if needed
                data = ds.all_data()
                scalar_data = data[scalar_field]
                vmin, vmax = scalar_data.min(), scalar_data.max()
                if vmax > vmin:
                    slc.set_zlim(scalar_field, vmin, vmax)
                
                slc.annotate_title(f"Passive Scalar {scalar_field[1]} - t={ds.current_time:.2e}")
                slc.save(f"scalar_{scalar_field[1]}_{n:05d}")
                
                print(f"Saved plot for {filename}")
                
            except FileNotFoundError:
                print(f"File {filename} not found, stopping...")
                break
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue

def check_blob_injection_scalar():
    """Check for blob injection scalar field specifically"""
    
    filename = "simulation_cool/parthenon.prim.00000.phdf"
    ds = yt.load(filename)
    
    # # Based on your input file, you have blob injection parameters
    # # The blob injection often uses passive scalars to track material
    
    # print("=== Checking for blob injection scalar ===")
    # print("From your input file, blob injection is configured:")
    # print("inject_once_at_cycle = 50001")
    # print("inject_n_blobs = 1")
    # print("inject_blob_radius_0 = 5.000e19")
    
    # Try to find the scalar used for blob tracking
    potential_blob_fields = [
        ("gas", "r0"), ("gas", "scalar_0"), ("gas", "chi"),
        ("athena", "r0"), ("parthenon", "r0"),
        ("cons", "scalar_0"), ("prim", "scalar_0")
    ]
    
    for field in potential_blob_fields:
        try:
            data = ds.all_data()[field]
            print(f"Found potential blob scalar: {field}")
            print(f"  Min value: {data.min()}")
            print(f"  Max value: {data.max()}")
            print(f"  Mean value: {data.mean()}")
            
            # Create a plot to visualize
            slc = yt.SlicePlot(ds, "z", field)
            slc.annotate_title(f"Blob Injection Scalar: {field[1]}")
            slc.save(f"blob_scalar_{field[1]}_initial")
            
        except:
            continue

def analyze_scalar_evolution():
    """Analyze how passive scalars evolve over time"""
    
    scalar_field = ("gas", "r0")  # Adjust based on what you find
    times = []
    scalar_stats = []
    
    for n in range(99, 1480, 50):
        try:
            filename = f"simulation_cool/parthenon.prim.{n:05d}.phdf"
            ds = yt.load(filename)
            data = ds.all_data()
            
            scalar_data = data[scalar_field]
            times.append(float(ds.current_time))
            scalar_stats.append({
                'min': float(scalar_data.min()),
                'max': float(scalar_data.max()),
                'mean': float(scalar_data.mean()),
                'std': float(np.std(scalar_data.d))
            })
            
        except:
            break
    
    # Plot evolution
    times = np.array(times)
    mins = [s['min'] for s in scalar_stats]
    maxs = [s['max'] for s in scalar_stats]
    means = [s['mean'] for s in scalar_stats]
    
    plt.figure(figsize=(10, 6))
    plt.plot(times, mins, label='Min', alpha=0.7)
    plt.plot(times, maxs, label='Max', alpha=0.7)
    plt.plot(times, means, label='Mean', alpha=0.7)
    plt.xlabel('Time')
    plt.ylabel(f'Scalar Field {scalar_field[1]}')
    plt.title('Passive Scalar Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('scalar_evolution.png')
    plt.show()

if __name__ == "__main__":
    # Run the analysis
    display_passive_scalars()
    check_blob_injection_scalar()
    
    # Uncomment to analyze evolution
    # analyze_scalar_evolution()