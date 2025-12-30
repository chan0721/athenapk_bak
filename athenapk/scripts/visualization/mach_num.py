import yt 
import numpy as np

# Define the speed of sound as a derived field
@yt.derived_field(name=("gas", "sound_speed"), units="cm/s", sampling_type="cell")
def _sound_speed(field, data):
    # Adiabatic index from your input file
    gamma = 1.66667
    # c_s = sqrt(gamma * P / rho)
    return np.sqrt(gamma * data["pressure"] / data["density"])

# Define the Mach number using the sound speed
@yt.derived_field(name=("gas", "mach_number"), units="dimensionless", sampling_type="cell")
def _mach_number(field, data):
    # Calculate the velocity magnitude
    v = np.sqrt(data["velocity_x"]**2 + data["velocity_y"]**2 + data["velocity_z"]**2)
    # Mach number = v / c_s
    return v / data["sound_speed"]


# read data files and make slice plots
for n in range (99,1480,50):
    filename = "simulation_cool/parthenon.prim.%05d" % n + ".phdf"
    title = "gnat-sternberg 0.1Z mach number"

    ds = yt.load(filename)
    dd = ds.all_data()

    slc = yt.SlicePlot(ds, "z", "mach_number", origin="native",
                            #weight_field="density",
                           )

    slc.set_zlim("mach_number", 0, 10)
    slc.annotate_title(title)
    slc.save(ds.basename)