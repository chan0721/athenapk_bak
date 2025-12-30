import yt 
import numpy as np
import matplotlib.pyplot as plt

# define number density
@yt.derived_field(name=("gas","number_density"), units="cm**-3",sampling_type="cell")
def _number_density(field,data):
    amu = data.ds.quan(1.6605402e-24, "g")
    return data["density"] / amu

#define cooling function Lambda(T) using interpolation
@yt.derived_field(name=("gas","Lambda_table"), units="erg*cm**3/s",sampling_type="cell")
def _Lambda_table(field,data):
    T = data["temperature"].d
    log_Lambda = np.interp(T, Tk, Lambdak, left=0.)
    Lambda = 10**log_Lambda
    return data.ds.arr(Lambda, "erg*cm**3/s")

# define cooling time
@yt.derived_field(name=("gas","cooling_time"), units="s",sampling_type="cell")
def _cooling_time(field,data):
    k = data.ds.quan(1.380658e-16, "erg/K")
    return 3./2. * k * data["temperature"] / (data["number_density"] * data["Lambda_table"])

# define an eddy turnover timescale
@yt.derived_field(name=("gas","eddy_turnover_time"), units="s",sampling_type="cell")
def _eddy_turnover_time(field,data):
    v = np.sqrt(data["velocity_x"]**2 + data["velocity_y"]**2 + data["velocity_z"]**2)
    dx = data.ds.domain_width[0]  # Assuming xlim corresponds to the x-dimension
    v[v == 0] = 1e-50
    return dx / v 

# define the ratio of cooling time to eddy turnover time
@yt.derived_field(name=("gas","tcool_over_teddy"), units="dimensionless",sampling_type="cell")
def _tcool_over_teddy(field,data):
    return data["cooling_time"] / data["eddy_turnover_time"] 

# read the cooling table and get Tk and Lambdak
cool_filename = "/root/athenapk_bak/athenapk/inputs/cooling_tables/gnat-sternberg.cooling_0.1Z"
data = np.loadtxt(cool_filename)
Tk = data[0,:]
Lambdak = data[1,:]


my_var = "tcool_over_teddy"

# read data files and make slice plots
for n in range (99,1480,50):
    filename = "simulation_cool/parthenon.prim.%05d" % n + ".phdf"
    title = "gnat-sternberg 0.1Z cool over eddy"
    cool_filename = "/root/athenapk_bak/athenapk/inputs/cooling_tables/gnat-sternberg.cooling_0.1Z"

    ds = yt.load(filename)
    dd = ds.all_data()

    slc = yt.SlicePlot(ds, "z", my_var, origin="native",
                            #weight_field="density",
                           )

    if my_var == ("gas","cooling_time"):
        slc.set_unit(my_var, "Myr")
    elif my_var == ("gas","eddy_turenover_time"):
        slc.set_unit(my_var, "Myr")
    elif my_var == ("gas","tcool_over_teddy"):
        slc.set_zlim(my_var, 0, 1e6)
    slc.annotate_title(title)
    slc.save(ds.basename)



