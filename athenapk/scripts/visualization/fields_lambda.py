import yt
import numpy as np
import time

@yt.derived_field(name=("gas","n"), units="cm**-3",sampling_type="cell")
def _n(field,data):
    amu = data.ds.quan(1.6605402e-24, "g")
    return data["density"] / amu

def get_Lambda(T):

    if T < Tk.min():
        return 0.
    else:
        #idx = np.sum(T >= Tk) - 1
        idx = np.where(T >= Tk)[0][-1]
        #print(idx,idx1)
        return Lambdak[idx] * (T / Tk[idx])**alphak[idx]

get_Lambda_v = np.vectorize(get_Lambda)

@yt.derived_field(name=("gas","Lambda"), units="erg*cm**3/s",sampling_type="cell")
def _Lambda(field,data):
    Lambda = get_Lambda_v(data["temperature"].d)
    return data.ds.arr(Lambda, "erg*cm**3/s")

@yt.derived_field(name=("gas","Lambda_table"), units="erg*cm**3/s",sampling_type="cell")
def _Lambda_table(field,data):
    T = data["temperature"].d
    log_Lambda = np.interp(T, Tk, Lambdak, left=0.)
    Lambda = 10**log_Lambda
    return data.ds.arr(Lambda, "erg*cm**3/s")

@yt.derived_field(name=("gas","number_density"), units="1/cm**3",sampling_type="cell")
def _number_density(field,data):
    amu = data.ds.quan(1.6605402e-24, "g")
    return data["density"] / amu

@yt.derived_field(name=("gas","cooling_time"), units="s",sampling_type="cell")
def _cooling_time(field,data):
    k = data.ds.quan(1.380658e-16, "erg/K")
    return 3./2. * k * data["temperature"] / (data["number_density"] * data["Lambda_table"])

@yt.derived_field(name=("gas","cs_tcool"), units="cm",sampling_type="cell")
def _cs_tcool(field,data):
    return data["sound_speed"] * data["cooling_time"]

@yt.derived_field(name=("gas","emissivity"), units="erg*g**2/(cm**3*s)",sampling_type="cell")
def _emissivity(field,data):
    return data["density"]**2 * data["Lambda"]

@yt.derived_field(name=("gas","luminosity_per_volume"), units="erg/(cm**3*s)",sampling_type="cell")
def _luminosity_per_volume(field,data):
    return data["number_density"]**2 * data["Lambda"]

# def get_luminosity_over_pressure(dirname):
#     filename = "%s/mixing_kh_hdf5_plt_cnt_0030" % dirname
#     ds = yt.load(filename)
#     dd = ds.h.all_data()
#     energy_sum = (dd["luminosity_per_volume"] * dd["cell_volume"]).sum()
#     energy_per_area = energy_sum / (ds.domain_width[0] * ds.domain_width[2])
#     pressure_init = ds.quan(2.1341223804967856e-14, "erg/cm**3")
#     print(energy_sum)
#     print(energy_per_area/pressure_init)



# cool_filename = "cool_func_new_z0.1.dat"
# data = np.loadtxt(cool_filename)
# Tk = data[0,:]
# Lambdak = data[1,:]
# alphak = data[2,:]
# get_luminosity_over_pressure('v1e7_b0_noph_eqi_z0.1')

# cool_filename = "cool_func_new_z1.dat"

cool_filename = "/root/athenapk_bak/athenapk/inputs/cooling_tables/gnat-sternberg.cooling_0.1Z"
data = np.loadtxt(cool_filename)
Tk = data[0,:]
Lambdak = data[1,:]


alphak = np.zeros_like(Tk)
for i in range(len(Tk)-1):
    if Tk[i+1] > Tk[i] and Lambdak[i+1] > 0 and Lambdak[i] > 0:
        alphak[i] = np.log(Lambdak[i+1]/Lambdak[i]) / np.log(Tk[i+1]/Tk[i])
    else:
        alphak[i] = 0.0
# Use the last calculated alpha for the final point
alphak[-1] = alphak[-2] if len(alphak) > 1 else 0.0

# alphak = data[2,:]
# get_luminosity_over_pressure('v1e7_b0_noph_eqi_z1')

# cool_filename = "cool_func_new_z1.dat"
# data = np.loadtxt(cool_filename)
# Tk = data[0,:]
# Lambdak = data[1,:] * 10
# alphak = data[2,:]
# get_luminosity_over_pressure('v1e7_b0_noph_eqi_z1_cool10')

# start = time.time()
# my_var = "luminosity_over_pressure"
pre_factor = 1.0

my_var = "cooling_time"

for n in range (99,1480,50):
    filename = "simulation_cool/parthenon.prim.%05d" % n + ".phdf"
    title = "gnat-sternberg 0.1Z"
    cool_filename = "/root/athenapk_bak/athenapk/inputs/cooling_tables/gnat-sternberg.cooling_0.1Z"
    # if "z1" in filename:
    #     cool_filename = "cool_func_new_z1.dat"
    #     title = "z1"
    #     if "cool10" in filename:
    #         pre_factor = 10.
    #         title = "10 x z1"
    # elif "z0.1" in filename:
    #     cool_filename = "cool_func_new_z0.1.dat"
    #     title = "z0.1"

    # data = np.loadtxt(cool_filename)
    # Tk = data[0,:]
    # Lambdak = data[1,:] * pre_factor
    # alphak = data[2,:]

    ds = yt.load(filename)
    dd = ds.all_data()
    # ds.periodicity = (True, True, True)
    slc = yt.SlicePlot(ds, "z", my_var, origin="native",
                            #weight_field="density",
                           )
    # slc.annotate_contour("temperature", ncont=5, clim=(1e4,1e6), label=True, take_log=True)
    if my_var == ("gas","cooling_time"):
        slc.set_unit(my_var, "Myr")
    elif my_var == ("gas","cs_tcool"):
        slc.set_unit(my_var, "pc")
    elif my_var == ("gas","luminosity_over_pressure"):
        slc.set_zlim(my_var, 1.e3, 1.e8)
    slc.annotate_title(title)
    slc.save(ds.basename)

# end = time.time()
# print(end - start)