import os
import yt
import numpy as np
from yt.units import kpc
import matplotlib.pyplot as plt

# os.chdir("D:/11639/test_file/PDF_plots/v1e7_b0_noph_eqi_z0.1") # this line ensures the output files are saved in the current director

file_forename = "simulation_cooling/mixing_cool_"
file_lastname = ".phdf"

plt.figure(figsize=(8, 6))

# ds = yt.load("D:/11639/test_file/data/v1e7_b0_noph_eqi_z1_cool10/mixing_kh_hdf5_plt_cnt_0050") # this line loads the dataset, you can change the path to your dataset

for i in range(1800):
    if i < 10:
        file_middlename = "0000" + str(i)
    elif 10 <= i < 100:
        file_middlename = "000" + str(i)
    elif 100 <= i < 1000:
        file_middlename = "00" + str(i)
    else:
        file_middlename = "0" + str(i)
    ds = yt.load(file_forename + file_middlename + file_lastname)
    data = ds.all_data()
    temp_bins = np.logspace(4, 7, 128)
    # print(ds.field_list)  # <-- Add this line to see available fields

    prof = yt.create_profile(
        data,
        ("gas", "temperature"),
        [("gas", "cell_mass")],
        weight_field=None,
        override_bins={("gas", "temperature"): temp_bins},
    )
    temps = prof.x
    pdf = prof["cell_mass"]
    plt.plot(temps, pdf,
             label=f"t={ds.current_time.in_units('Myr'):.1f}")


plt.xlabel("Temperature (K)", fontsize=14)
plt.ylabel("Cell Mass (g)", fontsize=14)
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.title("mixing_nocooling Cell Mass PDF", fontsize=16)
plt.savefig("mixing_nocooling_cell_mass_pdf.png")
plt.show()

