import yt
import os

os.chdir("D:/11639/test_file/PDF_plots/v1e7_b0_noph_eqi_z1_cool10") # this line ensures the output files are saved in the current directory

ds = yt.load("D:/11639/test_file/data/v1e7_b0_noph_eqi_z1_cool10/mixing_kh_hdf5_plt_cnt_0030")

ad = ds.all_data()
mixing_region = ad.cut_region([
    "obj['gas', 'temperature'] > 1e5",
    "obj['gas', 'temperature'] < 1e6",
])
center = mixing_region.quantities.center_of_mass()
# this is not an effective way to fix the mixing layer at the center, and although it actually change the location of the mixing layer, the added region is stranger and seemingly not physical.

slc = yt.SlicePlot(
    ds,
    "z",
    ("gas", "temperature"),
    # center = center,
)

slc.set_log(("gas", "temperature"), True)

slc.save("slice_plot1.png")
