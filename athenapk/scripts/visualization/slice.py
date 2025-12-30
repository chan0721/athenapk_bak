import yt
import numpy as np
 #loop over all the files in the directory  
# for i in range(0,50,1):
#     if i < 10:
#         ds = yt.load("parthenon.prim.0000" + str(i) + ".phdf")
#     elif i < 100:
#        # ds = yt.load("jet.out2.000" + str(i) + ".athdf")
#        # ds = yt.load("jet.out2.00" + str(i) + ".athdf")
#        ds = yt.load("parthenon.prim.000" + str(i) + ".phdf")
#     else:
#         ds = yt.load("jet.out2.00" + str(i) + ".athdf")
#     slc = yt.SlicePlot(ds, "z", "density")
#     # slc.annotate_line_integral_convolution('magnetic_field_x', 'magnetic_field_y',alpha=0.5)
#     slc.save()
# ds=yt.load("parthenon.prim.final.phdf")
# slc = yt.SlicePlot(ds, "z", "density")
# slc.save()

# ds = yt.load("jet.out2.00145.athdf")
# ad = ds.all_data()
# v_turb_velocity=(((ad["velocity_magnitude"]-np.average(ad["velocity_magnitude"]))**2*ad["cell_volume"]).sum()/ad["cell_volume"].sum())**0.5


# for i in range(0,50,1):
#     if i < 10:
#         ds = yt.load("parthenon.prim.0000" + str(i) + ".phdf")
#     elif i < 100:
#        # ds = yt.load("jet.out2.000" + str(i) + ".athdf")
#        # ds = yt.load("jet.out2.00" + str(i) + ".athdf")
#        ds = yt.load("parthenon.prim.000" + str(i) + ".phdf")
#     else:
#         ds = yt.load("jet.out2.00" + str(i) + ".athdf")
#     ad = ds.all_data()
#     v_turb_velocity=(((ad["velocity_magnitude"]-np.average(ad["velocity_magnitude"]))**2*ad["cell_volume"]).sum()/ad["cell_volume"].sum())**0.5
#     slc = yt.SlicePlot(ds, "z", "v_turb_velocity")
#     slc.save()
# ds=yt.load("parthenon.prim.final.phdf")
# ad = ds.all_data()
# v_turb_velocity=(((ad["velocity_magnitude"]-np.average(ad["velocity_magnitude"]))**2*ad["cell_volume"]).sum()/ad["cell_volume"].sum())**0.5
# slc = yt.SlicePlot(ds, "z", "v_turb_velocity")
# slc.save()


# velocity_magnitude = ad["velocity_magnitude"]
# magnetic_magnitude = ad["magnetic_field_magnitude"]
# varianceV = np.var(velocity_magnitude)
# varianceB = np.var(magnetic_magnitude)
# print(varianceV)
# print(varianceB)

ds = yt.load("simulation_cool/parthenon.prim.00540.phdf")
slc  = yt.SlicePlot(ds, "z", "temperature")
slc.save("cool_temp")

