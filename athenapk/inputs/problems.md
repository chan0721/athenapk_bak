# problems in athenapk

1. evan told me that his project used athenapk as well, but he didn't encountered the problem of setting gamma to be 1.0001 / 1.00001 and get with different different dt ? I said long running time for simulation, but I didn't it's based on the fact that chaning gamma from 1.6667 to 1.0001 will cause dt to drop some degrees of magnitude, so I'm not sure whether this is a problem or not.

2. the method I use to compute eddy turnover time is right or not? v is the mean velocity or the dispersion of velocity? L set to be the xlim (the length of the simulation box), is this correct?

3. the mach number derivation code I'm not sure is correct or not, and need to take a quick look at it.

4. the passive scalar code I'm also not quite sure, I set it to be proportional to the density to the blob, is this reasonable or not? If not, how to set the value for passive density?

4. cluster I didn't use now, can I postpone this later?

5. check the initial condition I need to set for the simulation on cluster : higher resolution (256^3), no cooling, low density contraction (previous 100, now 10 or so?), and plot passive scalar, density, temperature time evolution separately. 

6. for cooling, set cfl=0.0 seems failed to enable cooling successfully, need another way to raise dt for time evolution.