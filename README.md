# READ ME 


Multi-target PHD Filter -  (Vo et al., 2005) 

Authors: Gwendoline De Bie - Johann Faouzi - Hicham Janati 

----------------------------------------------------------


## 1- Code PHD: 

### File base.py: 

Principal abstract object PHDabstract: serves as a root for the child objects
(PHD) and (PHD_bootstrap). PHDabstract hides all plot and data generation methods
that are commonly shared between the children classes.

### File main.py: 

   Contains the (useful) PHD objects:

- class PHD: general PHD filter with Importance Sampling density (q, q_pdf) mandatory
in the prediction step where q is the random variable generator and q_pdf its density function.

- Class PHD_bootstrap: PHD filter with IS density taken equal to the transition 
model given by (f,f_pdf) in the initiation. The separation is useful since with
the bootstrap filter the prediction formulas are much simpler.

### File parameters.py:

   All parameters are stored:
 
Birth process properties, gaussian densities (for the moment) f and f_pdf, g and g_pdf (observation model) and their parameters.


## 2- Notebooks:

  - Simulation.ipynb gives an example of a simulation with clutter parameter equal to r = 10 and r = 30.

  - The remaining notebooks compare the effect of number of particles per target and the choice of the state estimation method (Kmeans centroids or maximum weight).
