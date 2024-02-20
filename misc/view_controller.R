# Install and load the rhdf5 package if you haven't already
if (!require("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

BiocManager::install("rhdf5")

library(rhdf5)

#set working directory to the location of the file
setwd("../HDF5_file_location/../")

#check you are in the location you think you are
getwd()

# Open the HDF5 file
h5file <- H5Fopen("controller.h5")

# List the datasets in the HDF5 file
h5ls(h5file)

# Read a dataset from the HDF5 file (replace "dataset_name" with the actual dataset name)
data <- h5read(h5file, "dataset")

#print dataset
print(data)

# Close the HDF5 file
H5Fclose(h5file)