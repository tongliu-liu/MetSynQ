# Load required libraries
library("MSnbase")
library("xcms")

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)

# Assign arguments to descriptive variables
data_dir <- args[1]             # Path to your .mzML files
charge <- args[2]               # Polarity/charge (e.g., positive or negative)
ppm_val <- as.numeric(args[3])
min_width <- as.numeric(args[4])
max_width <- as.numeric(args[5])
output_filename <- args[6]
noise_val <- as.numeric(args[7])     # Default/Example: 1000
s2n_threshold <- as.numeric(args[8]) # Default/Example: 1 or 14.14
prefilter_val <- as.numeric(args[9]) # Default/Example: 2 or 3
mz_diff <- as.numeric(args[10])      # Default/Example: 0.0001
min_fraction <- as.numeric(args[11]) # Default/Example: 0.5

# Register serial processing to avoid multiprocessing conflicts or memory issues
register(SerialParam())

# Set working directory to the data path
setwd(data_dir)

# List all mzML files in the directory (using correct regex pattern)
mzml_files <- list.files(pattern = "\\.mzML$")
num_files <- length(mzml_files)

# Load MS data (on-disk for memory efficiency)
ms_data <- readMSData(mzml_files, mode = "onDisk", msLevel = 1)

# Filter out empty spectra to prevent errors during peak picking
ms_data <- filterEmptySpectra(ms_data)

# Configure CentWave parameters for peak detection
centwave_params <- CentWaveParam(
  ppm = ppm_val, 
  peakwidth = c(min_width, max_width),
  noise = noise_val,
  snthresh = s2n_threshold,
  mzdiff = mz_diff,
  prefilter = c(prefilter_val, 100),
  mzCenterFun = "wMean",
  integrate = 1,
  fitgauss = FALSE,
  verboseColumns = FALSE
)

# Step 1: Find chromatographic peaks
xdata <- findChromPeaks(ms_data, centwave_params, return.type = "XCMSnExp")

# Configure Peak Density parameters for grouping
# Setting sampleGroups to 1 for all files assumes they are from the same batch/group for alignment
peak_density_params <- PeakDensityParam(
  sampleGroups = rep(1, num_files),
  minFraction = 1 / num_files,
  bw = 5
)

# Step 2: Group chromatographic peaks across samples
xdata <- groupChromPeaks(xdata, param = peak_density_params)

# Step 3: Retention time alignment (Obiwarp method)
xdata <- adjustRtime(xdata, param = ObiwarpParam(binSize = 1))

# Step 4: Regroup peaks after retention time alignment
xdata <- groupChromPeaks(xdata, param = peak_density_params)

# Step 5: Fill in missing peaks (integrate background where peaks are absent)
xdata_filled <- fillChromPeaks(xdata)

# Extract feature definitions and intensity values
peak_info <- featureDefinitions(xdata_filled)
intensities <- featureValues(xdata_filled)

# Merge feature info and intensities into a single peak table
peak_table <- merge(peak_info, intensities, by = "row.names", all = TRUE)

# Drop the unneeded 'peakidx' column
columns_to_drop <- c("peakidx")
peak_table_final <- peak_table[, !(names(peak_table) %in% columns_to_drop)]

# Write the final peak table to the specified output file
write.table(
  peak_table_final, 
  file = output_filename, 
  sep = "\t", 
  row.names = FALSE, 
  col.names = TRUE
)