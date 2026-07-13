#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library("MSnbase")
  library("xcms")
  library("BiocParallel")
})

parse_args <- function(args) {
  out <- list()
  i <- 1L
  while (i <= length(args)) {
    key <- args[[i]]
    if (!startsWith(key, "--")) {
      stop("Unexpected argument: ", key, call. = FALSE)
    }
    key <- sub("^--", "", key)
    if (i == length(args) || startsWith(args[[i + 1L]], "--")) {
      out[[key]] <- TRUE
      i <- i + 1L
    } else {
      out[[key]] <- args[[i + 1L]]
      i <- i + 2L
    }
  }
  out
}

arg_value <- function(args, name, default = NULL) {
  if (!is.null(args[[name]])) args[[name]] else default
}

numeric_arg <- function(args, name, default) {
  value <- as.numeric(arg_value(args, name, default))
  if (!is.finite(value)) {
    stop("Option --", name, " must be numeric.", call. = FALSE)
  }
  value
}

integer_arg <- function(args, name, default) {
  as.integer(numeric_arg(args, name, default))
}

make_peak_density_param <- function(sample_count, bw) {
  params <- list(
    sampleGroups = rep(1L, sample_count),
    minFraction = 1 / sample_count,
    bw = bw
  )
  if ("minSamples" %in% names(formals(PeakDensityParam))) {
    params$minSamples <- 1L
  }
  do.call(PeakDensityParam, params)
}

write_empty_outputs <- function(out_dir) {
  write.csv(data.frame(), file.path(out_dir, "xcms_feature_definitions.csv"), row.names = FALSE)
  write.csv(data.frame(), file.path(out_dir, "xcms_chrom_peaks.csv"), row.names = FALSE)
  write.csv(data.frame(), file.path(out_dir, "xcms_peak_assignments.csv"), row.names = FALSE)
}

args <- parse_args(commandArgs(trailingOnly = TRUE))
data_dir <- normalizePath(arg_value(args, "data-dir"), winslash = "/", mustWork = TRUE)
out_dir <- arg_value(args, "out-dir", file.path(dirname(data_dir), "temp", "xcms"))
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
out_dir <- normalizePath(out_dir, winslash = "/", mustWork = TRUE)

ppm_val <- numeric_arg(args, "ppm", 15)
min_width <- numeric_arg(args, "min-width", 5)
max_width <- numeric_arg(args, "max-width", 50)
noise_val <- numeric_arg(args, "noise", 100)
s2n_threshold <- numeric_arg(args, "s2n", 5)
prefilter_val <- integer_arg(args, "prefilter", 3)
prefilter_intensity <- numeric_arg(args, "prefilter-intensity", 100)
mz_diff <- numeric_arg(args, "mz-diff", 0.015)
group_bw <- numeric_arg(args, "group-bw", 5)
obiwarp_bin_size <- numeric_arg(args, "obiwarp-bin-size", 1)

register(SerialParam())

mzml_files <- list.files(data_dir, pattern = "\\.mzML$", full.names = TRUE, ignore.case = TRUE)
mzml_files <- sort(mzml_files)
sample_count <- length(mzml_files)
if (sample_count == 0L) {
  stop("No mzML files found in: ", data_dir, call. = FALSE)
}
sample_ids <- tools::file_path_sans_ext(basename(mzml_files))

sample_map <- data.frame(
  sample_index = seq_along(sample_ids),
  sampleID = sample_ids,
  file = normalizePath(mzml_files, winslash = "/", mustWork = FALSE),
  stringsAsFactors = FALSE
)
write.csv(sample_map, file.path(out_dir, "xcms_sample_map.csv"), row.names = FALSE)

ms_data <- readMSData(files = mzml_files, mode = "onDisk", msLevel = 1)
ms_data <- filterEmptySpectra(ms_data)

centwave_params <- CentWaveParam(
  ppm = ppm_val,
  peakwidth = c(min_width, max_width),
  noise = noise_val,
  snthresh = s2n_threshold,
  mzdiff = mz_diff,
  prefilter = c(prefilter_val, prefilter_intensity),
  mzCenterFun = "wMean",
  integrate = 1,
  fitgauss = FALSE,
  verboseColumns = TRUE
)

xdata <- findChromPeaks(ms_data, centwave_params, return.type = "XCMSnExp")
if (nrow(chromPeaks(xdata)) == 0L) {
  write_empty_outputs(out_dir)
  stop("xcms found no chromatographic peaks.", call. = FALSE)
}

peak_density_params <- make_peak_density_param(sample_count, group_bw)
xdata <- groupChromPeaks(xdata, param = peak_density_params)
xdata <- adjustRtime(xdata, param = ObiwarpParam(binSize = obiwarp_bin_size))
xdata <- groupChromPeaks(xdata, param = peak_density_params)

feature_defs_raw <- featureDefinitions(xdata)
chrom_peaks_raw <- chromPeaks(xdata)

if (nrow(feature_defs_raw) == 0L) {
  write_empty_outputs(out_dir)
  stop("xcms found chromatographic peaks but no grouped features.", call. = FALSE)
}

feature_ids <- paste0("M", seq_len(nrow(feature_defs_raw)))
peakidx_list <- as.list(feature_defs_raw$peakidx)

feature_defs <- as.data.frame(feature_defs_raw, stringsAsFactors = FALSE)
if ("peakidx" %in% names(feature_defs)) {
  feature_defs$peakidx <- NULL
}
feature_defs <- data.frame(
  feature_index = seq_len(nrow(feature_defs)),
  feature_id = feature_ids,
  feature_defs,
  check.names = FALSE
)

chrom_peaks <- as.data.frame(chrom_peaks_raw, stringsAsFactors = FALSE)
chrom_peaks <- data.frame(
  peak_index = seq_len(nrow(chrom_peaks)),
  chrom_peaks,
  check.names = FALSE
)
if (!"sample" %in% names(chrom_peaks)) {
  stop("xcms chromPeaks output does not include a sample column.", call. = FALSE)
}
chrom_peaks$sample_index <- as.integer(chrom_peaks$sample)
chrom_peaks$sampleID <- sample_ids[chrom_peaks$sample_index]

assignment_list <- vector("list", length(peakidx_list))
for (i in seq_along(peakidx_list)) {
  peak_indices <- as.integer(peakidx_list[[i]])
  if (length(peak_indices) == 0L) {
    assignment_list[[i]] <- NULL
  } else {
    assignment_list[[i]] <- data.frame(
      feature_index = i,
      feature_id = feature_ids[[i]],
      peak_index = peak_indices,
      stringsAsFactors = FALSE
    )
  }
}
assignments <- do.call(rbind, assignment_list)
if (is.null(assignments)) {
  assignments <- data.frame(feature_index = integer(), feature_id = character(), peak_index = integer())
}

write.csv(feature_defs, file.path(out_dir, "xcms_feature_definitions.csv"), row.names = FALSE)
write.csv(chrom_peaks, file.path(out_dir, "xcms_chrom_peaks.csv"), row.names = FALSE)
write.csv(assignments, file.path(out_dir, "xcms_peak_assignments.csv"), row.names = FALSE)

qc_lines <- c(
  "xcms peak extraction QC",
  paste("created_at:", format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z")),
  paste("data_dir:", data_dir),
  paste("out_dir:", out_dir),
  paste("xcms_version:", as.character(utils::packageVersion("xcms"))),
  paste("sample_count:", sample_count),
  paste("chrom_peaks:", nrow(chrom_peaks)),
  paste("features:", nrow(feature_defs)),
  paste("ppm:", ppm_val),
  paste("peakwidth_sec:", paste(c(min_width, max_width), collapse = ",")),
  paste("noise:", noise_val),
  paste("snthresh:", s2n_threshold),
  paste("mzdiff:", mz_diff),
  paste("prefilter:", paste(c(prefilter_val, prefilter_intensity), collapse = ",")),
  paste("group_bw_sec:", group_bw),
  paste("obiwarp_bin_size:", obiwarp_bin_size),
  "",
  "sessionInfo:",
  capture.output(utils::sessionInfo())
)
writeLines(qc_lines, file.path(out_dir, "xcms_qc_summary.txt"), useBytes = TRUE)
