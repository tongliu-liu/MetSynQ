#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  if (!requireNamespace("xcms", quietly = TRUE)) {
    stop("The R package 'xcms' is required. Install/load it on the server before running this script.", call. = FALSE)
  }
})

usage <- function() {
  cat(
    "Usage:\n",
    "  Rscript scripts/align_custom_peaks_xcms.R --input result_with_mz.csv [--out-dir out] [--raw-dir raw]\n\n",
    "Required input columns by default:\n",
    "  sampleID, mw ID, mz, rt, rtmin, rtmax, area, int\n\n",
    "Main options:\n",
    "  --input PATH          Input CSV with externally integrated peaks and an mz column.\n",
    "  --out-dir PATH        Output directory. Defaults to the input CSV directory.\n",
    "  --raw-dir PATH        Optional raw-data directory for sample-name validation only.\n",
    "  --rt-unit min|sec     Unit of rt/rtmin/rtmax in input. Default: min.\n",
    "  --ppm NUMBER          m/z ppm tolerance used for generated mzmin/mzmax and xcms grouping. Default: 10.\n",
    "  --bw NUMBER           Peak-density RT bandwidth in seconds. Default: 30.\n",
    "  --bin-size NUMBER     xcms peak-density binSize in Da. Default: 0.25.\n",
    "  --prefix TEXT         Output file prefix. Default: result_xcms.\n",
    "\n",
    sep = ""
  )
}

parse_args <- function(args) {
  out <- list()
  i <- 1L
  while (i <= length(args)) {
    key <- args[[i]]
    if (!startsWith(key, "--")) {
      stop("Unexpected argument: ", key, call. = FALSE)
    }
    key <- sub("^--", "", key)
    if (grepl("=", key, fixed = TRUE)) {
      parts <- strsplit(key, "=", fixed = TRUE)[[1]]
      out[[parts[[1]]]] <- paste(parts[-1], collapse = "=")
      i <- i + 1L
    } else {
      if (i == length(args) || startsWith(args[[i + 1L]], "--")) {
        out[[key]] <- TRUE
        i <- i + 1L
      } else {
        out[[key]] <- args[[i + 1L]]
        i <- i + 2L
      }
    }
  }
  out
}

get_arg <- function(args, name, default = NULL) {
  if (!is.null(args[[name]])) args[[name]] else default
}

as_number_arg <- function(args, name, default) {
  value <- as.numeric(get_arg(args, name, default))
  if (!is.finite(value)) {
    stop("Option --", name, " must be numeric.", call. = FALSE)
  }
  value
}

resolve_col <- function(names_vec, col, required = TRUE) {
  exact <- which(names_vec == col)
  if (length(exact) > 0L) {
    return(names_vec[[exact[[1L]]]])
  }
  fold <- which(tolower(names_vec) == tolower(col))
  if (length(fold) > 0L) {
    return(names_vec[[fold[[1L]]]])
  }
  if (required) {
    stop("Required column not found: ", col, call. = FALSE)
  }
  NULL
}

numeric_col <- function(df, col) {
  raw <- df[[col]]
  out <- suppressWarnings(as.numeric(raw))
  bad <- is.na(out) & !is.na(raw) & trimws(as.character(raw)) != ""
  if (any(bad)) {
    examples <- paste(head(unique(as.character(raw[bad])), 5L), collapse = ", ")
    stop("Column '", col, "' contains non-numeric values. Examples: ", examples, call. = FALSE)
  }
  out
}

value_or_default <- function(df, col, default) {
  if (!is.null(col) && col %in% names(df)) {
    numeric_col(df, col)
  } else {
    rep(default, nrow(df))
  }
}

read_peak_csv <- function(path) {
  df <- read.csv(
    path,
    check.names = FALSE,
    stringsAsFactors = FALSE,
    na.strings = c("", "NA", "NaN")
  )
  unnamed <- names(df) == "" | grepl("^\\.\\.\\.[0-9]+$", names(df))
  if (any(unnamed)) {
    df <- df[, !unnamed, drop = FALSE]
  }
  df
}

raw_file_map <- function(sample_ids, raw_dir) {
  result <- data.frame(
    sample_index = seq_along(sample_ids),
    sampleID = sample_ids,
    raw_file = NA_character_,
    raw_status = "raw_dir_not_provided",
    stringsAsFactors = FALSE
  )
  if (is.null(raw_dir) || is.na(raw_dir) || raw_dir == "") {
    return(result)
  }
  if (!dir.exists(raw_dir)) {
    result$raw_status <- "raw_dir_missing"
    return(result)
  }

  raw_paths <- list.files(raw_dir, full.names = TRUE, recursive = TRUE, include.dirs = TRUE)
  raw_paths <- raw_paths[grepl("\\.(mzML|mzXML|cdf|raw)$", raw_paths, ignore.case = TRUE)]
  raw_bases <- tools::file_path_sans_ext(basename(raw_paths))

  for (i in seq_along(sample_ids)) {
    hit <- which(raw_bases == sample_ids[[i]])
    if (length(hit) == 1L) {
      result$raw_file[[i]] <- normalizePath(raw_paths[[hit]], winslash = "/", mustWork = FALSE)
      result$raw_status[[i]] <- "matched"
    } else if (length(hit) > 1L) {
      result$raw_file[[i]] <- paste(normalizePath(raw_paths[hit], winslash = "/", mustWork = FALSE), collapse = ";")
      result$raw_status[[i]] <- "duplicate_raw_matches"
    } else {
      result$raw_status[[i]] <- "no_raw_match"
    }
  }
  result
}

feature_from_single_peak <- function(peaks, peak_id) {
  row <- peaks[1L, , drop = FALSE]
  data.frame(
    mzmed = row$mz,
    mzmin = row$mzmin,
    mzmax = row$mzmax,
    rtmed = row$rt,
    rtmin = row$rtmin,
    rtmax = row$rtmax,
    npeaks = 1L,
    peakidx = I(list(as.integer(peak_id))),
    stringsAsFactors = FALSE
  )
}

get_peakidx <- function(group_row) {
  if (!"peakidx" %in% names(group_row)) {
    return(integer())
  }
  as.integer(unlist(group_row$peakidx, use.names = FALSE))
}

median_or_na <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) == 0L) NA_real_ else stats::median(x)
}

range_or_na <- function(x, fun) {
  x <- x[is.finite(x)]
  if (length(x) == 0L) NA_real_ else fun(x)
}

rbind_fill <- function(frames) {
  frames <- frames[vapply(frames, function(x) !is.null(x) && nrow(x) > 0L, logical(1))]
  if (length(frames) == 0L) {
    return(data.frame())
  }
  all_names <- unique(unlist(lapply(frames, names), use.names = FALSE))
  aligned <- lapply(frames, function(frame) {
    missing <- setdiff(all_names, names(frame))
    for (name in missing) {
      frame[[name]] <- NA
    }
    frame[, all_names, drop = FALSE]
  })
  do.call(rbind, aligned)
}

run_density_grouping <- function(peaks, sample_groups, bw, min_fraction, min_samples,
                                 bin_size, ppm, peak_ids) {
  args <- list(
    peaks = peaks,
    sampleGroups = sample_groups,
    bw = bw,
    minFraction = min_fraction,
    minSamples = min_samples,
    binSize = bin_size,
    ppm = ppm,
    index = peak_ids
  )
  supported <- names(formals(xcms::do_groupChromPeaks_density))
  args <- args[names(args) %in% supported]
  groups <- do.call(xcms::do_groupChromPeaks_density, args)

  if (!"index" %in% names(args)) {
    groups_df <- as.data.frame(groups, stringsAsFactors = FALSE)
    if ("peakidx" %in% names(groups_df)) {
      groups_df$peakidx <- I(lapply(groups_df$peakidx, function(idx) {
        idx <- suppressWarnings(as.integer(unlist(idx, use.names = FALSE)))
        if (any(is.na(idx)) || any(idx < 1L | idx > length(peak_ids))) {
          stop(
            "xcms returned peakidx values that could not be mapped to input peaks ",
            "after omitting unsupported 'index' parameter.",
            call. = FALSE
          )
        }
        peak_ids[idx]
      }))
      groups <- groups_df
    }
  }

  attr(groups, "used_args") <- names(args)
  groups
}

make_feature_row <- function(feature_id, id_value, group_index, group_source, group_row,
                             assigned_df, sample_count) {
  mz_values <- assigned_df$mz
  rt_values <- assigned_df$rt_sec
  sample_ids <- unique(assigned_df$sampleID)
  data.frame(
    feature_id = feature_id,
    `mw ID` = id_value,
    group_index = group_index,
    group_source = group_source,
    mzmed = if ("mzmed" %in% names(group_row)) as.numeric(group_row$mzmed[[1L]]) else median_or_na(mz_values),
    mzmin = if ("mzmin" %in% names(group_row)) as.numeric(group_row$mzmin[[1L]]) else range_or_na(mz_values, min),
    mzmax = if ("mzmax" %in% names(group_row)) as.numeric(group_row$mzmax[[1L]]) else range_or_na(mz_values, max),
    rtmed_sec = if ("rtmed" %in% names(group_row)) as.numeric(group_row$rtmed[[1L]]) else median_or_na(rt_values),
    rtmed_min = (if ("rtmed" %in% names(group_row)) as.numeric(group_row$rtmed[[1L]]) else median_or_na(rt_values)) / 60,
    rtmin_sec = if ("rtmin" %in% names(group_row)) as.numeric(group_row$rtmin[[1L]]) else range_or_na(rt_values, min),
    rtmax_sec = if ("rtmax" %in% names(group_row)) as.numeric(group_row$rtmax[[1L]]) else range_or_na(rt_values, max),
    npeaks = nrow(assigned_df),
    n_detected = length(sample_ids),
    missing_rate = 1 - length(sample_ids) / sample_count,
    peak_ids = paste(assigned_df$peak_id, collapse = ";"),
    sample_ids = paste(sample_ids, collapse = ";"),
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
}

pick_best_peak <- function(candidates) {
  class_rank <- c(A = 3, B = 2, C = 1)
  candidate_class <- if ("class" %in% names(candidates)) as.character(candidates$class) else rep("", nrow(candidates))
  rank <- unname(class_rank[candidate_class])
  rank[is.na(rank)] <- 0
  conf <- if ("conf" %in% names(candidates)) candidates$conf_num else rep(0, nrow(candidates))
  conf[!is.finite(conf)] <- 0
  rt_theoretic <- if ("rt_theoretic_num" %in% names(candidates)) candidates$rt_theoretic_num else rep(NA_real_, nrow(candidates))
  rt_distance <- abs(candidates$rt_min - rt_theoretic)
  rt_distance[!is.finite(rt_distance)] <- Inf
  ordering <- order(-candidates$area_num, -rank, -conf, rt_distance, candidates$peak_id)
  candidates[ordering[[1L]], , drop = FALSE]
}

args <- parse_args(commandArgs(trailingOnly = TRUE))
if (isTRUE(args$help) || isTRUE(args$h)) {
  usage()
  quit(status = 0L)
}

input_path <- get_arg(args, "input")
if (is.null(input_path)) {
  usage()
  stop("--input is required.", call. = FALSE)
}
input_path <- normalizePath(input_path, winslash = "/", mustWork = TRUE)
out_dir <- get_arg(args, "out-dir", dirname(input_path))
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
out_dir <- normalizePath(out_dir, winslash = "/", mustWork = TRUE)

raw_dir <- get_arg(args, "raw-dir", NULL)
if (!is.null(raw_dir)) {
  raw_dir <- normalizePath(raw_dir, winslash = "/", mustWork = FALSE)
}

rt_unit <- tolower(get_arg(args, "rt-unit", "min"))
if (!rt_unit %in% c("min", "minute", "minutes", "sec", "second", "seconds")) {
  stop("--rt-unit must be 'min' or 'sec'.", call. = FALSE)
}
rt_factor <- if (rt_unit %in% c("min", "minute", "minutes")) 60 else 1

ppm <- as_number_arg(args, "ppm", 10)
bw <- as_number_arg(args, "bw", 30)
bin_size <- as_number_arg(args, "bin-size", 0.25)
min_samples <- as.integer(as_number_arg(args, "min-samples", 1))
min_fraction <- as_number_arg(args, "min-fraction", 0)
rt_center_fun <- get_arg(args, "rt-center-fun", "median")
prefix <- get_arg(args, "prefix", "result_xcms")

sample_col <- get_arg(args, "sample-col", "sampleID")
id_col <- get_arg(args, "id-col", "mw ID")
mz_col <- get_arg(args, "mz-col", "mz")
mzmin_col_arg <- get_arg(args, "mzmin-col", "mzmin")
mzmax_col_arg <- get_arg(args, "mzmax-col", "mzmax")
rt_col <- get_arg(args, "rt-col", "rt")
rtmin_col <- get_arg(args, "rtmin-col", "rtmin")
rtmax_col <- get_arg(args, "rtmax-col", "rtmax")
area_col <- get_arg(args, "area-col", "area")
int_col <- get_arg(args, "int-col", "int")
class_col_arg <- get_arg(args, "class-col", "class")
conf_col_arg <- get_arg(args, "conf-col", "conf")
rt_theoretic_col_arg <- get_arg(args, "rt-theoretic-col", "rt_theoretic")

peaks_raw <- read_peak_csv(input_path)
if (nrow(peaks_raw) == 0L) {
  stop("Input CSV contains no rows.", call. = FALSE)
}

sample_col <- resolve_col(names(peaks_raw), sample_col)
id_col <- resolve_col(names(peaks_raw), id_col)
mz_col <- resolve_col(names(peaks_raw), mz_col)
mzmin_col <- resolve_col(names(peaks_raw), mzmin_col_arg, required = FALSE)
mzmax_col <- resolve_col(names(peaks_raw), mzmax_col_arg, required = FALSE)
rt_col <- resolve_col(names(peaks_raw), rt_col)
rtmin_col <- resolve_col(names(peaks_raw), rtmin_col)
rtmax_col <- resolve_col(names(peaks_raw), rtmax_col)
area_col <- resolve_col(names(peaks_raw), area_col)
int_col <- resolve_col(names(peaks_raw), int_col)
class_col <- resolve_col(names(peaks_raw), class_col_arg, required = FALSE)
conf_col <- resolve_col(names(peaks_raw), conf_col_arg, required = FALSE)
rt_theoretic_col <- resolve_col(names(peaks_raw), rt_theoretic_col_arg, required = FALSE)

sample_values <- as.character(peaks_raw[[sample_col]])
id_values <- as.character(peaks_raw[[id_col]])
if (any(is.na(sample_values) | sample_values == "")) {
  stop("Column '", sample_col, "' contains missing sample IDs.", call. = FALSE)
}
if (any(is.na(id_values) | id_values == "")) {
  stop("Column '", id_col, "' contains missing mw IDs.", call. = FALSE)
}

mz <- numeric_col(peaks_raw, mz_col)
rt_min <- numeric_col(peaks_raw, rt_col)
rtmin_min <- numeric_col(peaks_raw, rtmin_col)
rtmax_min <- numeric_col(peaks_raw, rtmax_col)
area <- numeric_col(peaks_raw, area_col)
maxo <- numeric_col(peaks_raw, int_col)
conf_num <- value_or_default(peaks_raw, conf_col, NA_real_)
rt_theoretic_num <- value_or_default(peaks_raw, rt_theoretic_col, NA_real_)

required_numeric <- list(mz = mz, rt = rt_min, rtmin = rtmin_min, rtmax = rtmax_min, area = area, int = maxo)
for (col_name in names(required_numeric)) {
  if (any(!is.finite(required_numeric[[col_name]]))) {
    stop("Required numeric column '", col_name, "' contains missing or non-finite values.", call. = FALSE)
  }
}
if (any(mz <= 0)) {
  stop("Column '", mz_col, "' must contain positive m/z values.", call. = FALSE)
}
if (any(rtmin_min > rtmax_min)) {
  stop("Some peaks have rtmin > rtmax.", call. = FALSE)
}

if (!is.null(mzmin_col) && !is.null(mzmax_col)) {
  mzmin <- numeric_col(peaks_raw, mzmin_col)
  mzmax <- numeric_col(peaks_raw, mzmax_col)
} else {
  mz_delta <- mz * ppm / 1e6
  mzmin <- mz - mz_delta
  mzmax <- mz + mz_delta
}
if (any(!is.finite(mzmin)) || any(!is.finite(mzmax)) || any(mzmin > mzmax)) {
  stop("Invalid mzmin/mzmax values.", call. = FALSE)
}

sample_ids <- unique(sample_values)
sample_index <- match(sample_values, sample_ids)
sample_count <- length(sample_ids)
sample_map <- raw_file_map(sample_ids, raw_dir)

rt_sec <- rt_min * rt_factor
rtmin_sec <- rtmin_min * rt_factor
rtmax_sec <- rtmax_min * rt_factor

peaks <- data.frame(
  mz = mz,
  mzmin = mzmin,
  mzmax = mzmax,
  rt = rt_sec,
  rtmin = rtmin_sec,
  rtmax = rtmax_sec,
  into = area,
  maxo = maxo,
  sample = sample_index
)

work <- data.frame(
  peak_id = seq_len(nrow(peaks_raw)),
  sampleID = sample_values,
  sample_index = sample_index,
  `mw ID` = id_values,
  mz = mz,
  rt_min = rt_min,
  rt_sec = rt_sec,
  area_num = area,
  int_num = maxo,
  conf_num = conf_num,
  rt_theoretic_num = rt_theoretic_num,
  stringsAsFactors = FALSE,
  check.names = FALSE
)
if (!is.null(class_col)) {
  work$class <- as.character(peaks_raw[[class_col]])
}

sample_groups <- rep(1L, sample_count)
id_order <- unique(id_values)
feature_defs <- vector("list", length(id_order))
assignment_rows <- list()
feature_list_index <- 0L
assignment_index <- 0L
manual_singletons <- 0L
manual_unassigned_singletons <- 0L
density_args_used <- character()

for (id_i in seq_along(id_order)) {
  id_value <- id_order[[id_i]]
  row_idx <- which(id_values == id_value)
  sub_peaks <- peaks[row_idx, , drop = FALSE]
  sub_work <- work[row_idx, , drop = FALSE]
  sub_peak_ids <- sub_work$peak_id

  if (nrow(sub_peaks) == 1L) {
    groups <- feature_from_single_peak(sub_peaks, sub_peak_ids[[1L]])
    sources <- "single_peak"
    manual_singletons <- manual_singletons + 1L
  } else {
    groups <- run_density_grouping(
      peaks = sub_peaks,
      sample_groups = sample_groups,
      bw = bw,
      min_fraction = min_fraction,
      min_samples = min_samples,
      bin_size = bin_size,
      ppm = ppm,
      peak_ids = sub_peak_ids
    )
    used_density_args <- attr(groups, "used_args")
    density_args_used <- union(density_args_used, used_density_args)
    groups <- as.data.frame(groups, stringsAsFactors = FALSE)
    sources <- rep("xcms_density", nrow(groups))

    assigned_by_xcms <- unique(as.integer(unlist(groups$peakidx, use.names = FALSE)))
    unassigned <- setdiff(sub_peak_ids, assigned_by_xcms)
    if (length(unassigned) > 0L) {
      extras <- vector("list", length(unassigned))
      for (j in seq_along(unassigned)) {
        peak_id <- unassigned[[j]]
        extra_idx <- which(sub_peak_ids == peak_id)
        extras[[j]] <- feature_from_single_peak(sub_peaks[extra_idx, , drop = FALSE], peak_id)
      }
      groups <- rbind_fill(list(groups, do.call(rbind, extras)))
      sources <- c(sources, rep("xcms_unassigned_singleton", length(unassigned)))
      manual_unassigned_singletons <- manual_unassigned_singletons + length(unassigned)
    }
  }

  if (nrow(groups) == 0L) {
    stop("xcms returned no groups for mw ID: ", id_value, call. = FALSE)
  }

  group_peak_lists <- lapply(seq_len(nrow(groups)), function(i) get_peakidx(groups[i, , drop = FALSE]))
  all_group_peaks <- unlist(group_peak_lists, use.names = FALSE)
  duplicated_peak_ids <- unique(all_group_peaks[duplicated(all_group_peaks)])
  if (length(duplicated_peak_ids) > 0L) {
    stop("A peak was assigned to multiple xcms groups within mw ID ", id_value, ": ",
         paste(head(duplicated_peak_ids, 10L), collapse = ", "), call. = FALSE)
  }

  for (group_i in seq_len(nrow(groups))) {
    peak_ids <- group_peak_lists[[group_i]]
    assigned_idx <- match(peak_ids, work$peak_id)
    assigned_df <- work[assigned_idx, , drop = FALSE]
    feature_id <- sprintf("%s__G%03d", id_value, group_i)

    feature_list_index <- feature_list_index + 1L
    feature_defs[[feature_list_index]] <- make_feature_row(
      feature_id = feature_id,
      id_value = id_value,
      group_index = group_i,
      group_source = sources[[group_i]],
      group_row = groups[group_i, , drop = FALSE],
      assigned_df = assigned_df,
      sample_count = sample_count
    )

    assignment_index <- assignment_index + 1L
    assignment_rows[[assignment_index]] <- data.frame(
      peak_id = assigned_df$peak_id,
      feature_id = feature_id,
      `mw ID` = assigned_df$`mw ID`,
      group_index = group_i,
      group_source = sources[[group_i]],
      sampleID = assigned_df$sampleID,
      sample_index = assigned_df$sample_index,
      mz = assigned_df$mz,
      rt_min = assigned_df$rt_min,
      rt_sec = assigned_df$rt_sec,
      area = assigned_df$area_num,
      int = assigned_df$int_num,
      class = if ("class" %in% names(assigned_df)) assigned_df$class else NA_character_,
      conf = assigned_df$conf_num,
      rt_theoretic = assigned_df$rt_theoretic_num,
      used_in_matrix = FALSE,
      duplicate_within_feature_sample = FALSE,
      stringsAsFactors = FALSE,
      check.names = FALSE
    )
  }
}

feature_defs <- do.call(rbind, feature_defs[seq_len(feature_list_index)])
assignments <- do.call(rbind, assignment_rows)

if (nrow(assignments) != nrow(work)) {
  missing_peak_ids <- setdiff(work$peak_id, assignments$peak_id)
  if (length(missing_peak_ids) > 0L) {
    stop("Some input peaks were not assigned to any feature. First missing peak IDs: ",
         paste(head(missing_peak_ids, 10L), collapse = ", "), call. = FALSE)
  }
}

matrix_df <- feature_defs[, c("feature_id", "mw ID", "mzmed", "rtmed_min", "n_detected", "missing_rate"), drop = FALSE]
for (sample_id in sample_ids) {
  matrix_df[[sample_id]] <- NA_real_
}

selected_peak_ids <- integer()
duplicate_feature_sample_count <- 0L
feature_ids <- feature_defs$feature_id
for (feature_id in feature_ids) {
  feature_row <- match(feature_id, matrix_df$feature_id)
  feature_peak_ids <- assignments$peak_id[assignments$feature_id == feature_id]
  candidates <- work[match(feature_peak_ids, work$peak_id), , drop = FALSE]
  for (sample_id in unique(candidates$sampleID)) {
    sample_candidates <- candidates[candidates$sampleID == sample_id, , drop = FALSE]
    best <- pick_best_peak(sample_candidates)
    selected_peak_ids <- c(selected_peak_ids, best$peak_id)
    matrix_df[feature_row, sample_id] <- best$area_num
    if (nrow(sample_candidates) > 1L) {
      duplicate_feature_sample_count <- duplicate_feature_sample_count + 1L
    }
  }
}

assignments$used_in_matrix <- assignments$peak_id %in% selected_peak_ids
assignments$duplicate_within_feature_sample <- ave(
  assignments$peak_id,
  assignments$feature_id,
  assignments$sampleID,
  FUN = length
) > 1L

matrix_n_detected <- rowSums(!is.na(matrix_df[, sample_ids, drop = FALSE]))
matrix_df$n_detected <- matrix_n_detected
matrix_df$missing_rate <- 1 - matrix_n_detected / sample_count
feature_defs$n_detected <- matrix_n_detected[match(feature_defs$feature_id, matrix_df$feature_id)]
feature_defs$missing_rate <- matrix_df$missing_rate[match(feature_defs$feature_id, matrix_df$feature_id)]

main_out <- file.path(out_dir, paste0(prefix, "_aligned_area_matrix.csv"))
feature_out <- file.path(out_dir, paste0(prefix, "_feature_definitions.csv"))
assignment_out <- file.path(out_dir, paste0(prefix, "_peak_assignments.csv"))
sample_map_out <- file.path(out_dir, "sample_map.csv")
qc_out <- file.path(out_dir, "xcms_grouping_qc_summary.txt")

write.csv(matrix_df, main_out, row.names = FALSE, na = "NA", fileEncoding = "UTF-8")
write.csv(feature_defs, feature_out, row.names = FALSE, na = "NA", fileEncoding = "UTF-8")
write.csv(assignments, assignment_out, row.names = FALSE, na = "NA", fileEncoding = "UTF-8")
write.csv(sample_map, sample_map_out, row.names = FALSE, na = "NA", fileEncoding = "UTF-8")

missing_rates <- matrix_df$missing_rate
qc_lines <- c(
  "xcms custom peak grouping QC summary",
  paste("created_at:", format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z")),
  paste("input:", input_path),
  paste("out_dir:", out_dir),
  paste("xcms_version:", as.character(utils::packageVersion("xcms"))),
  paste("input_rows:", nrow(peaks_raw)),
  paste("samples:", sample_count),
  paste("mw_ids:", length(id_order)),
  paste("features:", nrow(matrix_df)),
  paste("assigned_peaks:", nrow(assignments)),
  paste("selected_matrix_peaks:", length(selected_peak_ids)),
  paste("manual_singletons:", manual_singletons),
  paste("manual_unassigned_singletons:", manual_unassigned_singletons),
  paste("duplicate_feature_sample_cells:", duplicate_feature_sample_count),
  paste("rt_unit_input:", rt_unit),
  paste("ppm:", ppm),
  paste("bw_sec:", bw),
  paste("bin_size_da:", bin_size),
  paste("min_samples:", min_samples),
  paste("min_fraction:", min_fraction),
  paste("rt_center_fun:", rt_center_fun),
  paste("density_args_used:", ifelse(length(density_args_used) > 0L, paste(density_args_used, collapse = ","), "none")),
  paste("mz_window_source:", if (!is.null(mzmin_col) && !is.null(mzmax_col)) "input_mzmin_mzmax" else "generated_from_ppm"),
  paste("raw_dir:", ifelse(is.null(raw_dir), "not_provided", raw_dir)),
  paste("raw_status_counts:", paste(names(table(sample_map$raw_status)), as.integer(table(sample_map$raw_status)), sep = "=", collapse = ";")),
  paste("missing_rate_min:", signif(min(missing_rates), 6)),
  paste("missing_rate_median:", signif(stats::median(missing_rates), 6)),
  paste("missing_rate_max:", signif(max(missing_rates), 6)),
  "",
  "outputs:",
  paste("  matrix:", main_out),
  paste("  feature_definitions:", feature_out),
  paste("  peak_assignments:", assignment_out),
  paste("  sample_map:", sample_map_out),
  "",
  "sessionInfo:",
  capture.output(utils::sessionInfo())
)
writeLines(qc_lines, qc_out, useBytes = TRUE)

message("Done.")
message("Matrix: ", main_out)
message("Feature definitions: ", feature_out)
message("Peak assignments: ", assignment_out)
message("Sample map: ", sample_map_out)
message("QC summary: ", qc_out)
