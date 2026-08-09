# plot_auc_jitter.R
#
# Reads result JSON files of the form:
#   [dataset][model]_trained_encoder_[context_size].json
# e.g. BACEKAN_modified_trained_encoder_500000.json
#
# Each file contains a single-key list, whose value is a vector of the
# best AUC across 100 seeds.
#
# Produces a jitter plot:
#   - x-axis clusters: dataset
#   - within each dataset: subclusters by context size
#   - within each subcluster: one jittered column per model
#
# Usage: Rscript plot_auc_jitter.R /path/to/results_dir /path/to/output.png

suppressPackageStartupMessages({
  library(jsonlite)
  library(dplyr)
  library(ggplot2)
  library(stringr)
})

args <- commandArgs(trailingOnly = TRUE)
results_dir <- if (length(args) >= 1) args[1] else "."
output_file <- if (length(args) >= 2) args[2] else "auc_jitter_plot.png"

# --- Config: known datasets and models ---------------------------------
# Order models longest-name-first so regex matching doesn't let "KAN"
# swallow "KAN_modified" / "KAN_mini".
datasets <- c("bace", "bbbp", "hiv", "tox21")
models   <- c("KAN_modified", "KAN_mini", "KAN", "MLP")
model_display_order <- c("KAN", "KAN_modified", "KAN_mini", "MLP")
model_display_labels <- c("KAN", "KAN Modified", "KAN mini", "MLP")

dataset_pattern <- paste0("^(", paste(datasets, collapse = "|"), ")")
model_pattern   <- paste0("(", paste(models, collapse = "|"), ")")

# --- Find and parse files ------------------------------------------------
files <- list.files(results_dir, pattern = "_trained_encoder_.*\\.json$",
                    full.names = TRUE)

if (length(files) == 0) {
  stop("No files matching '*_trained_encoder_*.json' found in: ", results_dir)
}

parse_one_file <- function(path) {
  fname <- basename(path)
  
  ds_match <- str_extract(fname, dataset_pattern)
  if (is.na(ds_match)) {
    warning("Could not parse dataset from filename: ", fname)
    return(NULL)
  }
  
  # Strip dataset prefix, then look for model right after it, before
  # "_trained_encoder_"
  remainder <- str_remove(fname, dataset_pattern)
  model_match <- str_extract(remainder, model_pattern)
  if (is.na(model_match)) {
    warning("Could not parse model from filename: ", fname)
    return(NULL)
  }
  
  context_match <- str_extract(fname, "(?<=_trained_encoder_)[0-9]+[kKmM]?(?=\\.json$)")
  if (is.na(context_match)) {
    warning("Could not parse context size from filename: ", fname)
    return(NULL)
  }
  
  # Convert "200k" -> 200000, "1m" -> 1000000, plain digits pass through
  num_part <- as.numeric(str_extract(context_match, "^[0-9]+"))
  suffix <- str_extract(context_match, "[kKmM]$")
  context_size_numeric <- case_when(
    is.na(suffix) ~ num_part,
    suffix %in% c("k", "K") ~ num_part * 1e3,
    suffix %in% c("m", "M") ~ num_part * 1e6,
    TRUE ~ num_part
  )
  
  data <- fromJSON(path)
  auc_values <- as.numeric(data[[1]])
  
  data.frame(
    dataset = ds_match,
    model = model_match,
    context_size = as.integer(context_size_numeric),
    context_size_label = context_match,
    auc = auc_values,
    stringsAsFactors = FALSE
  )
}

df_list <- lapply(files, parse_one_file)
df <- bind_rows(df_list)

if (nrow(df) == 0) stop("No data parsed successfully.")

# --- Set factor orders for consistent plotting ---------------------------
df$dataset <- factor(df$dataset, levels = datasets)
df$model <- factor(df$model, levels = model_display_order, labels = model_display_labels)

# Order context sizes numerically but display using original label (e.g. "200k")
context_order <- df %>%
  distinct(context_size, context_size_label) %>%
  arrange(context_size)

df$context_size_label <- factor(df$context_size_label,
                                levels = context_order$context_size_label)

# --- Build plot ------------------------------------------------------------
# x position: nested combination of dataset (outer cluster) and
# context_size (subcluster). Model is the dodge/color group within each
# subcluster.
p <- ggplot(df, aes(x = context_size_label, y = auc, color = model)) +
  geom_hline(yintercept = 0.5, color = "black", linewidth = 0.6) +
  geom_jitter(
    position = position_jitterdodge(jitter.width = 0.15, dodge.width = 0.7),
    alpha = 0.5, size = 2
  ) +
  facet_wrap(~ dataset, nrow = 2, ncol=2, scales = "free_x") +
  scale_color_brewer(palette = "Set2",name="Model") +
  guides(color = guide_legend(override.aes = list(shape = 15, size = 4, alpha = 1))) +
  labs(
    x=NULL,
    y = "Best AUC"
  ) +
  theme_bw(base_size = 13) +
  theme(
    strip.background = element_rect(fill = "grey90"),
    strip.text = element_text(face = "bold"),
    panel.spacing = unit(1, "lines"),
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "bottom"
  )

ggsave(output_file, plot = p, width = 8, height = 8, dpi = 300)
cat("Saved plot to:", output_file, "\n")

p

