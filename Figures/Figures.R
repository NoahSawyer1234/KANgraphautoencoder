library(ggplot2)
library(dplyr)
library(ggrepel)
library(scales)

df <- tibble::tribble(
  ~dataset, ~model,          ~mean_auc, ~std,   ~n,
  "BACE",   "KAN",           0.5564,    0.0209, 100,
  "BACE",   "KAN modified",  0.6799,    0.0071, 100,
  "BACE",   "KAN mini",      0.8114,    0.0105, 100,
  "BACE",   "MLP",           0.7281,    0.0067, 100,
  
  "BBBP",   "KAN",           0.5269,    0.0380, 100,
  "BBBP",   "KAN modified",  0.5999,    0.0053, 100,
  "BBBP",   "KAN mini",      0.7594,    0.0099, 100,
  "BBBP",   "MLP",           0.7399,    0.0097, 100,
  
  "HIV",    "KAN",           0.5476,    0.0218, 100,
  "HIV",    "KAN modified",  0.6812,    0.0150, 100,
  "HIV",    "KAN mini",      0.7314,    0.0128, 100,
  "HIV",    "MLP",           0.7091,    0.0124, 100,
  
  "TOX21",  "KAN",           0.5945,    0.0086, 100,
  "TOX21",  "KAN modified",  0.5954,    0.0228, 100,
  "TOX21",  "KAN mini",      0.7481,    0.0036, 100,
  "TOX21",  "MLP",           0.7221,    0.0050, 100
) %>%
  mutate(se = std / sqrt(n))

df$model <- factor(df$model, levels = c("KAN", "KAN modified", "KAN mini", "MLP"))

# Literature benchmark values (KA-GCN best per dataset, from Li 2025, Table 1)
lit_ref <- tibble::tribble(
  ~dataset, ~lit_auc,
  "BACE",   0.8900,
  "BBBP",   0.7870,
  "HIV",    0.8210,
  "TOX21",  0.7990
) %>%
  mutate(
    x_pos = as.numeric(factor(dataset, levels = c("BACE", "BBBP", "HIV", "TOX21"))),
    xmin = x_pos - 0.4,
    xmax = x_pos + 0.4
  )

p <- ggplot(df, aes(x = dataset, y = mean_auc, fill = model)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  geom_errorbar(
    aes(ymin = mean_auc - 2 * se, ymax = mean_auc + 2 * se),
    position = position_dodge(width = 0.8),
    width = 0.2,
    linewidth = 0.4
  ) +
  geom_segment(
    data = lit_ref,
    aes(x = xmin, xend = xmax, y = lit_auc, yend = lit_auc, linetype = "Li 2025"),
    inherit.aes = FALSE,
    color = "black",
    linewidth = 0.7
  ) +
  labs(
    x = "Dataset",
    y = "Mean AUC",
    fill = "Model",
    linetype = NULL
  ) +
  coord_cartesian(ylim = c(0.45, 0.9)) +
  scale_fill_brewer(palette = "Set2") +
  scale_linetype_manual(values = c("Li 2025" = "dashed")) +
  theme_minimal(base_size = 13) +
  theme(
    legend.position = "top",
    panel.grid.minor = element_blank()
  )

p

ggsave("best_auc_by_model.pdf", plot = p, width = 7, height = 5, units = "in")



model_sizes <- tibble::tribble(
  ~model,          ~n_params,
  "KAN",           2133264,
  "KAN modified",  140112,
  "KAN mini",      8387
)

df <- tibble::tribble(
  ~dataset, ~model,          ~best_auc, ~std,   ~n,
  "BACE",   "KAN",           0.5564,    0.0209, 100,
  "BACE",   "KAN modified",  0.6799,    0.0071, 100,
  "BACE",   "KAN mini",      0.8114,    0.0105, 100,
  
  "BBBP",   "KAN",           0.5269,    0.0380, 100,
  "BBBP",   "KAN modified",  0.5999,    0.0053, 100,
  "BBBP",   "KAN mini",      0.7594,    0.0099, 100,
  
  "HIV",    "KAN",           0.5476,    0.0218, 100,
  "HIV",    "KAN modified",  0.6812,    0.0150, 100,
  "HIV",    "KAN mini",      0.7314,    0.0128, 100,
  
  "TOX21",  "KAN",           0.5945,    0.0086, 100,
  "TOX21",  "KAN modified",  0.5954,    0.0228, 100,
  "TOX21",  "KAN mini",      0.7481,    0.0036, 100,
) %>%
  mutate(se = std / sqrt(n)) %>%
  left_join(model_sizes, by = "model") %>%
  arrange(dataset, n_params)

df$model <- factor(df$model, levels = c("KAN mini", "KAN modified", "KAN"))
df$dataset <- factor(df$dataset, levels = c("BACE", "BBBP", "HIV", "TOX21"))

label_df <- df %>%
  filter(model %in% c("KAN", "KAN modified", "KAN mini")) %>%
  distinct(model, n_params) %>%
  mutate(label_y = c(0.82,0.72,0.62))   %>%
  mutate(mod_label = c("KAN mini\n8,387", "KAN modified\n140,112", "KAN\n2,133,264"))

b <- ggplot(df, aes(x = n_params, y = best_auc, color = dataset)) +
  geom_line(aes(group = dataset), linewidth = 0.75, alpha = 0.7) +
  geom_text_repel(
    data = label_df,
    aes(x = n_params, y = label_y, label = mod_label),
    inherit.aes = FALSE,
    size = 3.2,
    color = "grey30",
    angle = 0,
    hjust= 0.5,
    vjust=1,
    fontface = "bold"
  ) +
  geom_errorbar(
    aes(ymin = best_auc - 2 * se, ymax = best_auc + 2 * se),
    width = 0.05,
    linewidth = 0.4,
    alpha = 0.6
  ) +
  geom_point(aes(), size = 2) +
  scale_x_log10(
    breaks = trans_breaks("log10", function(x) 10^x),
    labels = trans_format("log10", math_format(10^.x))
  ) + 
  scale_y_continuous(
    limits = c(0.5, 0.85),
    breaks = seq(0.5, 0.85, by = 0.1),
    minor_breaks = seq(0.5, 0.85, by = 0.05)
  ) +
  labs(
    x = "Number of Parameters",
    y = "Best AUC",
    color = "Dataset"
  ) +
  scale_color_brewer(palette = "Set2") +
  theme_minimal(base_size = 13) +
  theme(
    legend.position = "top",
    axis.text = element_text(face = "bold"),
    panel.grid.minor = element_line(linewidth = 0.25)
  )
b

ggsave("kan_sizes.pdf", plot = b, width = 7, height = 5, units = "in")

df <- tibble::tribble(
  ~dataset, ~model,          ~context_size, ~mean_auc,
  "BACE",   "KAN",           200000,  0.5564,
  "BACE",   "KAN",           500000,  0.4876,
  "BACE",   "KAN",           1000000, 0.5401,
  "BACE",   "KAN modified",  200000,  0.6418,
  "BACE",   "KAN modified",  500000,  0.6707,
  "BACE",   "KAN modified",  1000000, 0.6799,
  "BACE",   "KAN mini",      200000,  0.7624,
  "BACE",   "KAN mini",      500000,  0.7917,
  "BACE",   "KAN mini",      1000000, 0.8114,
  "BACE",   "MLP",           200000,  0.7281,
  "BACE",   "MLP",           500000,  0.7222,
  "BACE",   "MLP",           1000000, 0.7151,
  
  "BBBP",   "KAN",           200000,  0.5242,
  "BBBP",   "KAN",           500000,  0.5160,
  "BBBP",   "KAN",           1000000, 0.5269,
  "BBBP",   "KAN modified",  200000,  0.5949,
  "BBBP",   "KAN modified",  500000,  0.5897,
  "BBBP",   "KAN modified",  1000000, 0.5999,
  "BBBP",   "KAN mini",      200000,  0.7594,
  "BBBP",   "KAN mini",      500000,  0.7301,
  "BBBP",   "KAN mini",      1000000, 0.7383,
  "BBBP",   "MLP",           200000,  0.6328,
  "BBBP",   "MLP",           500000,  0.7399,
  "BBBP",   "MLP",           1000000, 0.7304,
  
  "HIV",    "KAN",           200000,  0.5476,
  "HIV",    "KAN",           500000,  0.5241,
  "HIV",    "KAN",           1000000, 0.5316,
  "HIV",    "KAN modified",  200000,  0.6812,
  "HIV",    "KAN modified",  500000,  0.6213,
  "HIV",    "KAN modified",  1000000, 0.6509,
  "HIV",    "KAN mini",      200000,  0.7314,
  "HIV",    "KAN mini",      500000,  0.7189,
  "HIV",    "KAN mini",      1000000, 0.7300,
  "HIV",    "MLP",           200000,  0.6748,
  "HIV",    "MLP",           500000,  0.6450,
  "HIV",    "MLP",           1000000, 0.7091,
  
  "TOX21",  "KAN",           200000,  0.5884,
  "TOX21",  "KAN",           500000,  0.5926,
  "TOX21",  "KAN",           1000000, 0.5945,
  "TOX21",  "KAN modified",  200000,  0.5954,
  "TOX21",  "KAN modified",  500000,  0.5506,
  "TOX21",  "KAN modified",  1000000, 0.5357,
  "TOX21",  "KAN mini",      200000,  0.7481,
  "TOX21",  "KAN mini",      500000,  0.7450,
  "TOX21",  "KAN mini",      1000000, 0.7310,
  "TOX21",  "MLP",           200000,  0.7221,
  "TOX21",  "MLP",           500000,  0.7200,
  "TOX21",  "MLP",           1000000, 0.7173
)

df$model <- factor(df$model, levels = c("KAN", "KAN modified", "KAN mini", "MLP"))
df$dataset <- factor(df$dataset, levels = c("BACE", "BBBP", "HIV", "TOX21"))

p3 <- ggplot(df, aes(x = context_size, y = mean_auc,
                     color = model, shape = dataset,
                     group = interaction(model, dataset))) +
  geom_line(linewidth = 0.6) +
  geom_point(size = 3) +
  scale_x_continuous(
    breaks = c(200000, 500000, 1000000),
    labels = c("200K", "500K", "1M")
  ) +
  labs(
    x = "Pretraining Set Size",
    y = "Mean AUC",
    color = "Model",
    shape = "Dataset"
  ) +
  scale_color_brewer(palette = "Set2") +
  theme_minimal(base_size = 13) +
  theme(
    legend.position = "right",
    panel.grid.minor = element_blank()
  )

p3

ggsave("mean_auc_vs_context_size.pdf", plot = p3, width = 8, height = 5.5, units = "in")

df <- tibble::tribble(
  ~dataset, ~model,          ~context_size, ~mean_auc,
  "BACE",   "KAN",           200000,  0.5564,
  "BACE",   "KAN",           500000,  0.4876,
  "BACE",   "KAN",           1000000, 0.5401,
  "BACE",   "KAN modified",  200000,  0.6418,
  "BACE",   "KAN modified",  500000,  0.6707,
  "BACE",   "KAN modified",  1000000, 0.6799,
  "BACE",   "KAN mini",      200000,  0.7624,
  "BACE",   "KAN mini",      500000,  0.7917,
  "BACE",   "KAN mini",      1000000, 0.8114,
  "BACE",   "MLP",           200000,  0.7281,
  "BACE",   "MLP",           500000,  0.7222,
  "BACE",   "MLP",           1000000, 0.7151,
  
  "BBBP",   "KAN",           200000,  0.5242,
  "BBBP",   "KAN",           500000,  0.5160,
  "BBBP",   "KAN",           1000000, 0.5269,
  "BBBP",   "KAN modified",  200000,  0.5949,
  "BBBP",   "KAN modified",  500000,  0.5897,
  "BBBP",   "KAN modified",  1000000, 0.5999,
  "BBBP",   "KAN mini",      200000,  0.7594,
  "BBBP",   "KAN mini",      500000,  0.7301,
  "BBBP",   "KAN mini",      1000000, 0.7383,
  "BBBP",   "MLP",           200000,  0.6328,
  "BBBP",   "MLP",           500000,  0.7399,
  "BBBP",   "MLP",           1000000, 0.7304,
  
  "HIV",    "KAN",           200000,  0.5476,
  "HIV",    "KAN",           500000,  0.5241,
  "HIV",    "KAN",           1000000, 0.5316,
  "HIV",    "KAN modified",  200000,  0.6812,
  "HIV",    "KAN modified",  500000,  0.6213,
  "HIV",    "KAN modified",  1000000, 0.6509,
  "HIV",    "KAN mini",      200000,  0.7314,
  "HIV",    "KAN mini",      500000,  0.7189,
  "HIV",    "KAN mini",      1000000, 0.7300,
  "HIV",    "MLP",           200000,  0.6748,
  "HIV",    "MLP",           500000,  0.6450,
  "HIV",    "MLP",           1000000, 0.7091,
  
  "TOX21",  "KAN",           200000,  0.5884,
  "TOX21",  "KAN",           500000,  0.5926,
  "TOX21",  "KAN",           1000000, 0.5945,
  "TOX21",  "KAN modified",  200000,  0.5954,
  "TOX21",  "KAN modified",  500000,  0.5506,
  "TOX21",  "KAN modified",  1000000, 0.5357,
  "TOX21",  "KAN mini",      200000,  0.7481,
  "TOX21",  "KAN mini",      500000,  0.7450,
  "TOX21",  "KAN mini",      1000000, 0.7310,
  "TOX21",  "MLP",           200000,  0.7221,
  "TOX21",  "MLP",           500000,  0.7200,
  "TOX21",  "MLP",           1000000, 0.7173
)

df$model <- factor(df$model, levels = c("KAN", "KAN modified", "KAN mini", "MLP"))
df$dataset <- factor(df$dataset, levels = c("BACE", "BBBP", "HIV", "TOX21"))

p4 <- ggplot(df, aes(x = context_size, y = mean_auc, color = model, group = model)) +
  geom_line(linewidth = 0.7) +
  geom_point(size = 2.5) +
  facet_wrap(~ dataset, nrow = 2) +
  scale_x_continuous(
    breaks = c(200000, 500000, 1000000),
    labels = c("200K", "500K", "1M")
  ) +
  labs(
    x = "Pretraining Set Size",
    y = "Mean AUC",
    color = "Model"
  ) +
  scale_color_brewer(palette = "Set2") +
  theme_minimal(base_size = 13) +
  theme(
    legend.position = "top",
    panel.grid.minor = element_blank(),
    strip.text = element_text(face = "bold")
  )

p4

ggsave("mean_auc_vs_context_size_faceted.pdf", plot = p4, width = 8, height = 6, units = "in")


df <- tibble::tribble(
  ~dataset, ~model,          ~mean_auc, ~std,   ~min_auc, ~max_auc, ~n,
  "BACE",   "KAN",           0.5564,    0.0209, 0.5159,   0.6667,   100,
  "BACE",   "KAN modified",  0.6799,    0.0071, 0.6684,   0.7067,   100,
  "BACE",   "KAN mini",      0.8114,    0.0105, 0.7848,   0.8349,   100,
  "BACE",   "MLP",           0.7281,    0.0067, 0.7171,   0.7509,   100,
  
  "BBBP",   "KAN",           0.5269,    0.0380, 0.4453,   0.6463,   100,
  "BBBP",   "KAN modified",  0.5999,    0.0053, 0.5886,   0.6151,   100,
  "BBBP",   "KAN mini",      0.7594,    0.0099, 0.7305,   0.7910,   100,
  "BBBP",   "MLP",           0.7399,    0.0097, 0.7170,   0.7674,   100,
  
  "HIV",    "KAN",           0.5476,    0.0218, 0.4953,   0.5909,   100,
  "HIV",    "KAN modified",  0.6812,    0.0150, 0.6535,   0.7210,   100,
  "HIV",    "KAN mini",      0.7314,    0.0128, 0.7097,   0.7776,   100,
  "HIV",    "MLP",           0.7091,    0.0124, 0.6762,   0.7418,   100,
  
  "TOX21",  "KAN",           0.5945,    0.0086, 0.5736,   0.6116,   100,
  "TOX21",  "KAN modified",  0.5954,    0.0228, 0.5368,   0.6470,   100,
  "TOX21",  "KAN mini",      0.7481,    0.0036, 0.7402,   0.7591,   100,
  "TOX21",  "MLP",           0.7221,    0.0050, 0.7092,   0.7344,   100
) %>%
  mutate(
    se     = std / sqrt(n),
    lower  = mean_auc - 2 * se,
    upper  = mean_auc + 2 * se,
    middle = mean_auc,
    ymin   = min_auc,
    ymax   = max_auc
  )

df$model <- factor(df$model, levels = c("KAN", "KAN modified", "KAN mini", "MLP"))

p5 <- ggplot(df, aes(x = dataset, fill = model)) +
  geom_boxplot(
    aes(ymin = ymin, lower = lower, middle = middle, upper = upper, ymax = ymax),
    stat = "identity",
    position = position_dodge(width = 0.8),
    width = 0.7,
    linewidth = 0.4
  ) +
  labs(
    x = "Dataset",
    y = "AUC",
    fill = "Model"
  ) +
  coord_cartesian(ylim = c(0.4, 0.9)) +
  scale_fill_brewer(palette = "Set2") +
  theme_minimal(base_size = 13) +
  theme(
    legend.position = "top",
    panel.grid.minor = element_blank()
  )

p5

ggsave("auc_boxplot_by_model.pdf", plot = p5, width = 8, height = 5.5, units = "in")


df <- tibble::tribble(
  ~dataset, ~model,          ~mean_auc, ~std,   ~min_auc, ~max_auc, ~n,
  "BACE",   "KAN",           0.5564,    0.0209, 0.5159,   0.6667,   100,
  "BACE",   "KAN modified",  0.6799,    0.0071, 0.6684,   0.7067,   100,
  "BACE",   "KAN mini",      0.8114,    0.0105, 0.7848,   0.8349,   100,
  "BACE",   "MLP",           0.7281,    0.0067, 0.7171,   0.7509,   100,
  
  "BBBP",   "KAN",           0.5269,    0.0380, 0.4453,   0.6463,   100,
  "BBBP",   "KAN modified",  0.5999,    0.0053, 0.5886,   0.6151,   100,
  "BBBP",   "KAN mini",      0.7594,    0.0099, 0.7305,   0.7910,   100,
  "BBBP",   "MLP",           0.7399,    0.0097, 0.7170,   0.7674,   100,
  
  "HIV",    "KAN",           0.5476,    0.0218, 0.4953,   0.5909,   100,
  "HIV",    "KAN modified",  0.6812,    0.0150, 0.6535,   0.7210,   100,
  "HIV",    "KAN mini",      0.7314,    0.0128, 0.7097,   0.7776,   100,
  "HIV",    "MLP",           0.7091,    0.0124, 0.6762,   0.7418,   100,
  
  "TOX21",  "KAN",           0.5945,    0.0086, 0.5736,   0.6116,   100,
  "TOX21",  "KAN modified",  0.5954,    0.0228, 0.5368,   0.6470,   100,
  "TOX21",  "KAN mini",      0.7481,    0.0036, 0.7402,   0.7591,   100,
  "TOX21",  "MLP",           0.7221,    0.0050, 0.7092,   0.7344,   100
) %>%
  mutate(se = std / sqrt(n))

df$dataset <- factor(df$dataset, levels = c("BACE", "BBBP", "HIV", "TOX21"))
df$model   <- factor(df$model,   levels = c("KAN", "KAN modified", "KAN mini", "MLP"))

# Manually compute dodge positions
n_models <- nlevels(df$model)
bar_width <- 0.18
group_width <- 0.85

df <- df %>%
  mutate(
    dataset_num = as.numeric(dataset),
    model_num   = as.numeric(model),
    x_center = dataset_num + (model_num - (n_models + 1) / 2) * (group_width / n_models),
    xmin = x_center - bar_width / 2,
    xmax = x_center + bar_width / 2,
    se_lower  = mean_auc - 2 * se,
    se_upper  = mean_auc + 2 * se
  )

# Literature benchmark values (KA-GCN best per dataset, from Li 2025, Table 1)
lit_ref <- tibble::tribble(
  ~dataset, ~lit_auc,
  "BACE",   0.8900,
  "BBBP",   0.7870,
  "HIV",    0.8210,
  "TOX21",  0.7990
) %>%
  mutate(
    x_pos = as.numeric(factor(dataset, levels = c("BACE", "BBBP", "HIV", "TOX21"))),
    xmin = x_pos - group_width / 2,
    xmax = x_pos + group_width / 2
  )

p6 <- ggplot(df) +
  # Muted full range: min to max
  geom_rect(
    aes(xmin = xmin, xmax = xmax, ymin = min_auc, ymax = max_auc, fill = model),
    alpha = 0.3
  ) +
  # Solid inner range: mean +/- 2SE
  geom_rect(
    aes(xmin = xmin, xmax = xmax, ymin = se_lower, ymax = se_upper, fill = model),
    alpha = 1
  ) +
  # Mean line marker
  geom_segment(
    aes(x = xmin, xend = xmax, y = mean_auc, yend = mean_auc),
    color = "black",
    linewidth = 0.5
  ) +
  # Li 2025 literature reference line
  geom_segment(
    data = lit_ref,
    aes(x = xmin, xend = xmax, y = lit_auc, yend = lit_auc, linetype = "Li 2025"),
    inherit.aes = FALSE,
    color = "black",
    linewidth = 0.7
  ) +
  scale_x_continuous(
    breaks = 1:4,
    labels = levels(df$dataset)
  ) +
  labs(
    x = "Dataset",
    y = "AUC",
    fill = "Model",
    linetype = NULL
  ) +
  coord_cartesian(ylim = c(0.4, 0.9)) +
  scale_fill_brewer(palette = "Set2") +
  scale_linetype_manual(values = c("Li 2025" = "dashed")) +
  theme_minimal(base_size = 13) +
  theme(
    legend.position = "top",
    panel.grid.minor = element_blank()
  )

p6

ggsave("auc_range_plot_with_lit.pdf", plot = p6, width = 8, height = 5.5, units = "in")