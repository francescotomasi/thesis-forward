#### Analysis Script — VulnYouth (VY) ####
# Reproduces analysis.R using output/VY/merged_BERT_survey_VY.xlsx
# (which already merges EmoBank VAD, GoEmotions, CORA survey variables, and STM 20 topics).

library(stm)
library(dplyr)
library(ggplot2)
library(tidytext)
library(tidyr)
library(tidyverse)
library(data.table)
library(readxl)
library(quanteda)
library(quanteda.textplots)
library(quanteda.textstats)
library(Matrix)
library(patchwork)
library(broom)
library(car)
library(pROC)
library(effsize)
library(stringr)

#### 0. Paths & Setup ####
INPUT_PATH <- "output/VY/merged_BERT_survey_VY.xlsx"
OUT_DIR    <- "output/VY"
FIG_DIR    <- file.path(OUT_DIR, "figures")
dir.create(FIG_DIR, showWarnings = FALSE, recursive = TRUE)

df <- read_xlsx(INPUT_PATH, guess_max = 10000)
cat("Loaded VY dataset:", nrow(df), "rows x", ncol(df), "cols\n")

# Compute word count from English text
df <- df %>%
  mutate(word_count = str_count(text, "\\S+"))

#### 1. Corpus Preprocessing, Mean Cosine Similarity & TF-IDF ####
processed <- textProcessor(
  documents = df$text,
  metadata = df,
  language = "english",
  stem = TRUE,
  removestopwords = TRUE,
  customstopwords = c("precariousness"),
  custompunctuation = c("-"),
  verbose = FALSE
)

out <- prepDocuments(
  documents = processed$documents,
  vocab = processed$vocab,
  meta = processed$meta,
  verbose = FALSE
)

row_indices <- rep(seq_along(out$documents), sapply(out$documents, ncol))
col_indices <- unlist(lapply(out$documents, function(x) x[1, ]))
counts      <- unlist(lapply(out$documents, function(x) x[2, ]))

sparse_mat <- sparseMatrix(
  i = row_indices,
  j = col_indices,
  x = counts,
  dims = c(length(out$documents), length(out$vocab))
)

filtered_dtm <- as.dfm(sparse_mat)
colnames(filtered_dtm) <- out$vocab
docvars(filtered_dtm)  <- out$meta

filtered_tfidf <- dfm_tfidf(filtered_dtm)

sim_matrix     <- textstat_simil(filtered_dtm, method = "cosine")
sim_matrix_raw <- as.matrix(sim_matrix)
diag(sim_matrix_raw) <- NA
mean_cosine <- mean(sim_matrix_raw, na.rm = TRUE)

tfidf_sums      <- colSums(filtered_tfidf)
top_tfidf_terms <- sort(tfidf_sums, decreasing = TRUE)

cat("\n=== Mean Cosine Similarity (VY) ===\n")
print(mean_cosine)
cat("\n=== Top 20 TF-IDF Terms (VY) ===\n")
print(head(top_tfidf_terms, 20))

#### 2. VAD Scatterplots ####
p1 <- ggplot(df, aes(x = V_predicted, y = A_predicted)) +
  geom_point(color = "#5F7D8F", alpha = 0.5) +
  geom_vline(xintercept = 3, color = "red", linetype = "dashed", linewidth = 1) +
  geom_hline(yintercept = 3, color = "blue", linetype = "dotdash", linewidth = 1) +
  theme_bw() +
  labs(title = "Valence vs Arousal")

p2 <- ggplot(df, aes(x = V_predicted, y = D_predicted)) +
  geom_point(color = "#5F7D8F", alpha = 0.5) +
  geom_vline(xintercept = 3, color = "red", linetype = "dashed", linewidth = 1) +
  geom_hline(yintercept = 3, color = "blue", linetype = "dotdash", linewidth = 1) +
  theme_bw() +
  labs(title = "Valence vs Dominance")

p3 <- ggplot(df, aes(x = A_predicted, y = D_predicted)) +
  geom_point(color = "#5F7D8F", alpha = 0.5) +
  geom_vline(xintercept = 3, color = "red", linetype = "dashed", linewidth = 1) +
  geom_hline(yintercept = 3, color = "blue", linetype = "dotdash", linewidth = 1) +
  theme_bw() +
  labs(title = "Arousal vs Dominance")

combined_plot <- p1 + p2 + p3 + plot_annotation(title = "Scatterplots (VulnYouth)")
ggsave(
  filename = file.path(FIG_DIR, "scatterplots_v_a_d.png"),
  plot = combined_plot,
  width = 18, height = 6, units = "in", dpi = 300
)

#### 3. VAD Marginal Distributions ####
p_v <- ggplot(df, aes(x = V_predicted)) +
  geom_density(fill = "#5F7D8F", color = "#5F7D8F", alpha = 0.6) +
  geom_vline(xintercept = 3, color = "red", linetype = "dashed") +
  theme_bw() +
  labs(title = "Valence (positivity)", x = "V_predicted")

p_a <- ggplot(df, aes(x = A_predicted)) +
  geom_density(fill = "#5F7D8F", color = "#5F7D8F", alpha = 0.6) +
  geom_vline(xintercept = 3, color = "red", linetype = "dashed") +
  theme_bw() +
  labs(title = "Arousal (intensity)", x = "A_predicted")

p_d <- ggplot(df, aes(x = D_predicted)) +
  geom_density(fill = "#5F7D8F", color = "#5F7D8F", alpha = 0.6) +
  geom_vline(xintercept = 3, color = "red", linetype = "dashed") +
  theme_bw() +
  labs(title = "Dominance (control)", x = "D_predicted")

combined_dist <- p_v + p_a + p_d + plot_annotation(title = "Marginal distributions (VulnYouth)")
ggsave(
  filename = file.path(FIG_DIR, "marginal_distributions.png"),
  plot = combined_dist,
  width = 18, height = 5, units = "in", dpi = 300
)

#### 4. Frequency of Predicted Emotions ####
emotions <- c(
  "admiration_pred", "amusement_pred", "anger_pred", "annoyance_pred", "approval_pred",
  "caring_pred", "confusion_pred", "curiosity_pred", "desire_pred", "disappointment_pred",
  "disapproval_pred", "disgust_pred", "embarrassment_pred", "excitement_pred", "fear_pred",
  "gratitude_pred", "grief_pred", "joy_pred", "love_pred", "nervousness_pred",
  "optimism_pred", "pride_pred", "realization_pred", "relief_pred", "remorse_pred",
  "sadness_pred", "surprise_pred", "neutral_pred"
)

df_long <- df %>%
  select(all_of(emotions)) %>%
  pivot_longer(cols = all_of(emotions), names_to = "emotion", values_to = "predicted") %>%
  group_by(emotion) %>%
  summarise(
    freq = sum(predicted > 0, na.rm = TRUE),
    perc = mean(predicted > 0, na.rm = TRUE) * 100,
    .groups = "drop"
  ) %>%
  filter(freq > 1) %>%
  mutate(
    emotion    = gsub("_pred", "", emotion),
    perc_label = paste0(round(perc, 1), "%")
  ) %>%
  arrange(desc(perc))

cat("\n=== Frequency of Predicted Emotions (VY) ===\n")
print(df_long, n = 30)

p_emo <- ggplot(df_long, aes(x = perc, y = reorder(emotion, perc))) +
  geom_col(fill = "#5F7D8F", color = "white", linewidth = 0.5, width = 0.7) +
  geom_text(aes(label = perc_label), hjust = -0.25, size = 7, color = "black") +
  coord_cartesian(clip = "off", xlim = c(0, max(df_long$perc) * 1.15)) +
  labs(
    title = "Frequency of predicted emotions in VulnYouth survey",
    x = "% predicted emotions",
    y = "Emotions"
  ) +
  theme_minimal(base_size = 20) +
  theme(
    plot.title = element_text(size = 26, hjust = 0.5),
    axis.text.y = element_text(size = 20),
    axis.text.x = element_text(size = 20),
    axis.title.x = element_text(size = 22),
    panel.grid.major.x = element_line(color = "gray90", linewidth = 0.5),
    panel.grid.minor = element_blank(),
    plot.margin = margin(t = 20, r = 60, b = 10, l = 40, unit = "pt"),
    plot.title.position = "plot"
  ) +
  scale_x_continuous(
    expand = c(0, 0),
    labels = scales::percent_format(scale = 1, accuracy = 0.1)
  )

ggsave(
  filename = file.path(FIG_DIR, "predicted_emotions_freq.png"),
  plot = p_emo,
  width = 12, height = 10, dpi = 300, bg = "white"
)

#### 5. Socio-Demographic Differences & Predicting Subjective Precarity ####
# Recode raw CORA survey variables to match analysis.R definitions
data <- df %>%
  mutate(
    ageyear = 2023 - as.numeric(V2_YEAR),
    age_dum = ifelse(ageyear >= 30, 1, 0),
    gender  = case_when(
      V1_GENDER == "Un hombre" ~ 0,
      V1_GENDER == "Una mujer" ~ 1,
      TRUE ~ NA_real_
    ),
    migr_dum = ifelse(V3_NAT == "Español/a", 0, 1),
    prec = case_when(
      V55_LIKERT == "Nada precario" ~ 0,
      V55_LIKERT == "Extremadamante precario" ~ 10,
      V55_LIKERT %in% as.character(1:9) ~ suppressWarnings(as.numeric(V55_LIKERT)),
      TRUE ~ NA_real_
    ),
    dum_prec = ifelse(prec >= 8, 1, 0),
    V_dum    = if_else(V_predicted > 3, 1, 0),
    A_dum    = if_else(A_predicted > 3, 1, 0),
    D_dum    = if_else(D_predicted > 3, 1, 0),
    dum_jobinsec = case_when(
      V44_JOBINSEC %in% c("Muy probable", "Probable") ~ 1,
      V44_JOBINSEC %in% c("Improbable", "Muy improbable") ~ 0,
      TRUE ~ NA_real_
    ),
    dum_endmeet = case_when(
      V57_HHECO %in% c("Con mucha dificultad", "Con dificultad") ~ 1,
      V57_HHECO %in% c("Con cierta dificultad", "Con cierta facilitad",
                       "Con facilitad", "Con mucha facilidad") ~ 0,
      TRUE ~ NA_real_
    ),
    dum_housburden = case_when(
      V61_HOUSECO == "Una carga pesada" ~ 1,
      V61_HOUSECO %in% c("Una carga razonable", "Ninguna carga") ~ 0,
      TRUE ~ NA_real_
    )
  )

model_logit  <- glm(dum_prec ~ factor(V_dum) + factor(A_dum) + factor(D_dum), data = data, family = binomial)
model_socio1 <- glm(dum_prec ~ factor(V_dum) + factor(A_dum) + factor(D_dum) + factor(gender), data = data, family = binomial)
model_prec5  <- glm(
  dum_prec ~ factor(V_dum) + factor(A_dum) + factor(D_dum) +
    factor(dum_jobinsec) + factor(dum_endmeet) + factor(dum_housburden),
  data = data, family = binomial
)

cat("\n=== Logit: Subjective Precarity ~ VAD (VY) ===\n")
print(summary(model_logit))
cat("\n=== Logit: Subjective Precarity ~ VAD + Gender (VY) ===\n")
print(summary(model_socio1))
cat("\n=== Logit: Subjective Precarity ~ VAD + Material Conditions (VY) ===\n")
print(summary(model_prec5))

socio_cat_vars <- c("gender", "age_dum", "migr_dum", "dum_prec")

V_socio_chi_results <- map_dfr(socio_cat_vars, ~ chisq.test(data[[.x]], data$V_dum) %>% tidy() %>% mutate(var = .x))
A_socio_chi_results <- map_dfr(socio_cat_vars, ~ chisq.test(data[[.x]], data$A_dum) %>% tidy() %>% mutate(var = .x))
D_socio_chi_results <- map_dfr(socio_cat_vars, ~ chisq.test(data[[.x]], data$D_dum) %>% tidy() %>% mutate(var = .x))

cat("\n=== Chi-Square: Socio-demographics x V_dum (VY) ===\n")
print(V_socio_chi_results)
cat("\n=== Chi-Square: Socio-demographics x A_dum (VY) ===\n")
print(A_socio_chi_results)
cat("\n=== Chi-Square: Socio-demographics x D_dum (VY) ===\n")
print(D_socio_chi_results)

#### 6. Sentiment Groups & Neutrality Paradox ####
pos_cols <- c(
  "admiration_pred", "amusement_pred", "approval_pred",
  "caring_pred", "curiosity_pred", "desire_pred",
  "excitement_pred", "gratitude_pred", "joy_pred",
  "love_pred", "optimism_pred", "pride_pred",
  "realization_pred", "relief_pred", "surprise_pred"
)
neg_cols <- c(
  "anger_pred", "annoyance_pred", "confusion_pred",
  "disappointment_pred", "disapproval_pred", "disgust_pred",
  "embarrassment_pred", "fear_pred", "grief_pred",
  "nervousness_pred", "remorse_pred", "sadness_pred"
)

data <- data %>%
  mutate(
    any_pos  = rowSums(across(all_of(pos_cols)) == 1, na.rm = TRUE) > 0,
    any_neg  = rowSums(across(all_of(neg_cols)) == 1, na.rm = TRUE) > 0,
    only_neu = (neutral_pred == 1 & !any_pos & !any_neg),
    sentiment_group = case_when(
      only_neu           ~ "Neutral",
      any_pos & !any_neg ~ "Positive",
      any_neg & !any_pos ~ "Negative",
      any_pos & any_neg  ~ "Ambivalent",
      TRUE               ~ "Neutral"
    ),
    sentiment_group = factor(sentiment_group, levels = c("Neutral", "Positive", "Negative", "Ambivalent")),
    sentg_dum = ifelse(sentiment_group == "Neutral", 0, 1)
  )

cat("\n=== Sentiment Group Counts (VY) ===\n")
print(data %>% count(sentiment_group))

p_sent <- ggplot(data, aes(x = sentiment_group, fill = sentiment_group)) +
  geom_bar() +
  scale_fill_manual(
    values = c("Neutral" = "#B0B0B0", "Positive" = "#1B9E77", "Negative" = "#D95F02", "Ambivalent" = "#7570B3"),
    name = NULL
  ) +
  labs(
    title = "Predicted Sentiment Groups (VulnYouth)",
    x = "Predicted emotions (sentiment groups recode)",
    y = "Count"
  ) +
  theme_minimal()

ggsave(
  filename = file.path(FIG_DIR, "predicted_sentiment_groups.png"),
  plot = p_sent,
  width = 6, height = 4, units = "in", dpi = 300, bg = "white"
)

mod_glm_sentprec <- glm(dum_prec ~ factor(sentiment_group), data = data, family = binomial)
mod_glm_sentdum  <- glm(dum_prec ~ factor(sentg_dum), data = data, family = binomial)
mod_socio_sent   <- glm(sentg_dum ~ factor(gender) + factor(age_dum) + factor(migr_dum), data = data, family = binomial)
mod_prec_sent    <- glm(
  sentg_dum ~ factor(dum_jobinsec) + factor(dum_endmeet) + factor(dum_housburden),
  data = data, family = binomial
)

cat("\n=== Neutrality Paradox Models (VY) ===\n")
print(summary(mod_glm_sentprec))
print(summary(mod_glm_sentdum))
print(summary(mod_socio_sent))
print(summary(mod_prec_sent))
cat("\nVIF (mod_prec_sent):\n")
print(vif(mod_prec_sent))

#### 7. STM Topics vs Predicted Emotions ####
analyze_emotion_topics <- function(df_in, emotion_col, emotion_label, fig_dir) {
  cat(sprintf("\n================ %s vs STM Topics (VY) ================\n", toupper(emotion_label)))
  n_pos <- sum(df_in[[emotion_col]] == 1, na.rm = TRUE)
  cat(sprintf("Positive cases (%s == 1): %d / %d\n", emotion_col, n_pos, nrow(df_in)))
  if (n_pos < 5) {
    cat("Too few positive cases (<5); skipping statistical tests.\n")
    return(invisible(NULL))
  }

  topic_cols <- grep("^Topic_[0-9]+$", names(df_in), value = TRUE)
  comp <- df_in %>%
    group_by(.data[[emotion_col]]) %>%
    summarise(across(all_of(topic_cols), ~ mean(.x, na.rm = TRUE)), .groups = "drop") %>%
    pivot_longer(cols = all_of(topic_cols), names_to = "Topic", values_to = "Mean_Prevalence") %>%
    pivot_wider(names_from = all_of(emotion_col), names_prefix = "Pred_", values_from = Mean_Prevalence) %>%
    mutate(
      Difference = Pred_1 - Pred_0,
      Likelihood_Ratio = Pred_1 / Pred_0
    ) %>%
    arrange(desc(Difference))

  print(head(comp, 10))

  top1 <- comp$Topic[1]
  top3 <- comp$Topic[1:3]
  cat(sprintf("\nTop 1 topic: %s | Top 3 topics: %s\n", top1, paste(top3, collapse = ", ")))

  f_lm     <- as.formula(paste(emotion_col, "~", top1))
  f_wilcox <- as.formula(paste(top1, "~", emotion_col))
  f_logit3 <- as.formula(paste(emotion_col, "~", paste(top3, collapse = " + ")))

  cat("\n-- Linear Regression (Top 1 Topic) --\n")
  print(summary(lm(f_lm, data = df_in)))

  cat("\n-- Wilcoxon Rank-Sum Test --\n")
  print(wilcox.test(f_wilcox, data = df_in))

  cohen_top1 <- cohen.d(df_in[[top1]] ~ factor(df_in[[emotion_col]]))
  cor_top1   <- cor.test(df_in[[top1]], as.numeric(df_in[[emotion_col]]))
  roc_top1   <- roc(df_in[[emotion_col]], df_in[[top1]], quiet = TRUE)

  cat("\n-- Point-Biserial Correlation --\n")
  print(cor_top1)

  cat("\n-- Logistic Regression (Top 3 Topics) --\n")
  print(summary(glm(f_logit3, data = df_in, family = binomial)))

  comb_vec   <- rowSums(df_in[, top3, drop = FALSE], na.rm = TRUE)
  cohen_comb <- cohen.d(comb_vec ~ factor(df_in[[emotion_col]]))
  roc_comb   <- roc(df_in[[emotion_col]], comb_vec, quiet = TRUE)

  res_tbl <- data.frame(
    Metric = c("Cohen's d", "AUC"),
    Top1_Topic = c(as.numeric(cohen_top1$estimate), as.numeric(auc(roc_top1))),
    Top3_Combined = c(as.numeric(cohen_comb$estimate), as.numeric(auc(roc_comb)))
  )
  cat("\n-- Effect Size & Predictive Power Comparison --\n")
  print(res_tbl)

  png(file.path(fig_dir, paste0("roc_", emotion_label, ".png")), width = 800, height = 650, res = 120)
  plot(roc_top1, col = "blue", main = sprintf("ROC Comparison (%s, VY): %s vs Top 3", emotion_label, top1))
  plot(roc_comb, col = "red", add = TRUE)
  legend(
    "bottomright",
    legend = c(sprintf("%s (AUC=%.3f)", top1, auc(roc_top1)),
               sprintf("Top 3: %s (AUC=%.3f)", paste(top3, collapse = "+"), auc(roc_comb))),
    col = c("blue", "red"), lwd = 2
  )
  dev.off()

  invisible(comp)
}

target_emotions <- c(
  sadness        = "sadness_pred",
  disappointment = "disappointment_pred",
  disapproval    = "disapproval_pred",
  neutral        = "neutral_pred",
  fear           = "fear_pred",
  anger          = "anger_pred"
)

for (nm in names(target_emotions)) {
  analyze_emotion_topics(df, target_emotions[[nm]], nm, FIG_DIR)
}

#### 8. Distribution of Word Count ####
df_binned <- df %>%
  mutate(
    bin = case_when(
      word_count <= 10 ~ "0-10",
      word_count <= 20 ~ "11-20",
      word_count <= 30 ~ "21-30",
      word_count <= 40 ~ "31-40",
      word_count <= 50 ~ "41-50",
      word_count <= 60 ~ "51-60",
      word_count <= 70 ~ "61-70",
      word_count <= 80 ~ "71-80",
      word_count <= 90 ~ "81-90",
      TRUE ~ ">91"
    ),
    bin = factor(bin, levels = c("0-10", "11-20", "21-30", "31-40", "41-50",
                                 "51-60", "61-70", "71-80", "81-90", ">91"))
  ) %>%
  count(bin, name = "frequency") %>%
  mutate(percentage = round(frequency / sum(frequency) * 100, 1))

cat("\n=== Word Count Distribution (VY) ===\n")
print(df_binned)

p_wc <- ggplot(df_binned, aes(x = bin, y = frequency)) +
  geom_col(fill = "#5F7D8F", color = "white", linewidth = 0.8, width = 0.7, alpha = 0.85) +
  geom_text(aes(label = paste0(frequency, "\n(", percentage, "%)")),
            vjust = -0.3, size = 6, color = "black") +
  labs(
    title = "Distribution of word count in VulnYouth dataset",
    x = "Word count bins",
    y = "Frequency (N responses)"
  ) +
  theme_minimal(base_size = 20) +
  theme(
    plot.title = element_text(size = 26, hjust = 0.5),
    axis.text.x = element_text(size = 18),
    axis.text.y = element_text(size = 20),
    axis.title = element_text(size = 22),
    panel.grid.major.x = element_line(color = "gray90", linewidth = 0.5),
    panel.grid.minor = element_blank(),
    plot.margin = margin(20, 20, 40, 20)
  ) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.18)))

ggsave(
  filename = file.path(FIG_DIR, "word_count_binned.png"),
  plot = p_wc,
  width = 14, height = 9, dpi = 300, bg = "white"
)

cat("\nAll VY figures saved to:", FIG_DIR, "\n")
