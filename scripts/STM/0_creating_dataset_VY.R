#### 0 — Creating the Unified Dataset (VY) ####
# Merges the dimensional model (EmoBank — VAD predictions),
# componential model (GoEmotions — categorical emotion predictions),
# and the STM + survey dataset (output/VY/df_topics_20K.csv) on ID.

library(readxl)
library(dplyr)
library(writexl)

#### 1. Configuration ####
BASE_DIR <- "../.."

PATH_DIMENSIONAL   <- file.path(BASE_DIR, "scripts/dimensional_model/output/english_documents_part1_emobank_VY.xlsx")
PATH_COMPONENTIAL  <- file.path(BASE_DIR, "scripts/componential_model/output/english_documents_part2_goemotions_VY.xlsx")
PATH_SURVEY_TOPICS <- "output/VY/df_topics_20K.csv"

OUTPUT_DIR      <- "output/VY"
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

OUTPUT_PATH_XLSX <- file.path(OUTPUT_DIR, "merged_BERT_survey_VY.xlsx")
OUTPUT_PATH_CSV  <- file.path(OUTPUT_DIR, "merged_BERT_survey_VY.csv")

cat("Dimensional model file :", PATH_DIMENSIONAL, "\n")
cat("Componential model file:", PATH_COMPONENTIAL, "\n")
cat("Survey + Topics file   :", PATH_SURVEY_TOPICS, "\n")
cat("Output paths           :", OUTPUT_PATH_XLSX, "|", OUTPUT_PATH_CSV, "\n\n")

#### 2. Load the BERT model outputs ####
df_dimensional  <- read_excel(PATH_DIMENSIONAL)
df_componential <- read_excel(PATH_COMPONENTIAL)

cat("Dimensional  (EmoBank)    — shape:", nrow(df_dimensional), "x", ncol(df_dimensional), "\n")
cat("Componential (GoEmotions) — shape:", nrow(df_componential), "x", ncol(df_componential), "\n")

#### 3. Sanity checks & combine BERT predictions ####
stopifnot(
  "Row count mismatch between dimensional and componential outputs!" =
    nrow(df_dimensional) == nrow(df_componential),
  "Text columns are not aligned between dimensional and componential outputs!" =
    all(df_dimensional$text == df_componential$text)
)
cat("Both BERT outputs have", nrow(df_dimensional), "rows with identical text alignment.\n")

# Combine by position (drop duplicate 'text' column from componential)
df_bert <- bind_cols(
  df_dimensional,
  df_componential %>% select(-text)
)
cat("Combined BERT predictions — shape:", nrow(df_bert), "x", ncol(df_bert), "\n")

#### 4. Load the Survey + STM Topics dataset ####
df_survey_topics <- read.csv(PATH_SURVEY_TOPICS, stringsAsFactors = FALSE)
if ("X" %in% names(df_survey_topics)) {
  df_survey_topics <- df_survey_topics %>% select(-X)
}

cat("Survey + Topics data — shape:", nrow(df_survey_topics), "x", ncol(df_survey_topics), "\n")
cat("Unique IDs                  :", n_distinct(df_survey_topics$ID), "\n")
cat("ID range                    :", min(df_survey_topics$ID), "–", max(df_survey_topics$ID), "\n")

#### 5. Inner join on ID (avoiding duplicate metadata columns) ####
# Drop overlapping metadata columns from df_bert (already present in df_survey_topics with clean encoding)
overlap_cols <- setdiff(intersect(names(df_bert), names(df_survey_topics)), "ID")
df_bert_clean <- df_bert %>% select(-all_of(overlap_cols))

df_merged <- inner_join(df_bert_clean, df_survey_topics, by = "ID")

stopifnot(
  "Unexpected row count after inner join!" = nrow(df_merged) == nrow(df_survey_topics)
)

cat("\nMerged dataset shape :", nrow(df_merged), "x", ncol(df_merged), "\n")
cat("  BERT columns       :", ncol(df_bert_clean) - 1, "\n")
cat("  Survey+STM columns :", ncol(df_survey_topics) - 1, "\n")
cat("  Total (with ID)    :", ncol(df_merged), "\n")

#### 6. Quick VAD summary statistics ####
cat("\nVAD summary statistics:\n")
print(summary(df_merged[, c("V_predicted", "A_predicted", "D_predicted")]))

#### 7. Save the unified dataset ####
write_xlsx(df_merged, OUTPUT_PATH_XLSX)
write.csv(df_merged, OUTPUT_PATH_CSV, row.names = FALSE, fileEncoding = "UTF-8")

# Verify saved files
df_check <- read_excel(OUTPUT_PATH_XLSX)
stopifnot(
  nrow(df_check) == nrow(df_merged),
  ncol(df_check) == ncol(df_merged)
)

cat("\nSaved and verified:\n")
cat("  XLSX:", OUTPUT_PATH_XLSX, sprintf("(%.2f MB)\n", file.info(OUTPUT_PATH_XLSX)$size / 1024^2))
cat("  CSV :", OUTPUT_PATH_CSV, sprintf("(%.2f MB)\n", file.info(OUTPUT_PATH_CSV)$size / 1024^2))
