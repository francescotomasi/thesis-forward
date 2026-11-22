#### Libraries ####
library(stm)
library(dplyr)
library(tidyr)
library(tidyverse)
library(tidytext)
library(data.table)
library(readxl)
library(wordcloud)

#### Data loading ####
ans <- read_excel("Data/english_documents_ott2026_OUTIP.xlsx")

cat("Loaded english_documents_ott2026_OUTIP.xlsx:", nrow(ans), "rows x", ncol(ans), "cols\n")

#### Preprocessing for STM ####
processed <- textProcessor(
  documents = ans$text,
  metadata = ans,
  removestopwords = TRUE,
  language = "eng",
  verbose = TRUE,
  custompunctuation = c("-"),
  customstopwords = c("precariousness"),
  stem = TRUE
)

out <- prepDocuments(processed$documents, processed$vocab, processed$meta)
docs <- out$documents
vocab <- out$vocab
meta <- out$meta

cat("Documents kept after prepDocuments:", length(docs), "\n")
cat("Documents removed:", length(out$docs.removed), "\n")
cat("Vocabulary size:", length(vocab), "\n")

#### 20-Topic Structural Topic Model ####
model_20K <- stm(
  documents = docs,
  vocab = vocab,
  K = 20,
  verbose = TRUE,
  seed = 2024
)

#### Build final datasets ####
topic_matrix <- as.data.frame(model_20K$theta)
colnames(topic_matrix) <- paste0("Topic_", seq_len(ncol(topic_matrix)))

df_topics <- bind_cols(meta, topic_matrix)
df_top <- bind_cols(meta %>% select(ident), topic_matrix)

#### Save outputs ####
dir.create("output/OUTIP", showWarnings = FALSE, recursive = TRUE)

saveRDS(model_20K, "output/OUTIP/stm_model_20K.rds")
saveRDS(out, "output/OUTIP/stm_prep_out.rds")

write.csv(df_topics, "output/OUTIP/df_topics_20K.csv", row.names = FALSE, fileEncoding = "UTF-8")
write.csv(df_top, "output/OUTIP/df_top_20K.csv", row.names = FALSE, fileEncoding = "UTF-8")

# Save readable summary of top words per topic and mean topic prevalence
topic_labels <- labelTopics(model_20K, n = 10)
mean_theta <- colMeans(model_20K$theta)

sink("output/OUTIP/topic_summary_20K.txt")
cat("=== Mean Topic Proportions (K = 20, OUTIP) ===\n")
for (k in order(mean_theta, decreasing = TRUE)) {
  cat(sprintf("Topic %2d: %.4f (%.2f%%)\n", k, mean_theta[k], mean_theta[k] * 100))
}
cat("\n=== Top 10 Words per Topic ===\n")
print(topic_labels)
sink()

cat("\n=== Top 7 Words per Topic (FREX & Highest Prob) ===\n")
print(labelTopics(model_20K, n = 7))
cat("\nAll outputs saved in output/OUTIP/.\n")
