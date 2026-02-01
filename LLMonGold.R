
## LLM EVALUATION ON GOLD CHUNKS (gold_df)

if ("chunk_text" %in% names(gold_df) && !"text_chunk" %in% names(gold_df)) {
  gold_df <- gold_df %>% rename(text_chunk = chunk_text)
}

gold_df <- gold_df %>%
  mutate(
    text_chunk = as.character(text_chunk),
    eval_id    = dplyr::row_number()
  )

gold_eval <- gold_df

n_rows_eval    <- nrow(gold_eval)
batch_size_eval <- 20
n_batches_eval  <- ceiling(n_rows_eval / batch_size_eval)

gold_eval <- gold_eval %>%
  mutate(llm_label = NA_integer_)

cat("Sending", n_rows_eval, "gold chunks to Gemini in",
    n_batches_eval, "batches of", batch_size_eval, "\n")

for (b in seq_len(n_batches_eval)) {
  cat("\n=== [EVAL] Batch", b, "of", n_batches_eval, "===\n")
  
  idx_start <- (b - 1) * batch_size_eval + 1
  idx_end   <- min(b * batch_size_eval, n_rows_eval)
  idx_vec   <- idx_start:idx_end

  
  text_vec <- gold_eval$text_chunk[idx_vec]
  
  labs_b <- classify_batch(
    text_vec,
    verbose = (b == 1)
  )
  
  gold_eval$llm_label[idx_vec] <- labs_b
  
  Sys.sleep(7)
}

table(gold_eval$llm_label, useNA = "ifany")

# 2. LLM vs GOLD: confusion matrix

df_eval_gold <- gold_eval %>%
  filter(!is.na(gold_label), !is.na(llm_label))

cat("\n=== LLM vs Gold (gold_df evaluation set) ===\n")
tab_llm_gold <- table(
  predicted = df_eval_gold$llm_label,
  actual    = df_eval_gold$gold_label
)
print(tab_llm_gold)

acc_llm_gold <- mean(df_eval_gold$llm_label == df_eval_gold$gold_label)
cat("Accuracy (LLM vs gold):", round(acc_llm_gold, 3), "\n")

# Precision / recall / F1 for class 1 (pro-green)
tp <- sum(df_eval_gold$llm_label == 1 & df_eval_gold$gold_label == 1)
fp <- sum(df_eval_gold$llm_label == 1 & df_eval_gold$gold_label == 0)
fn <- sum(df_eval_gold$llm_label == 0 & df_eval_gold$gold_label == 1)

precision_1_llm <- ifelse(tp + fp > 0, tp / (tp + fp), NA_real_)
recall_1_llm    <- ifelse(tp + fn > 0, tp / (tp + fn), NA_real_)
f1_1_llm        <- ifelse(
  is.na(precision_1_llm) | is.na(recall_1_llm) | (precision_1_llm + recall_1_llm) == 0,
  NA_real_,
  2 * precision_1_llm * recall_1_llm / (precision_1_llm + recall_1_llm)
)

cat("Precision (class 1):", round(precision_1_llm, 3), "\n")
cat("Recall    (class 1):", round(recall_1_llm,    3), "\n")
cat("F1        (class 1):", round(f1_1_llm,        3), "\n")

# 3. LLM vs Naive Bayes on gold set

if ("ml_label_chunk" %in% names(gold_eval)) {
  cat("\n=== LLM vs NB on gold_df ===\n")
  
  df_eval_nb <- gold_eval %>%
    filter(!is.na(llm_label), !is.na(ml_label_chunk))
  
  tab_llm_nb_gold <- table(
    llm = df_eval_nb$llm_label,
    nb  = df_eval_nb$ml_label_chunk
  )
  print(tab_llm_nb_gold)
  
  agree_llm_nb_gold <- mean(df_eval_nb$llm_label == df_eval_nb$ml_label_chunk)
  cat("Simple agreement (LLM vs NB, gold_df):", round(agree_llm_nb_gold, 3), "\n")
}