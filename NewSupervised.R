
## 0. Packages
library(tidyverse)
library(quanteda)
library(quanteda.textmodels)

## 1. Use gold_df as labeled CHUNK data
# gold_df should already exist and look like:
# train_id | chunk_text | coderA | coderB | agree | gold_label

chunk_df <- gold_df %>%
  filter(!is.na(gold_label)) %>%     
  mutate(
    chunk_text = as.character(chunk_text),
    gold_label = as.integer(gold_label)
  )

nrow(chunk_df)
table(chunk_df$gold_label)


## 2. Train / test split

set.seed(176) 
n <- nrow(chunk_df)
train_index <- sample(1:n, size = round(0.8 * n))

train_dat <- chunk_df[train_index, ]
test_dat  <- chunk_df[-train_index, ]

## 3. Build DFM for training and test (CHUNK text) \

# Training corpus + dfm
corp_train <- corpus(
  train_dat,
  text_field  = "chunk_text",
  docid_field = "train_id"
)

toks_train <- tokens(
  corp_train,
  remove_punct   = TRUE,
  remove_numbers = TRUE
)

dfm_train <- dfm(toks_train)

# Response variable as factor for NB
y_train <- factor(train_dat$gold_label)

# Test corpus + dfm (match features to training)
corp_test <- corpus(
  test_dat,
  text_field  = "chunk_text",
  docid_field = "train_id"
)

toks_test <- tokens(
  corp_test,
  remove_punct   = TRUE,
  remove_numbers = TRUE
)

dfm_test <- dfm(toks_test)
dfm_test <- dfm_match(dfm_test, features = featnames(dfm_train))

## 4. Naive Bayes: fit + evaluate on held-out test set 

model_nb <- textmodel_nb(dfm_train, y_train)
summary(model_nb)

# Predictions on test chunks
pred_test <- predict(model_nb, newdata = dfm_test)

# Confusion matrix
cmat <- table(predicted = pred_test, actual = test_dat$gold_label)
cmat

# Accuracy
accuracy <- mean(pred_test == test_dat$gold_label)
accuracy

# Precision / recall / F1 for class "1" (Pro-Green)
if (!all(c("0","1") %in% rownames(cmat))) {
  message("One class is missing in predictions; precision/recall may be NA.")
} else {
  precision_1 <- cmat["1", "1"] / sum(cmat["1", ])
  recall_1    <- cmat["1", "1"] / sum(cmat[, "1"])
  f1_1        <- 2 * precision_1 * recall_1 / (precision_1 + recall_1)
  
  precision_1
  recall_1
  f1_1
}

## NB vs GOLD on ALL chunks

# 1. Make dfm_all conform to the training feature set
dfm_all_nb <- dfm_match(dfm_all, features = featnames(dfm_train))

# 2. Predict NB labels for ALL gold chunks
pred_all_nb <- predict(model_nb, newdata = dfm_all_nb)

# 3. Build evaluation data frame
nb_vs_gold <- data.frame(
  predicted = as.integer(as.character(pred_all_nb)),
  actual    = chunk_df$gold_label
)

# 4. Confusion matrix
tab_nb_gold <- table(
  Predicted = nb_vs_gold$predicted,
  Actual    = nb_vs_gold$actual
)
tab_nb_gold

# 5. accuracy / precision / recall / F1 for class 1
acc_nb_gold <- mean(nb_vs_gold$predicted == nb_vs_gold$actual)

tp_nb <- sum(nb_vs_gold$predicted == 1 & nb_vs_gold$actual == 1)
fp_nb <- sum(nb_vs_gold$predicted == 1 & nb_vs_gold$actual == 0)
fn_nb <- sum(nb_vs_gold$predicted == 0 & nb_vs_gold$actual == 1)

precision_nb_1 <- ifelse(tp_nb + fp_nb > 0, tp_nb / (tp_nb + fp_nb), NA_real_)
recall_nb_1    <- ifelse(tp_nb + fn_nb > 0, tp_nb / (tp_nb + fn_nb), NA_real_)
f1_nb_1        <- ifelse(
  is.na(precision_nb_1) | is.na(recall_nb_1) | (precision_nb_1 + recall_nb_1) == 0,
  NA_real_,
  2 * precision_nb_1 * recall_nb_1 / (precision_nb_1 + recall_nb_1)
)

cat("NB vs Gold (all 180 chunks)\n")
cat("  Accuracy:", round(acc_nb_gold,   3), "\n")
cat("  Precision (class 1):", round(precision_nb_1, 3), "\n")
cat("  Recall    (class 1):", round(recall_nb_1,    3), "\n")
cat("  F1        (class 1):", round(f1_nb_1,        3), "\n")


## 5. 5-fold cross-validation for Naive Bayes 

# Corpus + dfm for ALL chunks
corp_all <- corpus(
  chunk_df,
  text_field  = "chunk_text",
  docid_field = "train_id"
)

toks_all <- tokens(
  corp_all,
  remove_punct   = TRUE,
  remove_numbers = TRUE
)

dfm_all <- dfm(toks_all)

y_all <- factor(chunk_df$gold_label)

set.seed(176)
K <- 5
n_all <- nrow(chunk_df)

fold_id <- sample(rep(1:K, length.out = n_all))

acc_vec  <- numeric(K)
prec_vec <- numeric(K)
rec_vec  <- numeric(K)

for (k in 1:K) {
  cat("\n===== Fold", k, "=====\n")
  
  test_idx  <- which(fold_id == k)
  train_idx <- which(fold_id != k)
  
  dfm_tr <- dfm_all[train_idx, ]
  dfm_te <- dfm_all[test_idx, ]
  
  y_tr <- y_all[train_idx]
  y_te <- y_all[test_idx]
  
  nb_fit <- textmodel_nb(dfm_tr, y_tr)
  y_pred <- predict(nb_fit, newdata = dfm_te)
  
  tab <- table(pred = y_pred, true = y_te)
  print(tab)
  
  acc_vec[k] <- mean(y_pred == y_te)
  
  # precision / recall for positive class "1"
  tp <- ifelse("1" %in% rownames(tab) & "1" %in% colnames(tab), tab["1","1"], 0)
  fp <- ifelse("1" %in% rownames(tab), sum(tab["1", ]) - tp, 0)
  fn <- ifelse("1" %in% colnames(tab), sum(tab[, "1"]) - tp, 0)
  
  prec_vec[k] <- ifelse(tp + fp > 0, tp / (tp + fp), NA)
  rec_vec[k]  <- ifelse(tp + fn > 0, tp / (tp + fn), NA)
}

cv_results <- data.frame(
  fold      = 1:K,
  accuracy  = acc_vec,
  precision = prec_vec,
  recall    = rec_vec
)

cv_results

mean(acc_vec,  na.rm = TRUE)  # overall CV accuracy
mean(prec_vec, na.rm = TRUE)  # overall CV precision (class 1)
mean(rec_vec,  na.rm = TRUE)  # overall CV recall    (class 1)


## 7.Train final NB on ALL gold chunks

nb_final <- textmodel_nb(dfm_all, y_all)

## 8. Apply final NB classifier to ALL trimmed energy chunks

# 1. Make sure text is character and chunk_id exists
df_energy_chunks <- df_energy_chunks %>%
  mutate(
    text_chunk = as.character(text_chunk),
    chunk_id   = dplyr::row_number()  
  )

# 2. Build corpus + dfm for all energy chunks
corp_energy_all <- corpus(
  df_energy_chunks,
  text_field  = "text_chunk",
  docid_field = "chunk_id"
)

toks_energy_all <- tokens(
  corp_energy_all,
  remove_punct   = TRUE,
  remove_numbers = TRUE
)

dfm_energy_all <- dfm(toks_energy_all)

# 3. Match features to training vocabulary (dfm_all from gold chunks)
dfm_energy_all <- dfm_match(dfm_energy_all, features = featnames(dfm_all))

# 4. Predict labels for each chunk
pred_chunks <- predict(nb_final, newdata = dfm_energy_all)

# class 1 probability
pred_prob_chunks <- predict(
  nb_final,
  newdata = dfm_energy_all,
  type    = "probability"
)

# 5. Attach predictions back to df_energy_chunks
df_energy_chunks <- df_energy_chunks %>%
  mutate(
    ml_label_chunk = as.integer(as.character(pred_chunks)),
    prob_class1    = pred_prob_chunks[, "1"]
  )

table(df_energy_chunks$ml_label_chunk)
summary(df_energy_chunks$prob_class1)
