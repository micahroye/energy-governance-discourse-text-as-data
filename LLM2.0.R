
## LLM WORKFLOW WITH GEMINI (BATCHED, STRATIFIED SAMPLE)
##   – Application to df_energy_chunks
##   – ~800 sampled chunks, 40 per API call (~20 calls)

## 0. Packages
library(tidyverse)
library(httr)
library(jsonlite)
library(stringr)

## 0a. SET GEMINI API KEY HERE
Sys.setenv(GEMINI_API_KEY = "Gemini Key")

df_energy_chunks <- df_energy_chunks %>%
  mutate(
    text_chunk = as.character(text_chunk),
    chunk_id   = dplyr::row_number()
  )

gemini_call <- function(prompt,
                        model = "gemini-2.5-flash-lite") {
  key <- Sys.getenv("GEMINI_API_KEY")
  if (identical(key, "") || is.null(key)) {
    stop("GEMINI_API_KEY is not set. ",
         "Run Sys.setenv(GEMINI_API_KEY = 'YOUR_KEY') or use ~/.Renviron.")
  }
  
  url <- paste0(
    "https://generativelanguage.googleapis.com/v1beta/models/",
    model,
    ":generateContent"
  )
  
  body <- list(
    contents = list(
      list(
        role  = "user",
        parts = list(list(text = prompt))
      )
    )
  )
  
  resp <- httr::POST(
    url,
    query  = list(key = key),
    body   = body,
    encode = "json"
  )
  
  if (httr::http_error(resp)) {
    stop("Gemini API request failed: HTTP ", httr::status_code(resp))
  }
  
  parsed <- httr::content(resp, as = "parsed")
  
  txt <- NULL
  try({
    txt <- parsed$candidates[[1]]$content$parts[[1]]$text
  }, silent = TRUE)
  
  if (is.null(txt)) {
    warning("Could not find text field in Gemini response; returning fallback.")
    return(paste(capture.output(str(parsed)), collapse = "\n"))
  }
  
  as.character(txt)
}

## PROMPT BUILDING + PARSING FOR BATCHES

build_batch_prompt <- function(text_vec) {
  n <- length(text_vec)
  
  header <- paste0(
    "You are coding political text about energy.\n\n",
    "Label each chunk as:\n",
    "0 = NOT clearly pro-green / pro-renewables / pro-decarbonization.\n",
    "1 = Clearly pro-green / pro-renewables / pro-decarbonization.\n\n",
    "Rules:\n",
    "- Focus on whether the speaker clearly supports renewable energy, ",
    "low-carbon transition, or decarbonization.\n",
    "- If the chunk is neutral, mixed, or unclear, label it 0.\n",
    "- Do NOT explain your reasoning.\n",
    "- Your reply MUST contain exactly ", n, " lines.\n",
    "- Line i must contain ONLY a single character: 0 or 1, for chunk i.\n",
    "- Do not add any other text.\n\n",
    "Here are the chunks:\n\n"
  )
  
  chunks_part <- paste0(
    purrr::map2_chr(
      seq_along(text_vec),
      text_vec,
      ~ paste0("CHUNK ", .x, ":\n", .y, "\n\n")
    ),
    collapse = ""
  )
  
  paste0(header, chunks_part, "Now output the ", n, " labels.\n")
}

parse_batch_labels <- function(raw_text, n_expected) {
  if (!is.character(raw_text)) {
    raw_text <- as.character(raw_text)
  }
  if (length(raw_text) > 1) {
    raw_text <- paste(raw_text, collapse = "\n")
  }
  
  lines <- unlist(strsplit(raw_text, "\n"))
  lines <- stringr::str_trim(lines)
  lines <- lines[nchar(lines) > 0]
  
  labels_str <- lines[grepl("^[01]$", lines)]
  
  if (length(labels_str) < n_expected) {
    warning("Expected ", n_expected, " labels, found ",
            length(labels_str), ". Filling rest with NA.")
    labels_str <- c(labels_str, rep(NA_character_, n_expected - length(labels_str)))
  }
  
  as.integer(labels_str[seq_len(n_expected)])
}

classify_batch <- function(text_vec,
                           model   = "gemini-2.5-flash-lite",
                           verbose = FALSE) {
  prompt <- build_batch_prompt(text_vec)
  raw    <- gemini_call(prompt, model = model)
  
  if (verbose) {
    cat("\nRaw Gemini batch output:\n", raw, "\n\n")
  }
  
  parse_batch_labels(raw, length(text_vec))
}

## STRATIFIED SAMPLING BY YEAR 

set.seed(176)

# Target total number of chunks to send to Gemini:
# 40 chunks/call * 20 calls = ~800
target_total <- 800  

n_years    <- dplyr::n_distinct(df_energy_chunks$year)  # should be 7 (2010–2016)
n_per_year <- ceiling(target_total / n_years)

cat("Years in data:", n_years, "\n")
cat("Target total:", target_total, "=> ~", n_per_year, "per year\n")

df_llm_sample <- df_energy_chunks %>%
  group_by(year) %>%
  group_modify(
    ~ dplyr::slice_sample(.x, n = min(n_per_year, nrow(.x)))
  ) %>%
  ungroup() %>%
  mutate(llm_label = NA_integer_)

n_rows   <- nrow(df_llm_sample)
cat("Actual sample size sent to Gemini:", n_rows, "chunks\n")

## LOOP OVER BATCHES (40 CHUNKS PER CALL)

batch_size <- 40
n_batches  <- ceiling(n_rows / batch_size)

cat("Batch size:", batch_size, "=>", n_batches, "API calls\n")

for (b in seq_len(n_batches)) {
  cat("\n=== Batch", b, "of", n_batches, "===\n")
  
  idx_start <- (b - 1) * batch_size + 1
  idx_end   <- min(b * batch_size, n_rows)
  idx_vec   <- idx_start:idx_end
  

  text_vec <- df_llm_sample$text_chunk[idx_vec]
  
  # Only show raw output for the very first batch
  any_labeled_before <- any(!is.na(df_llm_sample$llm_label))
  labs_b <- classify_batch(text_vec, verbose = (!any_labeled_before))
  
  df_llm_sample$llm_label[idx_vec] <- labs_b
  
  Sys.sleep(7)
}

table(df_llm_sample$llm_label, useNA = "ifany")

## COMPARE GEMINI TO NAIVE BAYES

if ("ml_label_chunk" %in% names(df_llm_sample)) {
  cat("\n=== LLM vs Naive Bayes on sampled df_energy_chunks ===\n")
  
  df_eval_nb <- df_llm_sample %>%
    filter(!is.na(llm_label), !is.na(ml_label_chunk))
  
  tab_llm_nb <- table(
    llm = df_eval_nb$llm_label,
    nb  = df_eval_nb$ml_label_chunk
  )
  print(tab_llm_nb)
  
  agreement_llm_nb <- mean(df_eval_nb$llm_label == df_eval_nb$ml_label_chunk)
  cat("Simple agreement (LLM vs NB, sampled chunks):",
      round(agreement_llm_nb, 3), "\n")
}

## LLM-BASED TIME SERIES ON SAMPLE

llm_year <- df_llm_sample %>%
  filter(!is.na(llm_label)) %>%
  group_by(year) %>%
  summarise(
    n_chunks_llm    = n(),
    share_green_llm = mean(llm_label == 1, na.rm = TRUE),
    .groups = "drop"
  )

llm_year