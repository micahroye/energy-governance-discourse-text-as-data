library(dplyr)
library(readr)
library(stringr)
library(irr)   

# --------------------------------------------------
# 1. Start from df_energy_chunks (200-word chunks)
# --------------------------------------------------

df_energy_chunks$text_chunk <- as.character(df_energy_chunks$text_chunk)

nrow(df_energy_chunks)  # optional


# --------------------------------------------------
# 2. Inspect word counts per chunk
# --------------------------------------------------

df_energy_chunks <- df_energy_chunks %>%
  mutate(
    n_words = str_count(text_chunk, "\\S+")
  )

summary(df_energy_chunks$n_words)


# 2.A. Tag "likely positive" chunks for stratified sampling

renewable_pattern <- regex(
  "renewable|renewables|solar|wind|hydro|hydropower|geothermal|biofuel|biofuels|biomass|green transition|clean energy|energy transition|decarbon|net[- ]?zero|zero[- ]?carbon|emissions reduction|climate[- ]?neutral|green deal|fit for 55",
  ignore_case = TRUE
)

df_energy_chunks <- df_energy_chunks %>%
  mutate(
    likely_positive = str_detect(text_chunk, renewable_pattern)
  )

table(df_energy_chunks$likely_positive) 


# 3. Stratified sample 200 chunks for hand-coding

set.seed(176)

n_pos <- 100
n_neg <- 100

positive_pool <- df_energy_chunks %>% filter(likely_positive)
negative_pool <- df_energy_chunks %>% filter(!likely_positive)

positive_sample <- positive_pool %>% sample_n(n_pos)
negative_sample <- negative_pool %>% sample_n(n_neg)

# randomize after combining
train_chunks <- bind_rows(positive_sample, negative_sample) %>%
  sample_frac(1) %>%                
  mutate(train_id = row_number()) %>%
  select(
    train_id,
    chunk_id,
    doc_id,
    date_correct,
    year,
    Debate_Topic,
    Council_Config_final,
    has_fossil,
    has_renewable,
    has_security,
    has_energy_keyword,
    post_fukushima,
    post_ukraine_2014,
    post_paris,
    n_words,
    likely_positive,
    text_chunk
  )

write_csv(
  train_chunks,
  "training_chunks_200_with_metadata.csv"
)


# 5. Clean coder version

coder_file <- train_chunks %>%
  transmute(
    train_id = train_id,              
    chunk_text = str_squish(text_chunk),
    label = ""                        
  )

write_csv(coder_file, "coderA_200_chunks.csv")
write_csv(coder_file, "coderB_200_chunks.csv")

dfA <- read_csv("coderA_200_chunks.csv")

writeLines(
  paste0(
    "CHUNK ", dfA$train_id, ":\n",
    dfA$chunk_text,
    "\n------------------------\n"
  ),
  "all_chunks_with_ids.txt"
)


# Cohen's kappa 
kappa_result <- kappa2(dfA[, c("coderA", "coderB")], 
                       weight = "unweighted")

kappa_result