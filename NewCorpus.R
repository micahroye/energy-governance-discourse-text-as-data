# Load packages
library(tidyverse) # for data manipulation
library(quanteda) # for text + docbars handling
library(stringr) # regex / string utilities
library(tokenizers) # used for splitting into words for 200-word chunks
library(lubridate) # date handling for shocks

# load the council corpus
setwd("~/Desktop/UCSD/UCSD25-26/FA25/POLI 176/dataverse_files/data")
load("council_corpus_apsr.RData")

# start from the corpus object
corp_all <- council_all_final

# convert corpus + docvars to a data frame
df_all <- convert(corp_all, to = "data.frame")

names(df_all)[1:10]
head(df_all[, 1:5])

range(df_all$date_correct, na.rm = TRUE)

## Energy keyword list (pro + anti + neutral terms)

# Fossil / conventional energy
fossil_keywords <- c(
  "gas", "natural gas", "lng", "pipeline", "pipelines",
  "nord stream", "oil", "petroleum", "diesel", "shale",
  "fracking", "coal", "lignite", "fossil", "fossil fuel", "fossil fuels",
  "nuclear", "atomic", "reactor", "reactors", "uranium"
)

# Renewables / green transition
renewable_keywords <- c(
  "renewable", "renewables", "solar", "photovoltaic",
  "wind", "offshore wind", "hydro", "hydropower",
  "geothermal", "biofuel", "biofuels", "biomass",
  "green transition", "clean energy", "energy transition",
  "decarbonisation", "decarbonization", "carbon-free", "carbon free",
  "net-zero", "net zero", "zero-carbon", "zero carbon",
  "emissions reduction", "climate neutrality", "climate-neutral",
  "fit for 55", "green deal", "european green deal"
)

# Energy Security
security_keywords <- c(
  "energy security", "security of supply", "dependencies", "dependency",
  "imports", "energy dependence", "energy dependency", "energy diversification",
  "interruption", "resilience", "reliability",
  "strategic autonomy", "energy autonomy",
  "emissions", "ghg", "greenhouse gas", "greenhouse gases",
  "carbon", "co2", "methane", "carbon capture", "ccs", "carbon pricing",
  "paris agreement", "kyoto", "cop21", "cop26", "cop", 
  "climate and energy package", "energy union", "repowereu",
  "ets", "emissions trading", "carbon market", "interconnector", "interconnectors",
  "grid", "energy market", "market integration",
  "capacity market", "infrastructure", "distribution", "transmission",
  "russian gas", "gazprom", "ukraine", "energy shock",
  "energy crisis", "supply shock", "gas cutoff",
  "energy system", "energy supply", "energy mix",
  "energy efficiency", "sustainable energy", "sustainability",
  "investment in energy", "transition finance",
  "critical minerals", "strategic reserves"
)

energy_keywords <- c(fossil_keywords,
                     renewable_keywords,
                     security_keywords)

# build patterns for each group
p_fossil     <- paste(fossil_keywords,     collapse = "|")
p_renewable  <- paste(renewable_keywords,  collapse = "|")
p_security   <- paste(security_keywords,   collapse = "|")
p_energy_all <- paste(energy_keywords,     collapse = "|")

df_flagged <- df_all %>%
  mutate(
    has_fossil = str_detect(
      text,
      regex(p_fossil, ignore_case = TRUE)
    ),
    has_renewable = str_detect(
      text,
      regex(p_renewable, ignore_case = TRUE)
    ),
    has_security = str_detect(
      text,
      regex(p_security, ignore_case = TRUE)
    ),
    has_energy_keyword = has_fossil | has_renewable | has_security
  )

dim(df_flagged)                     # same number of rows as df_all
colnames(df_flagged)[1:15]          # confirm new columns exist

# Proportion of paragraphs with any energy content
mean(df_flagged$has_energy_keyword, na.rm = TRUE)

# Look at some examples
df_flagged %>%
  filter(has_energy_keyword) %>%
  slice_sample(n = 5) %>%
  select(doc_id, date_correct, Council_Config_final, text) %>%
  View()

# Compare fossil vs renewables hits
mean(df_flagged$has_fossil, na.rm = TRUE)
mean(df_flagged$has_renewable, na.rm = TRUE)

df_energy <- df_flagged %>%
  filter(has_energy_keyword) %>%
  dplyr::select(
    doc_id,
    text,
    date_correct,
    Debate_Topic,
    Council_Config_final,
    has_fossil,
    has_renewable,
    has_security,
    has_energy_keyword
  )

# keep the year + shock indicators (very important for your design)
df_energy <- df_energy %>%
  mutate(
    year = lubridate::year(date_correct),
    post_fukushima = year >= 2011,
    post_ukraine_2014 = year >= 2014,
    post_paris = year >= 2016
  )

dim(df_energy)
range(df_energy$date_correct, na.rm = TRUE)

# --------------------------------------------------
# 3. Chunk energy-related text into 200-word windows
# --------------------------------------------------

chunk_size <- 200  # words per chunk

make_chunks <- function(row) {
  # row is a 1-row tibble from df_energy
  words <- tokenizers::tokenize_words(row$text, simplify = TRUE)
  
  if (length(words) == 0) {
    return(NULL)
  }
  
  n_chunks <- ceiling(length(words) / chunk_size)
  
  chunks <- map_chr(1:n_chunks, function(i) {
    start <- (i - 1) * chunk_size + 1
    end   <- min(i * chunk_size, length(words))
    paste(words[start:end], collapse = " ")
  })
  
  tibble(
    doc_id             = row$doc_id,
    date_correct       = row$date_correct,
    year               = row$year,
    Debate_Topic       = row$Debate_Topic,
    Council_Config_final = row$Council_Config_final,
    has_fossil         = row$has_fossil,
    has_renewable      = row$has_renewable,
    has_security       = row$has_security,
    has_energy_keyword = row$has_energy_keyword,
    post_fukushima     = row$post_fukushima,
    post_ukraine_2014  = row$post_ukraine_2014,
    post_paris         = row$post_paris,
    chunk_index        = seq_len(n_chunks),
    chunk_id           = paste0(row$doc_id, "_chunk", seq_len(n_chunks)),
    text_chunk         = chunks
  )
}

df_energy_chunks <- df_energy %>%
  mutate(row_id = row_number()) %>%
  group_by(row_id) %>%
  group_modify(~ make_chunks(.x)) %>%
  ungroup() %>%
  select(-row_id)

dim(df_energy_chunks)
head(df_energy_chunks, 3)

