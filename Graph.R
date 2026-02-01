
library(tidyverse)
library(dplyr)
library(ggplot2)

# Make surechunk-level dataset has model predictions added

df_chunks_ml <- df_energy_chunks  
pro_green_by_year <- df_chunks_ml %>%
  group_by(year) %>%
  summarise(
    n_chunks        = n(),                                
    n_green_chunks  = sum(ml_label_chunk == 1, na.rm = TRUE),
    share_green     = mean(ml_label_chunk == 1, na.rm = TRUE),
    avg_prob_green  = mean(prob_class1, na.rm = TRUE)      
  )

pro_green_by_year


## 1. Define the shocks
shocks <- tibble::tribble(
  ~year, ~label,
  2011,  "Fukushima\nnuclear crisis",
  2014,  "Ukraine/Russia\ngas crisis",
  2015,  "Paris\nAgreement"
)

y_max <- max(pro_green_by_year$avg_prob_green, na.rm = TRUE)

## 2. Plot time series
ggplot(pro_green_by_year, aes(x = year, y = avg_prob_green)) +
  geom_line(size = 1.1, color = "#7570b3") +
  geom_point(size = 2) +
  
  # vertical dashed lines at shock years
  geom_vline(data = shocks,
             aes(xintercept = year),
             linetype = "dashed",
             color = "red",
             alpha = 0.6) +
  
  # text labels for shocks
  geom_text(data = shocks,
            aes(x = year,
                y = y_max * 1.05, 
                label = label),
            angle = 90,
            vjust = -0.2,
            size = 3) +
  
  scale_y_continuous(expand = expansion(mult = c(0, 0.2))) +
  
  theme_minimal() +
  labs(
    title = "Average Probability of Pro-Green Framing Over Time",
    x     = "Year",
    y     = "Average P(Pro-Green)"
  )

## 5-fold Cross-Validation Performance

cv_long <- cv_results %>%
  tidyr::pivot_longer(
    cols = c(accuracy, precision, recall),
    names_to  = "metric",
    values_to = "value"
  )

ggplot(cv_long, aes(x = factor(fold), y = value, group = metric)) +
  geom_line(aes(color = metric)) +
  geom_point(aes(color = metric)) +
  theme_minimal() +
  labs(
    title = "5-Fold Cross-Validation Performance (Naive Bayes)",
    x     = "Fold",
    y     = "Score",
    color = "Metric"
  ) +
  ylim(0, 1)


# LLM vs Gold Confusion Matrix
cm_llm <- as.data.frame(tab_llm_gold)
names(cm_llm) <- c("Predicted", "Actual", "Freq")

ggplot(cm_llm, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white", size = 6) +
  scale_fill_gradient(low = "#4575b4", high = "#d73027") +
  theme_minimal() +
  labs(title = "Confusion Matrix: LLM vs Gold Labels",
       x = "Actual (Gold)", y = "Predicted (LLM)")

# NB vs Gold Confusion Matrix
cm_nb <- as.data.frame(tab_nb_gold)
names(cm_nb) <- c("Predicted", "Actual", "Freq")

ggplot(cm_nb, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white", size = 6) +
  scale_fill_gradient(low = "#4575b4", high = "#d73027") +
  theme_minimal() +
  labs(
    title = "Confusion Matrix: Naive Bayes vs Gold Labels (All Gold Chunks)",
    x = "Actual (Gold)",
    y = "Predicted (NB)"
  )


# LLM metrics

llm_results <- tibble(
  metric = c("accuracy", "precision", "recall", "F1"),
  value  = c(0.889, 0.652, 0.882, 0.750)
)

llm_results <- llm_results %>%
  mutate(fold = 1)

ggplot(llm_results, aes(x = factor(fold), y = value, group = metric)) +
  geom_line(aes(color = metric), linewidth = 1.1) +
  geom_point(aes(color = metric), size = 3) +
  theme_minimal() +
  labs(
    title = "Performance Metrics (LLM – Gemini)",
    x     = "Model",
    y     = "Score",
    color = "Metric"
  ) +
  scale_x_discrete(labels = "LLM (Gemini)") +
  ylim(0, 1)