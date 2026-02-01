coderA <- c(
  coderA_1_20,
  coderA_21_60,
  coderA_61_120,
  coderA_121_200
)

coderB <- c(
  coderB_1_20,
  coderB_21_60,
  coderB_61_120,
  coderB_121_200
)

dfA$coderA <- coderA
dfA$coderB <- coderB

dfA$agree <- as.integer(dfA$coderA == dfA$coderB)
mean(dfA$agree)

dfA$gold_label <- ifelse(dfA$coderA == dfA$coderB, coderA, NA)

gold_df <- subset(dfA, !is.na(gold_label))
