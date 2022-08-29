Sys.setlocale("LC_NUMERIC","en_US.UTF-8")

library(scmamp)
library(yaml)

paths <- snakemake@input

d <- read.csv(paths[[1]])
d["X"] <- NULL
for (idx in 2:length(paths)) {
  df = read.csv(paths[[idx]])
  df["X"] <- NULL
  d <- rbind(d, df)
}

d <- d[(d$cat != "mvo"), ]

for (c in unique(d[["cat"]])) {
  d[d["cat"] == c, "idx"] <- 1:sum(d["cat"] == c)
}

df_cats <- reshape(d[, -2:-4], idvar = "idx", timevar = "cat", direction = "wide")
df_cats$idx <- NULL
colnames(df_cats) <- gsub("mcc.", "", colnames(df_cats))

# plotCD(results.matrix = df_cats, alpha = 0.05)

ar_cats <- sort(colMeans(rankMatrix(df_cats)))

nm_cats <- nemenyiTest(df_cats)
rownames(nm_cats$diff.matrix) <- colnames(nm_cats$diff.matrix)

lists <- apply(
  X = expand.grid(unique(d[["model"]]), unique(d[["meta_model"]])),
  MARGIN = 1,
  FUN = function(row) {
    df <- data.frame(d[d["model"] == row[1] & d["meta_model"] == row[2], "mcc"])
    colnames(df) <- paste0(row[1], "_", row[2])
    return(df)
  }
)

df_models <- do.call(cbind, lists)

ar_models <- sort(colMeans(rankMatrix(df_models)))

nm_models <- nemenyiTest(df_models)
rownames(nm_models$diff.matrix) <- colnames(nm_models$diff.matrix)

lres <- list(
  cats = list(
    cd = nm_cats$statistic,
    average_ranking = ar_cats,
    names = names(ar_cats)
  ),
  models = list(
    cd = nm_models$statistic,
    average_ranking = ar_models,
    names = names(ar_models)
  )
)

write_yaml(lres, snakemake@output[[1]])

# test.res <- postHocTest(data = d_tmp, test ='friedman', correct ='bergmann')
# average.ranking <- colMeans(rankMatrix(d_tmp))
# drawAlgorithmGraph(
#   pvalue.matrix = test.res$corrected.pval,
#   mean.value = average.ranking,
#   font.size = 8,
#   node.width = 3,
#   node.height = 1
# )
#
#
#
# print(test.res$corrected.pval)





