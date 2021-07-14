library(broom)

d <- read.csv(snakemake@input[[1]])
# d <- read.csv("data/temp/avp_amppred/kappa_error_res/plot_data.csv")

res.man <- manova(cbind(d$x, d$y) ~ model, data = d)

manova_summary <- summary(res.man)
write.csv(tidy(res.man), snakemake@output[[1]])

manova_summary_aov <- summary.aov(res.man)
write.csv(rbind(
  data.frame(
    manova_summary_aov[[1]],
    response=names(manova_summary_aov[1])
  ),
  data.frame(
    manova_summary_aov[[2]],
    response=names(manova_summary_aov[2])
  )
), snakemake@output[[2]])

df_res <- do.call(rbind, lapply(unique(d[["model"]]), function(m) {
  d_tmp <- d[d["model"] == m, ]
  d_tmp <- d_tmp[
    (d_tmp$ensemble_best == "False") &
    (d_tmp$ensemble_rand == "False") &
    (d_tmp$ensemble_chull == "False") &
    (d_tmp$ensemble_pfront == "False") &
    (d_tmp$chull_complete == -1)
  , ]
  # d_tmp <- d_tmp[sample(nrow(d_tmp), 1000), ]
  data.frame(kappa = d_tmp[ ,"x"], error = d_tmp[ ,"y"], model = m)
}))

anova_kappa_aov <- aov(df_res$kappa ~ df_res$model)
write.csv(tidy(anova_kappa_aov), snakemake@output[[3]])

anova_kappa_tukey_hsd <- TukeyHSD(aov(df_res$kappa ~ df_res$model))
write.csv(tidy(anova_kappa_tukey_hsd), snakemake@output[[4]])

anova_error_aov <- aov(df_res$error ~ df_res$model)
write.csv(tidy(anova_error_aov), snakemake@output[[5]])

anova_error_tukey_hsd <- TukeyHSD(aov(df_res$error ~ df_res$model))
write.csv(tidy(anova_error_tukey_hsd), snakemake@output[[6]])

### areas

df_res <- do.call(
  rbind,
  lapply(
    snakemake@input[2:5],
    # Sys.glob("data/temp/avp_amppred/areas/*/res.csv"),
    read.csv
  )
)

anova_area_aov <- aov(df_res$area ~ df_res$model)
write.csv(tidy(anova_area_aov), snakemake@output[[7]])

anova_area_tukey_hsd <- TukeyHSD(aov(df_res$area ~ df_res$model))
write.csv(tidy(anova_area_tukey_hsd), snakemake@output[[8]])
