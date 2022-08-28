import pandas as pd
import altair as alt

from glob import glob

df_res = pd.DataFrame()
for p in glob("data/ensembles_res/*/*/*.csv"):
    df_tmp = pd.read_csv(p, index_col=0)
    df_res = pd.concat([df_res, df_tmp])

df_res = df_res.loc[df_res.cat != "rand_single_best"]

df_tmp = df_res.loc[df_res.model == "bayes"]
df_tmp["model"] = "mlp"

df_res = pd.concat([df_res, df_tmp])

df_tmp2 = df_res.loc[df_res.cat == "single_best"]
df_tmp2.loc[df_tmp2.meta_model == "stacking", "meta_model"] = "Top 1"
df_tmp2.loc[df_tmp2.meta_model == "voting_soft", "meta_model"] = "Top 2"
df_tmp2.loc[df_tmp2.meta_model == "voting_hard", "meta_model"] = "Top 3"

df_res = df_res.loc[df_res.cat != "single_best"]

# df_res = pd.concat([df_res, df_tmp2])

c1 = alt.Chart(df_res).mark_boxplot(
    size=8, color="#000000", opacity=1.0, outliers={"size": 0}, median=False
).encode(
    x=alt.X("meta_model:N", title=None, axis=alt.Axis(labelAngle=-35, grid=True)),
    y=alt.Y(
        "mcc:Q",
        scale=alt.Scale(domain=[0.0, 1.0]),
        axis=alt.Axis(values=[0.1, 0.3, 0.5, 0.7, 0.9], title=None)
    ),
    # color=alt.Color("cat:N", title="Pruning")
).properties(
    width=100,
    height=100
).facet(
    row=alt.Row("model:N", title=None),
    column=alt.Column("cat:N", title=None, sort=["pfront", "chull", "mvo", "best", "rand", "single_best"]),
    spacing=1
)

c2 = alt.Chart(df_tmp2).mark_boxplot(
    size=8, color="#000000", opacity=1.0, outliers={"size": 0}, median=False
).encode(
    x=alt.X("meta_model:N", title=None, axis=alt.Axis(labelAngle=-35, grid=True)),
    y=alt.Y(
        "mcc:Q",
        scale=alt.Scale(domain=[0.0, 1.0]),
        axis=alt.Axis(values=[0.1, 0.3, 0.5, 0.7, 0.9], title=None, orient="right")
    ),
    # color=alt.Color("cat:N", title="Pruning")
).properties(
    width=100,
    height=100
).facet(
    row=alt.Row("model:N", title=None, header=alt.Header(title=None, labels=False)),
    column=alt.Column("cat:N", title=None, sort=["pfront", "chull", "mvo", "best", "rand", "single_best"]),
    spacing=1
)

alt.hconcat(c1, c2, spacing=0.7).save(
    "boxplot.html",
    vegalite_version="5.1.0"
)