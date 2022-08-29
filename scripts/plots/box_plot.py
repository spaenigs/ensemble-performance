import pandas as pd
import altair as alt

from glob import glob

df_res = pd.DataFrame()
# for p in glob("data/ensembles_res/*/*/*.csv"):
for p in list(snakemake.input.ensemble_res):
    df_tmp = pd.read_csv(p, index_col=0)
    df_res = pd.concat([df_res, df_tmp])

df_res_single = pd.DataFrame()
for p in list(snakemake.input.single_res):
    df_tmp = pd.read_csv(p, index_col=0)
    df_res = pd.concat([df_res, df_tmp])

c1 = alt.Chart(df_res).mark_boxplot(
    size=8, color="#000000", opacity=1.0, outliers={"size": 0}, median=False
).encode(
    x=alt.X("meta_model:N", title=None, axis=alt.Axis(labelAngle=-35, grid=True)),
    y=alt.Y(
        "mcc:Q",
        scale=alt.Scale(domain=[0.0, 1.0]),
        axis=alt.Axis(values=[0.1, 0.3, 0.5, 0.7, 0.9], title=None)
    ),
).properties(
    width=100,
    height=100
).facet(
    row=alt.Row("model:N", title=None),
    column=alt.Column("cat:N", title=None, sort=["pfront", "chull", "mvo", "best", "rand"]),
    spacing=1
)

c2 = alt.Chart(df_res_single).mark_boxplot(
    size=8, color="#000000", opacity=1.0, outliers={"size": 0}, median=False
).encode(
    x=alt.X("rank:N", title=None, axis=alt.Axis(labelAngle=-35, grid=True)),
    y=alt.Y(
        "mcc:Q",
        scale=alt.Scale(domain=[0.0, 1.0]),
        axis=alt.Axis(values=[0.1, 0.3, 0.5, 0.7, 0.9], title=None, orient="right")
    ),
).properties(
    width=100,
    height=100
).facet(
    row=alt.Row("model:N", title=None, header=alt.Header(title=None, labels=False)),
    column=alt.Column("cat:N", title=None, sort=["pfront", "chull", "mvo", "best", "rand", "single_best"]),
    spacing=1
)

alt.hconcat(c1, c2, spacing=0.7).save(
    # "boxplot.html",
    snakemake.output[0],
    vegalite_version="5.1.0"
)