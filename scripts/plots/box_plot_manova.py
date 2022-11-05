import pandas as pd
import altair as alt
import numpy as np

df_res = pd.read_csv(snakemake.input[0], index_col=0)

df_out = pd.DataFrame()
for m in df_res.model.unique():
    df_tmp = df_res.loc[df_res.model == m]
    df_tmp = df_tmp.loc[np.bitwise_not(
        df_tmp.ensemble_mvo |
        df_tmp.ensemble_best |
        df_tmp.ensemble_rand |
        df_tmp.ensemble_chull |
        df_tmp.ensemble_pfront
    ) & (df_tmp.chull_complete == -1)]
    df_tmp = pd.concat([
        pd.DataFrame({"variable": df_tmp.x, "type": "kappa", "model": m}),
        pd.DataFrame({"variable": df_tmp.y, "type": "error", "model": m})
    ])
    df_out = pd.concat([df_out, df_tmp])

chart = alt.Chart(df_out).mark_boxplot(
    color="grey",
    size=15
).encode(
    x=alt.X("type:N", title=None, axis=None),
    y=alt.Y("variable:Q", title=None),
    color=alt.Color(
        "type:N", title="Type",
        scale=alt.Scale(scheme="greys")
    ),
    column=alt.Column("model:N", title="Model", spacing=2)
).properties(
    width=50,
    height=100
)

chart.save(snakemake.output[0])  # html

