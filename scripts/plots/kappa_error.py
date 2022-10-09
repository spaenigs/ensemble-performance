import pandas as pd
import altair as alt
import numpy as np

df_res = pd.read_csv(snakemake.input[0], index_col=0)
# df_res = pd.read_csv("data/temp/amp_antibp2/kappa_error_res/plot_data.csv", index_col=0)

x_min, y_min = df_res.loc[df_res.fold == 0].x.min(), df_res.loc[df_res.fold == 0].y.min()
x_max, y_max = df_res.loc[df_res.fold == 0].x.max(), df_res.loc[df_res.fold == 0].y.max()

df_res = df_res.loc[(df_res.x >= x_min) & (df_res.x <= x_max) & (df_res.y >= y_min) & (df_res.y <= y_max)]

# adopted from https://realpython.com/python-rounding/#rounding-up
def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return np.ceil(n * multiplier) / multiplier

# adopted from https://realpython.com/python-rounding/#rounding-down
def round_down(n, decimals=0):
    multiplier = 10 ** decimals
    return np.floor(n * multiplier) / multiplier


x_min = round_down(x_min, 1)
y_min = round_down(y_min, 1)

scatter = alt.Chart().mark_point(filled=True, opacity=1.0).encode(
    x=alt.X(
        "x:Q", title="kappa",
        scale=alt.Scale(domain=[x_min, x_max])
    ),
    y=alt.Y(
        "y:Q", title="average pair-wise error", axis=alt.Axis(grid=True),
        scale=alt.Scale(domain=[y_min, y_max])
    ),
    color=alt.Color(
        "cat:N", title="Pruning",
        scale=alt.Scale(
            domain=["all", "best", "chull", "mvo", "pfront", "rand"],
            range=["gray", "#fdae61", "#2c7bb6", "yellow", "#d7191c", "#abd9e9"]),
        legend=alt.Legend(orient="bottom", offset=12)
    ),
    size=alt.condition(
        alt.datum.cat == "all",
        alt.value(50),
        alt.value(100)
    ),
).properties(
    width=300,
    height=200
)

convex_hull = alt.Chart().mark_line(
    color="#2c7bb6",
    size=1.1
).encode(
    x=alt.X("x:Q", title=None),
    y=alt.Y("y:Q", title=None),
    order="chull:N",
).transform_filter(
    alt.datum.chull != -1
)

pareto_frontier = alt.Chart().mark_line(
    strokeDash=[5, 1],
    color="#d7191c",
    size=1.1
).encode(
    x="x:Q",
    y="y:Q",
    order="pfront:N"
).transform_filter(
    alt.datum.pfront != -1
)

vals = np.array(range(51)) / 100
vals = [e for e in vals if e <= y_max]
df = pd.DataFrame({"x": [1 - (1 / (1 - i)) for i in vals], "y": vals})
df = df.loc[(df.x >= x_min) & (df.y >= y_min)]

bound_line = alt.Chart(df).mark_line(color="gray", strokeDash=[4, 4]).encode(
    x=alt.X("x:Q"),
    y="y:Q"
)

c1 = alt.layer(
    convex_hull,
    pareto_frontier,
    scatter,
    bound_line,
    data=df_res.loc[df_res.fold == 0]
).facet(
    row=alt.Column("model", title=None),
    spacing=10
)

heatmap = alt.Chart().mark_rect().encode(
    x=alt.X(
        "x:Q",
        title=None,
        bin=alt.Bin(maxbins=40),
        axis=alt.Axis(values=[-1.0, -0.5, 0.0, 0.5, 1.0], format=".1f", grid=True),
        scale=alt.Scale(domain=[x_min, x_max])
    ),
    y=alt.Y(
        "y:Q",
        title=None,
        bin=alt.Bin(maxbins=40),
        axis=alt.Axis(
            values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5], 
            format=".1f", 
            grid=True, domain=False, 
            ticks=False, labels=False
        ),
        scale=alt.Scale(domain=[y_min, y_max])
    ),
    color=alt.Color(
        "count(x):Q",
        title="Count",
        legend=alt.Legend(
            gradientLength=90,
            orient="bottom",
            offset=12
            # values=[0, 1500, 3000, 4500]
            # values=[0, np.histogram2d(x=df_res.x, y=df_res.y, bins=45)[0].max()]
        ),
        scale=alt.Scale(scheme="greys")
    ),
    tooltip="count(x):Q"
).properties(
    width=300,
    height=200
)

c2 = alt.layer(
    heatmap,
    bound_line,
    # data=df_res.loc[df_res.fold.isin([0, 1, 2, 3, 4, 5])].reset_index()
    data=df_res.reset_index()
).facet(
    row=alt.Row(
        "model:N",
        title=None,
        header=alt.Header(labels=False)
    ),
    spacing=10
)

resc = alt.hconcat(
    c1,
    c2,
    spacing=1
).resolve_scale(
    color="shared"
)

# resc.save("kappa.html")
resc.configure_header(
    labelFontSize=14
).configure_axis(
    labelFontSize=12
).configure_legend(
    gradientThickness=10,
    labelFontSize=13,
    columns=3
).save(snakemake.output[0])
