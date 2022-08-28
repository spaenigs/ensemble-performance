import altair as alt
import pandas as pd

from glob import glob
from more_itertools import chunked

paths = glob("data/temp/amp_antibp2/ensemble_mvo/*/*/gens_vs_perf_*")

res2 = []
for p in paths:
    mmodel = p.split("/")[4]
    model = p.split("/")[5]
    fold = int(p[-5:-4])
    with open(p) as f:
        res = list(chunked(f.readlines(), 6))
        for idx, l in enumerate(res):
            fitness, mcc = l[2].rstrip().split(",")
            fitness = float(fitness.replace("Best Fitness: ", ""))
            mcc = float(mcc.replace(" best metrics: {'mcc': ", "").replace("}", ""))
            res2.append([idx, fitness, mcc, fold, model, mmodel])

source = pd.DataFrame(res2, columns=["gen", "fitness", "mcc", "fold", "model", "mmodel"])

line = alt.Chart(source).mark_line(color="black").encode(
    x="gen:O",
    y="mean(fitness):Q"
)

band = alt.Chart(source).mark_errorband(extent="ci", color="black").encode(
    x=alt.X("gen:O", title="Generation"),
    y=alt.Y("fitness:Q", title="Fitness (1-MCC)")
)

line2 = alt.Chart(source).mark_line().encode(
    x="gen:O",
    y="mean(mcc):Q"
)

band2 = alt.Chart(source).mark_errorband(extent="ci").encode(
    x="gen:O",
    y=alt.Y("mcc:Q")
)

(band + line).properties(
    height=150,
    width=150
).facet(
    column=alt.Column("model:N", title="Model"),
    row=alt.Row("mmodel:N", title="Ensemble")
).save("chart.html")
