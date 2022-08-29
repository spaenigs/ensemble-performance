from more_itertools import chunked

import altair as alt
import pandas as pd

res2 = []
for p in list(snakemake.input):
    mmodel = p.split("/")[4]
    model = p.split("/")[5]
    fold = int(p[-5:-4])
    with open(p) as f:
        res = list(chunked(f.readlines(),6))
        for idx, l in enumerate(res):
            fitness, mcc = l[2].rstrip().split(",")
            fitness = float(fitness.replace("Best Fitness: ",""))
            mcc = float(mcc.replace(" best metrics: {'mcc': ","").replace("}",""))
            res2.append([idx, fitness, mcc, fold, model, mmodel])

source = pd.DataFrame(res2,columns=["gen", "fitness", "mcc", "fold", "model", "mmodel"])

line = alt.Chart(source).mark_line(color="black").encode(
    x="gen:O",
    y="mean(fitness):Q"
)

band = alt.Chart(source).mark_errorband(extent="ci", color="black").encode(
    x=alt.X("gen:O",title="Generation"),
    y=alt.Y("fitness:Q",title="Fitness (1-MCC)")
)

(band + line).properties(
    height=150,
    width=150
).facet(
    column=alt.Column("model:N",title="Model"),
    row=alt.Row("mmodel:N",title="Ensemble")
).save(snakemake.output[0])