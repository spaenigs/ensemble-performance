from xcd_plot import XCDChart

import pandas as pd
import altair as alt

import yaml

# with open("data/ensembles_res/cd.yaml") as f:
with open(snakemake.input[0]) as f:
    cd_data = yaml.safe_load(f)

from glob import glob

df_res = pd.DataFrame()
# for p in glob("data/ensembles_res/*/*/*.csv"):
for p in list(snakemake.input)[1:]:
    df_tmp = pd.read_csv(p, index_col=0)
    df_res = pd.concat([df_res, df_tmp])

df_res = df_res.loc[df_res.cat != "mvo"]

from pprint import pprint

s = sorted(zip(cd_data["models"]["average_ranking"], cd_data["models"]["names"]), key=lambda tup: tup[1])

# cd_data["models"]["average_ranking"] = [e[0] for e in s]
# cd_data["models"]["names"] = [e[1] for e in s]

xcd_chart = XCDChart(ensemble_data=df_res, cd_data=cd_data)
xcd_chart.save(snakemake.output[0])