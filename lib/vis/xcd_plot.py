import pandas as pd
import altair as alt

import more_itertools


class XCDChart:

    def __init__(self, ensemble_data, cd_data, width=500):
        self.width = width
        self.cd_cats = cd_data["cats"]
        self.cd_cats_data = self.init_cd_data(self.cd_cats)
        self.cd_models = cd_data["models"]
        self.cd_models_data = self.init_cd_data(self.cd_models)
        self.ensemble_data, self.x_order, self.y_order = \
            self.init_ensemble_data(ensemble_data)

    def init_ensemble_data(self, ensemble_data):

        source = ensemble_data\
            .groupby(["model", "cat", "meta_model"])["mcc"]\
            .mean().reset_index()

        source["mcc_size"] = source.mcc.apply(
            lambda m:
            "-1.0 - 0.6" if m < 0.6 else
            "0.60 - 0.65" if 0.6 <= m < 0.65 else
            "0.65 - 0.70" if 0.65 <= m < 0.70 else
            "0.70 - 0.75" if 0.7 <= m < 0.75 else
            "0.75 - 0.80" if 0.75 <= m < 0.8 else
            "0.80 - 0.85" if 0.8 <= m < 0.85 else
            "0.85 - 0.90" if 0.85 <= m < 0.9 else
            "0.9 - 1.0"
        )

        mappings_x = dict(zip(self.cd_cats["names"], range(len(self.cd_cats["names"]))))
        source["cat_num"] = source["cat"].apply(lambda v: mappings_x[v])

        source["model_names"] = \
            source[["model", "meta_model"]].apply(lambda row: "_".join(row), axis=1)

        mappings_y = dict(zip(self.cd_models["names"], range(len(self.cd_models["names"]))))
        source["model_names_num"] = source["model_names"].apply(lambda v: mappings_y[v])

        return source, mappings_x, mappings_y

    def init_cd_data(self, cd_data):

        def find_sets(vec, threshold):
            tmp_vec = sorted(vec)
            if len(vec) == 1:
                return False
            elif (tmp_vec[-1] - tmp_vec[0]) <= threshold:
                return True
            else:
                return False

        ranks = sorted(cd_data["average_ranking"])
        cd = cd_data["cd"]
        sets = [s for s in more_itertools.substrings(ranks) if find_sets(s, cd)]

        indices = list(set(
            [idx
             for idx, l1 in enumerate(sets)
             for l2 in sets
             if l1 != l2 and set(l1).issubset(l2)]
        ))

        for i in reversed(indices):
            del sets[i]

        sets = sorted(sets, key=lambda s: sorted(s)[0])

        if len(sets) == 0:
            source = pd.DataFrame({
                "x": range(len(ranks)),
                "y": [0] * len(ranks),
                "diff": ranks,
                "diff_in": False,
                "mcc_size": "[-1.0,0.6[",
                "placeholder": True
            })
            source.loc[
                (source.x == 0) |
                (source.x == source.x.unique()[-1]), "diff_in"
            ] = True
            return source

        source = pd.concat([
            pd.DataFrame({
                "x": range(len(ranks)),
                "y": i,
                "diff": ranks,
                "diff_in": False,
                "mcc_size": "[-1.0,0.6[",
                "placeholder": False
            })
            for i in range(len(sets))
        ])

        for i in range(len(sets)):
            source.loc[(source.y == i) & (source["diff"].isin(sets[i])), "diff_in"] = True

        return source

    def make_main_component(self):

        dots = alt.Chart(self.ensemble_data).mark_rect(
            filled=True, opacity=1.0, color="black", stroke="white"
        ).encode(
            x=alt.X(
                "cat_num:O", title=None,
                axis=alt.Axis(ticks=False, labels=False),
                sort=alt.SortArray(list(self.x_order.values()))
            ),
            y=alt.Y(
                "model_names_num:N", title=None,
                axis=alt.Axis(ticks=False, labels=False),
                sort=alt.SortArray(list(self.y_order.values()))
            ),
            color=alt.Color(
                "mcc_size:N", title="MCC",
                scale=alt.Scale(scheme="greys"),
                legend=alt.Legend(orient="top", offset=3)
            ),
            tooltip=["mcc:Q", "model:N", "meta_model:N", "cat:N"]
        ).properties(
            width=self.width
        )

        return dots

    def make_cd_component_dots(self, data, x, y):

        placeholder = data["placeholder"].values[0]

        if placeholder:
            opacity = alt.value(0.0)
        else:
            opacity = alt.condition(
                alt.datum.diff_in,
                alt.value(1.0),
                alt.value(0.0)
            )

        points = alt.Chart(data).mark_point(
            size=50, opacity=0.0, stroke="black", shape="diamond"
        ).encode(
            x=x,
            y=y,
            fill=alt.condition(alt.datum.diff_in, alt.value("black"), alt.value("white")),
            opacity=opacity
        )

        points.encoding.x.axis = alt.Axis(ticks=False, labels=False)
        points.encoding.x.title = None

        points.encoding.y.axis = alt.Axis(ticks=False, labels=False)
        points.encoding.y.title = None

        return points

    def make_cd_componet_rules(self, data, fields):

        placeholder = data["placeholder"].values[0]

        source = data\
            .loc[data.diff_in].groupby("y") \
            .apply(lambda df: pd.DataFrame({
                "x": [df.iloc[0, 0]],
                "x2": df.iloc[-1, 0],
                "size": [df.iloc[-1, 2] - df.iloc[0, 2]]
            })).reset_index().drop("level_1", axis=1)

        lines = source.shape[0]
        direction = "x_axis" if fields["y"] == "y:N" else "y_axis"

        base_size = 20

        if direction == "x_axis":
            width = self.width
            height = base_size if lines == 0 else base_size * lines
        else:
            height = 300
            width = base_size if lines == 0 else base_size * lines

        rules = alt.Chart(source).transform_calculate(
            size_="datum.size == 0 ? 10 : 10 + (datum.size * 10)"
        ).mark_rule(
            opacity=0.0 if placeholder else 1.0,
            color="black",
        ).encode(
            **fields,
            size=alt.Size("size_:Q", legend=None)
        ).properties(
            height=height,
            width=width
        )

        return rules

    def annotate_axis(self, fields, width=0, height=0):

        axis_chart = alt.Chart(
            self.ensemble_data,
            view=alt.ViewConfig(stroke="white")
        ).mark_point(
            color="white",
            size=0
        ).encode(
            **fields,
        ).properties(
            width=width,
            height=height
        )

        if "y" in fields:
            axis_chart.encoding.y.axis = alt.Axis(
                offset=-20,
                orient="right"
            )
        else:
            axis_chart.encoding.x.axis = alt.Axis(
                offset=-20,
                labelAngle=360
            )

        axis_chart.encoding.x.title = None
        axis_chart.encoding.y.title = None

        return axis_chart

    def save(self, path):
        alt.hconcat(
            alt.vconcat(
                self.make_main_component(),
                alt.layer(
                    self.make_cd_componet_rules(
                        data=self.cd_cats_data,
                        fields={
                            "x": alt.X("x:O", axis=alt.Axis(grid=True, domainOpacity=0.0)),
                            # "x": alt.X("x:O", axis=alt.Axis(grid=True)),
                            "x2": "x2",
                            # "y": alt.Y("y:N", axis=alt.Axis(domainOpacity=1.0)),
                            # "y": alt.Y("y:N", axis=alt.Axis(grid=True, domainOpacity=0.0)),
                            "y": "y:N"
                        }
                    ),
                    self.make_cd_component_dots(
                        data=self.cd_cats_data,
                        x=alt.X("x:O", axis=None),
                        y="y:N"
                    )
                ),
                self.annotate_axis(
                    fields={
                        "x": alt.X(
                            "cat:N",
                            sort=alt.SortArray(list(self.x_order.keys()))
                        )},
                    width=self.width
                ),
                spacing=0
            ),
            alt.layer(
                self.make_cd_componet_rules(
                    data=self.cd_models_data,
                    fields={
                        "y": alt.X("x:O", axis=alt.Axis(grid=True, domain=False)),
                        "y2": "x2",
                        "x": alt.X("y:N", axis=alt.Axis(domain=False)),
                    }
                ),
                self.make_cd_component_dots(
                    data=self.cd_models_data,
                    x=alt.X("y:N", axis=alt.Axis(domain=False)),
                    y="x:O"
                ),
            ),
            self.annotate_axis(fields={
                "y": alt.Y(
                    "model_names:N",
                    sort=alt.SortArray(list(self.y_order.keys()))
                )
            }),
            spacing=0
        ).save(path)
