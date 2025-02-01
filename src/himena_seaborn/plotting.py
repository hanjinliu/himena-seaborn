from io import StringIO
from typing import Literal
from himena import Parametric, StandardType, WidgetDataModel
from himena.plugins import register_function, configure_gui
from himena.data_wrappers import wrap_dataframe
import numpy as np
from himena_seaborn._figure import figure_and_axes

MENUS = ["tools/dataframe/seaborn", "/model_menu/seaborn"]
TYPES = [StandardType.DATAFRAME, StandardType.TABLE]


@register_function(
    menus=MENUS,
    types=TYPES,
    command_id="himena-seaborn:plotting:stripplot",
)
def stripplot(model: WidgetDataModel) -> Parametric:
    import seaborn as sns

    columns, x_default, y_default = _get_columns_and_defaults(model)

    @configure_gui(
        x={"choices": columns, "value": x_default},
        y={"choices": columns, "value": y_default},
        hue={"choices": columns},
    )
    def run(
        x: str,
        y: str,
        hue: str | None = None,
        jitter: float = 0.1,
        dodge: bool = False,
        orient: Literal["vertical", "horizontal"] = "vertical",
        linewidth: float = 0.0,
    ) -> WidgetDataModel:
        fig, ax = figure_and_axes()
        data = _norm_data(model)
        sns.stripplot(
            x=x,
            y=y,
            hue=hue,
            data=data,
            jitter=jitter,
            dodge=dodge,
            orient=orient[0],
            linewidth=linewidth,
            ax=ax,
        )
        return WidgetDataModel(
            value=fig, type=StandardType.MPL_FIGURE, title=f"Plot of {model.title}"
        )

    return run


@register_function(
    menus=MENUS,
    types=TYPES,
    command_id="himena-seaborn:plotting:swarmplot",
)
def swarmplot(model: WidgetDataModel) -> Parametric:
    import seaborn as sns

    columns, x_default, y_default = _get_columns_and_defaults(model)

    @configure_gui(
        x={"choices": columns, "value": x_default},
        y={"choices": columns, "value": y_default},
        hue={"choices": columns},
    )
    def run(
        x: str,
        y: str,
        hue: str | None = None,
        dodge: bool = False,
        orient: Literal["vertical", "horizontal"] = "vertical",
        size: float = 5,
        linewidth: float = 0.0,
    ) -> WidgetDataModel:
        fig, ax = figure_and_axes()
        data = _norm_data(model)
        sns.swarmplot(
            x=x,
            y=y,
            hue=hue,
            data=data,
            dodge=dodge,
            size=size,
            orient=orient[0],
            linewidth=linewidth,
            warn_thresh=0,
            ax=ax,
        )
        return WidgetDataModel(
            value=fig, type=StandardType.MPL_FIGURE, title=f"Plot of {model.title}"
        )

    return run


@register_function(
    menus=MENUS,
    types=TYPES,
    command_id="himena-seaborn:plotting:boxplot",
)
def boxplot(model: WidgetDataModel) -> Parametric:
    import seaborn as sns

    columns, x_default, y_default = _get_columns_and_defaults(model)

    @configure_gui(
        x={"choices": columns, "value": x_default},
        y={"choices": columns, "value": y_default},
        hue={"choices": columns},
    )
    def run(
        x: str,
        y: str,
        hue: str | None = None,
        dodge: bool = False,
        orient: Literal["vertical", "horizontal"] = "vertical",
        saturation: float = 0.75,
        fill: bool = True,
        width: float = 0.8,
        gap: float = 0.0,
        whis: float = 1.5,
        linewidth: float = 0.0,
    ) -> WidgetDataModel:
        fig, ax = figure_and_axes()
        data = _norm_data(model)
        sns.boxplot(
            x=x,
            y=y,
            hue=hue,
            data=data,
            dodge=dodge,
            saturation=saturation,
            fill=fill,
            width=width,
            gap=gap,
            whis=whis,
            orient=orient[0],
            linewidth=linewidth,
            ax=ax,
        )
        return WidgetDataModel(
            value=fig, type=StandardType.MPL_FIGURE, title=f"Plot of {model.title}"
        )

    return run


@register_function(
    menus=MENUS,
    types=TYPES,
    command_id="himena-seaborn:plotting:violinplot",
)
def violinplot(model: WidgetDataModel) -> Parametric:
    import seaborn as sns

    columns, x_default, y_default = _get_columns_and_defaults(model)

    @configure_gui(
        x={"choices": columns, "value": x_default},
        y={"choices": columns, "value": y_default},
        hue={"choices": columns},
    )
    def run(
        x: str,
        y: str,
        hue: str | None = None,
        split: bool = False,
        saturation: float = 0.75,
        fill: bool = True,
        orient: Literal["vertical", "horizontal"] = "vertical",
        linewidth: float = 0.0,
    ) -> WidgetDataModel:
        fig, ax = figure_and_axes()
        data = _norm_data(model)
        sns.violinplot(
            x=x,
            y=y,
            hue=hue,
            data=data,
            split=split,
            saturation=saturation,
            fill=fill,
            orient=orient[0],
            linewidth=linewidth,
            ax=ax,
        )
        return WidgetDataModel(
            value=fig, type=StandardType.MPL_FIGURE, title=f"Plot of {model.title}"
        )

    return run


@register_function(
    menus=MENUS,
    types=TYPES,
    command_id="himena-seaborn:plotting:boxenplot",
)
def boxenplot(model: WidgetDataModel) -> Parametric:
    import seaborn as sns

    columns, x_default, y_default = _get_columns_and_defaults(model)

    @configure_gui(
        x={"choices": columns, "value": x_default},
        y={"choices": columns, "value": y_default},
        hue={"choices": columns},
    )
    def run(
        x: str,
        y: str,
        hue: str | None = None,
        orient: Literal["vertical", "horizontal"] = "vertical",
        saturation: float = 0.75,
        fill: bool = True,
        width: float = 0.8,
        gap: float = 0.0,
        linewidth: float = 0.0,
        width_method: Literal["exponential", "linear", "area"] = "exponential",
        k_depth: Literal["tukey", "proportion", "trustworthy", "full"] = "tukey",
        outlier_prop: float = 0.007,
        trust_alpha: float = 0.05,
        showfliers: bool = True,
    ) -> WidgetDataModel:
        fig, ax = figure_and_axes()
        data = _norm_data(model)
        sns.boxenplot(
            x=x,
            y=y,
            hue=hue,
            data=data,
            orient=orient[0],
            fill=fill,
            saturation=saturation,
            width=width,
            gap=gap,
            linewidth=linewidth,
            width_method=width_method,
            k_depth=k_depth,
            outlier_prop=outlier_prop,
            trust_alpha=trust_alpha,
            showfliers=showfliers,
            ax=ax,
        )
        return WidgetDataModel(
            value=fig, type=StandardType.MPL_FIGURE, title=f"Plot of {model.title}"
        )

    return run


@register_function(
    menus=MENUS,
    types=TYPES,
    command_id="himena-seaborn:plotting-fig:pairplot",
)
def pairplot(model: WidgetDataModel) -> Parametric:
    import seaborn as sns

    columns, _, _ = _get_columns_and_defaults(model)

    @configure_gui(
        vars={"choices": columns, "value": [columns[0][1], columns[1][1]]},
        hue={"choices": columns},
    )
    def run(
        vars: list[str],
        hue: str | None = None,
        kind: Literal["scatter", "kde", "hist", "reg"] = "scatter",
        diag_kind: Literal["auto", "hist", "kde"] | None = "auto",
    ) -> WidgetDataModel:
        data = _norm_data(model)
        fig = sns.pairplot(
            data=data,
            vars=vars,
            hue=hue,
            kind=kind,
            diag_kind=diag_kind,
        ).figure
        return WidgetDataModel(
            value=fig, type=StandardType.MPL_FIGURE, title=f"Plot of {model.title}"
        )

    return run


@register_function(
    menus=MENUS,
    types=TYPES,
    command_id="himena-seaborn:plotting-fig:jointplot",
)
def jointplot(model: WidgetDataModel) -> Parametric:
    import seaborn as sns

    columns, x_default, y_default = _get_columns_and_defaults(model)

    @configure_gui(
        x={"choices": columns, "value": x_default},
        y={"choices": columns, "value": y_default},
        hue={"choices": columns},
    )
    def run(
        x: str,
        y: str,
        hue: str | None = None,
        kind: Literal["scatter", "kde", "hist", "reg"] = "scatter",
        height: int = 6,
        ratio: int = 5,
        space: float = 0.2,
        dropna: bool = True,
        marginal_ticks: bool = True,
    ) -> WidgetDataModel:
        data = _norm_data(model)
        fig = sns.jointplot(
            x=x,
            y=y,
            data=data,
            hue=hue,
            kind=kind,
            height=height,
            ratio=ratio,
            space=space,
            dropna=dropna,
            marginal_ticks=marginal_ticks,
        ).figure
        return WidgetDataModel(
            value=fig, type=StandardType.MPL_FIGURE, title=f"Plot of {model.title}"
        )

    return run


@register_function(
    menus=MENUS,
    types=TYPES,
    command_id="himena-seaborn:plotting-fig:lmplot",
)
def lmplot(model: WidgetDataModel) -> Parametric:
    import seaborn as sns

    columns, x_default, y_default = _get_columns_and_defaults(model)

    @configure_gui(
        x={"choices": columns, "value": x_default},
        y={"choices": columns, "value": y_default},
        col={"choices": columns},
        row={"choices": columns},
        hue={"choices": columns},
    )
    def run(
        x: str,
        y: str,
        hue: str | None = None,
        col: str | None = None,
        row: str | None = None,
        height: int = 5,
        aspect: float = 1,
        markers: list[str] | None = None,
        scatter_kws: dict = {},
        line_kws: dict = {},
    ) -> WidgetDataModel:
        data = _norm_data(model)
        fig = sns.lmplot(
            x=x,
            y=y,
            data=data,
            hue=hue,
            col=col,
            row=row,
            height=height,
            aspect=aspect,
            markers=markers,
            scatter_kws=scatter_kws,
            line_kws=line_kws,
        ).figure
        return WidgetDataModel(
            value=fig, type=StandardType.MPL_FIGURE, title=f"Plot of {model.title}"
        )

    return run


def _norm_data(model: WidgetDataModel):
    if model.type == StandardType.DATAFRAME:
        return model.value
    elif model.type == StandardType.TABLE:
        import pandas as pd

        buf = StringIO()
        np.savetxt(buf, model.value, fmt="%s", delimiter=",")
        buf.seek(0)
        df = pd.read_csv(buf, sep=",")
        return df
    else:
        raise ValueError(f"Unsupported type: {model.type!r}")


def _get_columns_and_defaults(
    model: WidgetDataModel,
    x_cat: bool = True,
    y_cat: bool = False,
) -> tuple[tuple[str, str], str, str]:
    df = wrap_dataframe(_norm_data(model))
    columns = df.column_names()
    if len(columns) < 2:
        raise ValueError("DataFrame must have at least two columns to plot")
    dtypes = df.dtypes
    choices = [
        (f"{cname} ({dtype.name})", cname) for cname, dtype in zip(columns, dtypes)
    ]
    x = columns[0]
    y = columns[1]
    return choices, x, y
