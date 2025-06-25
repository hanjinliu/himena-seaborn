from io import StringIO
from typing import Literal
from himena import Parametric, StandardType, WidgetDataModel
from himena.plugins import register_function, configure_gui
from himena.data_wrappers import wrap_dataframe
import numpy as np
from himena_seaborn._figure import figure_and_axes
from himena_seaborn._doc import Doc

MENUS = ["tools/dataframe/seaborn", "/model_menu/seaborn"]
TYPES = [StandardType.DATAFRAME, StandardType.TABLE]


@register_function(
    menus=MENUS,
    types=TYPES,
    title="Strip Plot",
    command_id="himena-seaborn:stripplot",
    group="plotting-categorical",
)
def stripplot(model: WidgetDataModel) -> Parametric:
    """Make a strip plot"""
    import seaborn as sns

    columns, x_default, y_default = _get_columns_and_defaults(model)

    @configure_gui(
        x={"choices": columns, "value": x_default, "tooltip": Doc.x},
        y={"choices": columns, "value": y_default, "tooltip": Doc.y},
        hue={"choices": columns, "tooltip": Doc.hue},
        jitter={"tooltip": Doc.jitter},
        dodge={"tooltip": Doc.dodge},
        orient={"tooltip": Doc.orient},
        size={"tooltip": Doc.size},
        linewidth={"tooltip": Doc.linewidth},
        log_scale={"tooltip": Doc.log_scale},
    )
    def run(
        x: str | None = None,
        y: str | None = None,
        hue: str | None = None,
        jitter: float = 0.1,
        dodge: bool = False,
        orient: Literal["vertical", "horizontal"] = "vertical",
        size: float = 5.0,
        linewidth: float = 0.0,
        log_scale: bool = False,
    ) -> WidgetDataModel:
        fig, ax = figure_and_axes()
        data = _norm_data(model)
        sns.stripplot(
            x=x, y=y, hue=hue, data=data, jitter=jitter, dodge=dodge, size=size,
            orient=_get_orient(orient), linewidth=linewidth, log_scale=log_scale, ax=ax,
        )  # fmt: skip
        return WidgetDataModel(
            value=fig, type=StandardType.MPL_FIGURE, title=f"Plot of {model.title}"
        )

    return run


@register_function(
    menus=MENUS,
    types=TYPES,
    title="Swarm Plot",
    command_id="himena-seaborn:swarmplot",
    group="plotting-categorical",
)
def swarmplot(model: WidgetDataModel) -> Parametric:
    """Make a swarm plot"""
    import seaborn as sns

    columns, x_default, y_default = _get_columns_and_defaults(model)

    @configure_gui(
        x={"choices": columns, "value": x_default, "tooltip": Doc.x},
        y={"choices": columns, "value": y_default, "tooltip": Doc.y},
        hue={"choices": columns, "tooltip": Doc.hue},
        dodge={"tooltip": Doc.dodge},
        orient={"tooltip": Doc.orient},
        size={"tooltip": Doc.size},
        linewidth={"tooltip": Doc.linewidth},
        log_scale={"tooltip": Doc.log_scale},
    )
    def run(
        x: str | None = None,
        y: str | None = None,
        hue: str | None = None,
        dodge: bool = False,
        orient: Literal["vertical", "horizontal"] = "vertical",
        size: float = 5,
        linewidth: float = 0.0,
        log_scale: bool = False,
    ) -> WidgetDataModel:
        fig, ax = figure_and_axes()
        data = _norm_data(model)
        sns.swarmplot(
            x=x, y=y, hue=hue, data=data, dodge=dodge, size=size, log_scale=log_scale,
            orient=_get_orient(orient), linewidth=linewidth, warn_thresh=1, ax=ax,
        )  # fmt: skip
        return WidgetDataModel(
            value=fig, type=StandardType.MPL_FIGURE, title=f"Plot of {model.title}"
        )

    return run


@register_function(
    menus=MENUS,
    types=TYPES,
    title="Box Plot",
    command_id="himena-seaborn:boxplot",
    group="plotting-categorical",
)
def boxplot(model: WidgetDataModel) -> Parametric:
    """Make a box plot"""
    import seaborn as sns

    columns, x_default, y_default = _get_columns_and_defaults(model)

    @configure_gui(
        x={"choices": columns, "value": x_default, "tooltip": Doc.x},
        y={"choices": columns, "value": y_default, "tooltip": Doc.y},
        hue={"choices": columns, "tooltip": Doc.hue},
        dodge={"tooltip": Doc.dodge},
        orient={"tooltip": Doc.orient},
        saturation={"tooltip": Doc.saturation},
        fill={"tooltip": Doc.fill},
        width={"tooltip": "Width of the boxes"},
        gap={"tooltip": Doc.gap},
        whis={
            "tooltip": "Proportion of the IQR past the low and high quartiles to extend the plot whiskers"
        },
        linewidth={"tooltip": Doc.linewidth},
    )
    def run(
        x: str | None = None,
        y: str | None = None,
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
            x=x, y=y, hue=hue, data=data, dodge=dodge, saturation=saturation, fill=fill,
            width=width, gap=gap, whis=whis, orient=_get_orient(orient),
            linewidth=linewidth, ax=ax,
        )  # fmt: skip
        return WidgetDataModel(
            value=fig, type=StandardType.MPL_FIGURE, title=f"Plot of {model.title}"
        )

    return run


@register_function(
    menus=MENUS,
    types=TYPES,
    title="Violin Plot",
    command_id="himena-seaborn:violinplot",
    group="plotting-categorical",
)
def violinplot(model: WidgetDataModel) -> Parametric:
    """Make a violin plot"""
    import seaborn as sns

    columns, x_default, y_default = _get_columns_and_defaults(model)

    @configure_gui(
        x={"choices": columns, "value": x_default, "tooltip": Doc.x},
        y={"choices": columns, "value": y_default, "tooltip": Doc.y},
        hue={"choices": columns, "tooltip": Doc.hue},
        split={"tooltip": "If checked, split the violins for different hue levels"},
        width={"tooltip": "Width of the violins"},
        saturation={"tooltip": Doc.saturation},
        fill={"tooltip": Doc.fill},
        orient={"tooltip": Doc.orient},
        linewidth={"tooltip": Doc.linewidth},
    )
    def run(
        x: str | None = None,
        y: str | None = None,
        hue: str | None = None,
        split: bool = False,
        width: float = 0.8,
        saturation: float = 0.75,
        fill: bool = True,
        orient: Literal["vertical", "horizontal"] = "vertical",
        linewidth: float = 0.0,
    ) -> WidgetDataModel:
        fig, ax = figure_and_axes()
        data = _norm_data(model)
        sns.violinplot(
            x=x, y=y, hue=hue, data=data, split=split, saturation=saturation, fill=fill,
            orient=_get_orient(orient), linewidth=linewidth, width=width, ax=ax,
        )  # fmt: skip
        return WidgetDataModel(
            value=fig, type=StandardType.MPL_FIGURE, title=f"Plot of {model.title}"
        )

    return run


@register_function(
    menus=MENUS,
    types=TYPES,
    title="Boxen Plot",
    command_id="himena-seaborn:boxenplot",
    group="plotting-categorical",
)
def boxenplot(model: WidgetDataModel) -> Parametric:
    """Make a boxen plot"""
    import seaborn as sns

    columns, x_default, y_default = _get_columns_and_defaults(model)

    @configure_gui(
        x={"choices": columns, "value": x_default, "tooltip": Doc.x},
        y={"choices": columns, "value": y_default, "tooltip": Doc.y},
        hue={"choices": columns, "tooltip": Doc.hue},
        orient={"tooltip": Doc.orient},
        saturation={"tooltip": Doc.saturation},
        fill={"tooltip": Doc.fill},
        width={"tooltip": "Width of the boxes"},
        gap={"tooltip": Doc.gap},
        linewidth={"tooltip": Doc.linewidth},
        width_method={"tooltip": "Method to calculate the width of the boxes"},
        k_depth={"tooltip": "Number of boxes to include in each level of the plot"},
        outlier_prop={"tooltip": "Proportion of data that is considered as outliers"},
        trust_alpha={"tooltip": "Alpha level for trustworthiness of the boxplot"},
        showfliers={"tooltip": "If checked, show the outliers"},
    )
    def run(
        x: str | None = None,
        y: str | None = None,
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
            x=x, y=y, hue=hue, data=data, orient=_get_orient(orient), fill=fill,
            saturation=saturation, width=width, gap=gap, linewidth=linewidth,
            width_method=width_method, k_depth=k_depth, outlier_prop=outlier_prop,
            trust_alpha=trust_alpha, showfliers=showfliers, ax=ax,
        )  # fmt: skip
        return WidgetDataModel(
            value=fig, type=StandardType.MPL_FIGURE, title=f"Plot of {model.title}"
        )

    return run


@register_function(
    menus=MENUS,
    types=TYPES,
    title="Pair Plot",
    command_id="himena-seaborn:pairplot",
    group="plotting-fig",
)
def pairplot(model: WidgetDataModel) -> Parametric:
    """Make a pair plot (pairwise relationships in a dataset)"""
    import seaborn as sns

    columns, _, _ = _get_columns_and_defaults(model)

    @configure_gui(
        hue={"choices": columns},
        kind={"tooltip": "Kind of plot for the diagonal subplots"},
        diag_kind={"tooltip": "Kind of plot for the diagonal subplots"},
    )
    def run(
        hue: str | None = None,
        kind: Literal["scatter", "kde", "hist", "reg"] = "scatter",
        diag_kind: Literal["auto", "hist", "kde"] | None = "auto",
    ) -> WidgetDataModel:
        data = _norm_data(model)
        fig = sns.pairplot(
            data=data, vars=vars, hue=hue, kind=kind, diag_kind=diag_kind,
        ).figure  # fmt: skip
        return WidgetDataModel(
            value=fig, type=StandardType.MPL_FIGURE, title=f"Plot of {model.title}"
        )

    return run


@register_function(
    menus=MENUS,
    types=TYPES,
    title="Joint Plot",
    command_id="himena-seaborn:jointplot",
    group="plotting-fig",
)
def jointplot(model: WidgetDataModel) -> Parametric:
    """Make a joint plot (joint and marginal views of two variables)"""
    import seaborn as sns

    columns, x_default, y_default = _get_columns_and_defaults(model)

    @configure_gui(
        x={"choices": columns, "value": x_default, "tooltip": Doc.x},
        y={"choices": columns, "value": y_default, "tooltip": Doc.y},
        hue={"choices": columns, "tooltip": Doc.hue},
        kind={"tooltip": "Kind of plot to draw in the joint plot"},
        height={"tooltip": "Size of the figure (it will be square)"},
        ratio={"tooltip": "Ratio of joint axes height to marginal axes height"},
        space={"tooltip": "Space between the joint and marginal axes"},
        dropna={"tooltip": "If checked, remove missing values"},
        marginal_ticks={"tooltip": "If checked, show the ticks on the marginal axes"},
    )
    def run(
        x: str | None = None,
        y: str | None = None,
        hue: str | None = None,
        kind: Literal["scatter", "kde", "hist", "hex", "reg", "resid"] = "scatter",
        height: int = 6,
        ratio: int = 5,
        space: float = 0.2,
        dropna: bool = True,
        marginal_ticks: bool = True,
    ) -> WidgetDataModel:
        data = _norm_data(model)
        fig = sns.jointplot(
            x=x, y=y, data=data, hue=hue, kind=kind, height=height, ratio=ratio,
            space=space, dropna=dropna, marginal_ticks=marginal_ticks,
        ).figure  # fmt: skip
        return WidgetDataModel(
            value=fig, type=StandardType.MPL_FIGURE, title=f"Plot of {model.title}"
        )

    return run


@register_function(
    menus=MENUS,
    types=TYPES,
    title="Bar Plot",
    command_id="himena-seaborn:barplot",
    group="plotting-categorical",
)
def barplot(model: WidgetDataModel) -> Parametric:
    """Make a bar plot"""
    import seaborn as sns

    columns, x_default, y_default = _get_columns_and_defaults(model)

    @configure_gui(
        x={"choices": columns, "value": x_default, "tooltip": Doc.x},
        y={"choices": columns, "value": y_default, "tooltip": Doc.y},
        hue={"choices": columns, "tooltip": Doc.hue},
        n_boot={"tooltip": Doc.n_boot},
        units={
            "tooltip": "Identifier of sampling units, which will be used to perform a multilevel bootstrap"
        },
        orient={"tooltip": Doc.orient},
        saturation={"tooltip": Doc.saturation},
        capsize={"tooltip": "Width of the caps on the error bars"},
        dodge={"tooltip": Doc.dodge},
        width={"tooltip": "Width of the bars"},
        gap={"tooltip": Doc.gap},
        log_scale={"tooltip": Doc.log_scale},
    )
    def run(
        x: str | None = None,
        y: str | None = None,
        hue: str | None = None,
        n_boot: int = 1000,
        units: str | None = None,
        orient: Literal["vertical", "horizontal"] = "vertical",
        saturation: float = 0.75,
        capsize: float = 0.0,
        dodge: bool = True,
        width: float = 0.8,
        gap: float = 0,
        log_scale: bool = False,
    ) -> WidgetDataModel:
        fig, ax = figure_and_axes()
        data = _norm_data(model)
        sns.barplot(
            x=x, y=y, hue=hue, data=data, n_boot=n_boot, units=units, width=width,
            orient=_get_orient(orient), saturation=saturation, dodge=dodge,
            capsize=capsize, gap=gap, log_scale=log_scale, ax=ax,
        )  # fmt: skip
        return WidgetDataModel(
            value=fig, type=StandardType.MPL_FIGURE, title=f"Plot of {model.title}"
        )

    return run


@register_function(
    menus=MENUS,
    types=TYPES,
    title="Point Plot",
    command_id="himena-seaborn:pointplot",
    group="plotting-categorical",
)
def pointplot(model: WidgetDataModel) -> Parametric:
    """Make a point plot"""
    import seaborn as sns

    columns, x_default, y_default = _get_columns_and_defaults(model)

    @configure_gui(
        x={"choices": columns, "value": x_default, "tooltip": Doc.x},
        y={"choices": columns, "value": y_default, "tooltip": Doc.y},
        hue={"choices": columns, "tooltip": Doc.hue},
        n_boot={"tooltip": Doc.n_boot},
        units={
            "tooltip": "Identifier of sampling units, which will be used to perform a multilevel bootstrap"
        },
        orient={"tooltip": Doc.orient},
        markers={"tooltip": "Marker style"},
        linestyles={"tooltip": "Line style"},
        dodge={"tooltip": Doc.dodge},
        estimator={
            "tooltip": "Statistical function to estimate within each categorical bin"
        },
        capsize={"tooltip": "Width of the caps on the error bars"},
    )
    def run(
        x: str | None = None,
        y: str | None = None,
        hue: str | None = None,
        n_boot: int = 1000,
        units: str | None = None,
        orient: Literal["vertical", "horizontal"] = "vertical",
        markers: str = "o",
        linestyles: str = "-",
        dodge: bool = False,
        estimator: Literal[
            "mean", "median", "count", "sum", "std", "sem", "ci", "sd"
        ] = "mean",
        capsize: float = 0.0,
    ) -> WidgetDataModel:
        fig, ax = figure_and_axes()
        data = _norm_data(model)
        sns.pointplot(
            x=x, y=y, hue=hue, data=data, n_boot=n_boot, units=units,
            orient=_get_orient(orient), markers=markers, linestyles=linestyles,
            dodge=dodge, estimator=estimator, capsize=capsize, ax=ax,
        )  # fmt: skip
        return WidgetDataModel(
            value=fig, type=StandardType.MPL_FIGURE, title=f"Plot of {model.title}"
        )

    return run


@register_function(
    menus=MENUS,
    types=TYPES,
    title="Count Plot",
    command_id="himena-seaborn:countplot",
    group="plotting-categorical",
)
def countplot(model: WidgetDataModel) -> Parametric:
    """Make a count plot"""
    import seaborn as sns

    columns, x_default, y_default = _get_columns_and_defaults(model)

    @configure_gui(
        x={"choices": columns, "value": x_default, "tooltip": Doc.x},
        y={"choices": columns, "value": y_default, "tooltip": Doc.y},
        hue={"choices": columns, "tooltip": Doc.hue},
        orient={"tooltip": Doc.orient},
        saturation={"tooltip": Doc.saturation},
        dodge={"tooltip": Doc.dodge},
        fill={"tooltip": Doc.fill},
        stat={
            "tooltip": "Statistical function to estimate within each categorical bin"
        },
        width={"tooltip": "Width of the bars"},
        gap={"tooltip": Doc.gap},
        log_scale={"tooltip": Doc.log_scale},
    )
    def run(
        x: str | None = None,
        y: str | None = None,
        hue: str | None = None,
        orient: Literal["vertical", "horizontal"] = "vertical",
        saturation: float = 0.75,
        dodge: bool = True,
        fill: bool = True,
        stat: Literal["count", "frequency", "proportion"] = "count",
        width: float = 0.8,
        gap: float = 0,
        log_scale: bool = False,
    ) -> WidgetDataModel:
        fig, ax = figure_and_axes()
        data = _norm_data(model)
        sns.countplot(
            x=x, y=y, hue=hue, data=data, orient=_get_orient(orient), fill=fill,
            saturation=saturation, dodge=dodge, stat=stat, width=width, gap=gap,
            log_scale=log_scale, ax=ax,
        )  # fmt: skip
        return WidgetDataModel(
            value=fig, type=StandardType.MPL_FIGURE, title=f"Plot of {model.title}"
        )

    return run


@register_function(
    menus=MENUS,
    types=TYPES,
    title="Categorical Plot (Figure-Level)",
    command_id="himena-seaborn:catplot",
    group="plotting-fig",
)
def catplot(model: WidgetDataModel) -> Parametric:
    """Make a figure-level categorical plot"""
    import seaborn as sns

    columns, x_default, y_default = _get_columns_and_defaults(model)

    @configure_gui(
        x={"choices": columns, "value": x_default, "tooltip": Doc.x},
        y={"choices": columns, "value": y_default, "tooltip": Doc.y},
        hue={"choices": columns, "tooltip": Doc.hue},
        col={"choices": columns, "tooltip": Doc.col},
        row={"choices": columns, "tooltip": Doc.row},
        kind={"tooltip": "Kind of plot to draw"},
        height={"tooltip": "Size of the figure (it will be square)"},
        aspect={"tooltip": "Aspect ratio of each facet"},
    )
    def run(
        x: str | None = None,
        y: str | None = None,
        hue: str | None = None,
        col: str | None = None,
        row: str | None = None,
        kind: Literal[
            "strip", "swarm", "box", "violin", "boxen", "point", "bar", "count"
        ] = "strip",
        height: float = 5.0,
        aspect: float = 1.0,
    ) -> WidgetDataModel:
        data = _norm_data(model)
        g = sns.catplot(
            x=x, y=y, hue=hue, col=col, row=row, data=data, kind=kind, height=height,
            aspect=aspect,
        )  # fmt: skip
        return WidgetDataModel(
            value=g.figure,
            type=StandardType.MPL_FIGURE,
            title=f"Plot of {model.title}",
        )

    return run


@register_function(
    menus=MENUS,
    types=TYPES,
    title="Distribution Plot (Figure-Level)",
    command_id="himena-seaborn:displot",
    group="plotting-distribution",
)
def displot(model: WidgetDataModel) -> Parametric:
    """Make a figure-level distribution plot"""
    import seaborn as sns

    columns, x_default, y_default = _get_columns_and_defaults(model)

    @configure_gui(
        x={"choices": columns, "value": x_default, "tooltip": Doc.x},
        y={"choices": columns, "value": y_default, "tooltip": Doc.y},
        hue={"choices": columns, "tooltip": Doc.hue},
        kind={"tooltip": "Kind of plot to draw"},
        binwidth={"tooltip": "Width of the bins"},
        binrange={"tooltip": "Range of the bins"},
        discrete={"tooltip": "If checked, treat the data as discrete"},
        kde={"tooltip": "If checked, plot a kernel density estimate"},
        log_scale={"tooltip": "If checked, use a log scale"},
    )
    def run(
        x: str | None = None,
        y: str | None = None,
        hue: str | None = None,
        kind: Literal["hist", "kde", "ecdf"] = "hist",
        binwidth: float | None = None,
        binrange: list[float] | None = None,
        discrete: bool = False,
        kde: bool = False,
        log_scale: bool = False,
    ) -> WidgetDataModel:
        fig, ax = figure_and_axes()
        data = _norm_data(model)
        sns.displot(
            data=data, x=x, y=y, hue=hue, kind=kind, binwidth=binwidth,
            binrange=_norm_bin_range(binrange), discrete=discrete, kde=kde,
            log_scale=log_scale, ax=ax,
        )  # fmt: skip
        return WidgetDataModel(
            value=fig.figure,
            type=StandardType.MPL_FIGURE,
            title=f"Plot of {model.title}",
        )

    return run


@register_function(
    menus=MENUS,
    types=TYPES,
    title="Histogram Plot",
    command_id="himena-seaborn:histplot",
    group="plotting-distribution",
)
def histplot(model: WidgetDataModel) -> Parametric:
    """Make a histogram plot"""
    import seaborn as sns

    columns, x_default, y_default = _get_columns_and_defaults(model)

    @configure_gui(
        x={"choices": columns, "value": x_default, "tooltip": Doc.x},
        y={"choices": columns, "value": y_default, "tooltip": Doc.y},
        hue={"choices": columns, "tooltip": Doc.hue},
        binwidth={"tooltip": "Width of the bins"},
        binrange={"tooltip": "Range of the bins"},
        discrete={"tooltip": "If checked, treat the data as discrete"},
        kde={"tooltip": "If checked, plot a kernel density estimate"},
        log_scale={"tooltip": "If checked, use a log scale"},
    )
    def run(
        x: str | None = None,
        y: str | None = None,
        hue: str | None = None,
        binwidth: float | None = None,
        binrange: list[float] | None = None,
        discrete: bool = False,
        kde: bool = False,
        log_scale: bool = False,
    ) -> WidgetDataModel:
        fig, ax = figure_and_axes()
        data = _norm_data(model)
        sns.histplot(
            data=data, x=x, y=y, hue=hue, binwidth=binwidth,
            binrange=_norm_bin_range(binrange), discrete=discrete, kde=kde,
            log_scale=log_scale, ax=ax,
        )  # fmt: skip
        return WidgetDataModel(
            value=fig, type=StandardType.MPL_FIGURE, title=f"Plot of {model.title}"
        )

    return run


@register_function(
    menus=MENUS,
    types=TYPES,
    title="KDE Plot",
    command_id="himena-seaborn:kdeplot",
    group="plotting-distribution",
)
def kdeplot(model: WidgetDataModel) -> Parametric:
    """Make a KDE plot"""
    import seaborn as sns

    columns, x_default, y_default = _get_columns_and_defaults(model)

    @configure_gui(
        x={"choices": columns, "value": x_default, "tooltip": Doc.x},
        y={"choices": columns, "value": y_default, "tooltip": Doc.y},
        hue={"choices": columns, "tooltip": Doc.hue},
        fill={"tooltip": Doc.fill},
        bw_adjust={"tooltip": "Bandwidth adjustment factor"},
        log_scale={"tooltip": "If checked, use a log scale"},
        cumulative={"tooltip": "If checked, plot a cumulative distribution"},
        common_norm={"tooltip": "If checked, normalize the density estimate"},
        common_grid={"tooltip": "If checked, use the same grid for all levels"},
        levels={"tooltip": "Number of contour levels"},
        thresh={"tooltip": "Threshold for removing small levels"},
    )
    def run(
        x: str | None = None,
        y: str | None = None,
        hue: str | None = None,
        fill: bool = False,
        bw_adjust: float = 1,
        log_scale: bool = False,
        cumulative: bool = False,
        common_norm: bool = True,
        common_grid: bool = False,
        levels: int = 10,
        thresh: float = 0.05,
    ) -> WidgetDataModel:
        fig, ax = figure_and_axes()
        data = _norm_data(model)
        sns.kdeplot(
            x=x, y=y, hue=hue, data=data, fill=fill, bw_adjust=bw_adjust,
            log_scale=log_scale, cumulative=cumulative, common_norm=common_norm,
            common_grid=common_grid, levels=levels, thresh=thresh, ax=ax,
        )  # fmt: skip
        return WidgetDataModel(
            value=fig, type=StandardType.MPL_FIGURE, title=f"Plot of {model.title}"
        )

    return run


@register_function(
    menus=MENUS,
    types=TYPES,
    title="Empirical CDF Plot",
    command_id="himena-seaborn:ecdfplot",
    group="plotting-distribution",
)
def ecdfplot(model: WidgetDataModel) -> Parametric:
    """Make an empirical cumulative distribution function plot"""
    import seaborn as sns

    columns, x_default, y_default = _get_columns_and_defaults(model)

    @configure_gui(
        x={"choices": columns, "value": x_default, "tooltip": Doc.x},
        y={"choices": columns, "value": y_default, "tooltip": Doc.y},
        hue={"choices": columns, "tooltip": Doc.hue},
        stat={"tooltip": "Statistic to compute within each bin"},
        complementary={"tooltip": "If checked, plot the complementary CDF"},
        log_scale={"tooltip": Doc.log_scale},
    )
    def run(
        x: str | None = None,
        y: str | None = None,
        hue: str | None = None,
        stat: Literal["proportion", "percent", "count"] = "proportion",
        complementary: bool = False,
        log_scale: bool = False,
    ) -> WidgetDataModel:
        fig, ax = figure_and_axes()
        data = _norm_data(model)
        sns.ecdfplot(
            data=data, x=x, y=y, hue=hue, stat=stat, complementary=complementary,
            log_scale=log_scale, ax=ax,
        )  # fmt: skip
        return WidgetDataModel(
            value=fig, type=StandardType.MPL_FIGURE, title=f"Plot of {model.title}"
        )

    return run


@register_function(
    menus=MENUS,
    types=TYPES,
    title="Rug Plot",
    command_id="himena-seaborn:rugplot",
    group="plotting-distribution",
)
def rugplot(model: WidgetDataModel) -> Parametric:
    """Make a rug plot"""
    import seaborn as sns

    columns, x_default, y_default = _get_columns_and_defaults(model)

    @configure_gui(
        x={"choices": columns, "value": x_default, "tooltip": Doc.x},
        y={"choices": columns, "value": y_default, "tooltip": Doc.y},
        hue={"choices": columns, "tooltip": Doc.hue},
        height={"tooltip": "Height of the ticks"},
        expand_margins={"tooltip": "If checked, expand the plot margins"},
        linewidth={"tooltip": Doc.linewidth},
        alpha={"tooltip": "Transparency of the ticks"},
    )
    def run(
        x: str | None = None,
        y: str | None = None,
        hue: str | None = None,
        height: float = 0.025,
        expand_margins: bool = True,
        linewidth: float = 1,
        alpha: float = 1,
    ) -> WidgetDataModel:
        fig, ax = figure_and_axes()
        data = _norm_data(model)
        sns.rugplot(
            x=x, y=y, hue=hue, data=data, height=height, expand_margins=expand_margins,
            linewidth=linewidth, alpha=alpha, ax=ax,
        )  # fmt: skip
        return WidgetDataModel(
            value=fig, type=StandardType.MPL_FIGURE, title=f"Plot of {model.title}"
        )

    return run


@register_function(
    menus=MENUS,
    types=TYPES,
    title="Regression Plot (Figure-Level)",
    command_id="himena-seaborn:lmplot",
    group="plotting-regression",
)
def lmplot(model: WidgetDataModel) -> Parametric:
    """Make a figure-level regression plot"""
    import seaborn as sns

    columns, x_default, y_default = _get_columns_and_defaults(model)

    @configure_gui(
        x={"choices": columns, "value": x_default, "tooltip": Doc.x},
        y={"choices": columns, "value": y_default, "tooltip": Doc.y},
        col={"choices": columns, "tooltip": Doc.col},
        row={"choices": columns, "tooltip": Doc.row},
        hue={"choices": columns, "tooltip": Doc.hue},
        height={"tooltip": "Size of the figure (it will be square)"},
        aspect={"tooltip": "Aspect ratio of each facet"},
    )
    def run(
        x: str | None = None,
        y: str | None = None,
        hue: str | None = None,
        col: str | None = None,
        row: str | None = None,
        height: int = 5,
        aspect: float = 1,
    ) -> WidgetDataModel:
        data = _norm_data(model)
        fig = sns.lmplot(
            x=x, y=y, data=data, hue=hue, col=col, row=row, height=height,
            aspect=aspect,
        ).figure  # fmt: skip
        return WidgetDataModel(
            value=fig, type=StandardType.MPL_FIGURE, title=f"Plot of {model.title}"
        )

    return run


@register_function(
    menus=MENUS,
    types=TYPES,
    title="Regression Plot",
    command_id="himena-seaborn:regplot",
    group="plotting-regression",
)
def regplot(model: WidgetDataModel) -> Parametric:
    """Make a regression plot"""
    import seaborn as sns

    columns, x_default, y_default = _get_columns_and_defaults(model)

    @configure_gui(
        x={"choices": columns, "value": x_default, "tooltip": Doc.x},
        y={"choices": columns, "value": y_default, "tooltip": Doc.y},
        scatter={"tooltip": "If checked, show the scatter plot"},
        fit_reg={"tooltip": "If checked, show the regression line"},
        n_boot={"tooltip": Doc.n_boot},
        units={
            "tooltip": "Identifier of sampling units, which will be used to perform a multilevel bootstrap"
        },
        order={"tooltip": "Order of the polynomial to fit"},
        logistic={"tooltip": "If checked, fit a logistic regression model"},
        lowess={"tooltip": "If checked, fit a lowess smoother"},
        robust={"tooltip": "If checked, fit a robust regression model"},
        logx={"tooltip": "If checked, use a log scale for the x-axis"},
    )
    def run(
        x: str | None = None,
        y: str | None = None,
        scatter: bool = True,
        fit_reg: bool = True,
        n_boot: int = 1000,
        units: str | None = None,
        order: int = 1,
        logistic: bool = False,
        lowess: bool = False,
        robust: bool = False,
        logx: bool = False,
    ) -> WidgetDataModel:
        fig, ax = figure_and_axes()
        sns.regplot(
            data=_norm_data(model), x=x, y=y, scatter=scatter, fit_reg=fit_reg,
            n_boot=n_boot, units=units, order=order, logistic=logistic,
            lowess=lowess, robust=robust, logx=logx, ax=ax,
        )  # fmt: skip
        return WidgetDataModel(
            value=fig, type=StandardType.MPL_FIGURE, title=f"Plot of {model.title}"
        )

    return run


@register_function(
    menus=MENUS,
    types=TYPES,
    title="Scatter Plot",
    command_id="himena-seaborn:scatterplot",
    group="plotting-rel",
)
def scatterplot(model: WidgetDataModel) -> Parametric:
    """Make a scatter plot"""
    import seaborn as sns

    columns, x_default, y_default = _get_columns_and_defaults(model)

    @configure_gui(
        x={"choices": columns, "value": x_default, "tooltip": Doc.x},
        y={"choices": columns, "value": y_default, "tooltip": Doc.y},
        hue={"choices": columns, "tooltip": Doc.hue},
        style={"choices": columns, "tooltip": Doc.style},
        size={"choices": columns, "tooltip": Doc.size},
    )
    def run(
        x: str | None = None,
        y: str | None = None,
        hue: str | None = None,
        size: str | None = None,
        style: str | None = None,
    ) -> WidgetDataModel:
        fig, ax = figure_and_axes()
        sns.scatterplot(
            data=_norm_data(model), x=x, y=y, hue=hue, style=style, size=size,
            ax=ax,
        )  # fmt: skip
        return WidgetDataModel(
            value=fig, type=StandardType.MPL_FIGURE, title=f"Plot of {model.title}"
        )

    return run


@register_function(
    menus=MENUS,
    types=TYPES,
    title="Line Plot",
    command_id="himena-seaborn:lineplot",
    group="plotting-rel",
)
def lineplot(model: WidgetDataModel) -> Parametric:
    """Make a line plot"""
    import seaborn as sns

    columns, x_default, y_default = _get_columns_and_defaults(model)

    @configure_gui(
        x={"choices": columns, "value": x_default, "tooltip": Doc.x},
        y={"choices": columns, "value": y_default, "tooltip": Doc.y},
        hue={"choices": columns, "tooltip": Doc.hue},
        style={"choices": columns, "tooltip": Doc.style},
        size={"choices": columns, "tooltip": Doc.size},
        units={"choices": columns, "tooltip": "Identifier of sampling units"},
        estimator={"tooltip": "Statistical function to estimate within each bin"},
        n_boot={"tooltip": Doc.n_boot},
        sort={"tooltip": "If checked, sort the x-axis"},
        err_style={"tooltip": "Style of error bars"},
    )
    def run(
        x: str | None = None,
        y: str | None = None,
        hue: str | None = None,
        style: str | None = None,
        size: str | None = None,
        units: str | None = None,
        estimator: Literal[
            "mean", "median", "count", "sum", "std", "sem", "ci", "sd"
        ] = "mean",
        n_boot: int = 1000,
        sort: bool = True,
        err_style: Literal["band", "bars"] = "band",
    ) -> WidgetDataModel:
        fig, ax = figure_and_axes()
        sns.lineplot(
            data=_norm_data(model), x=x, y=y, hue=hue, style=style, size=size,
            units=units, estimator=estimator, n_boot=n_boot, sort=sort,
            err_style=err_style, ax=ax,
        )  # fmt: skip
        return WidgetDataModel(
            value=fig, type=StandardType.MPL_FIGURE, title=f"Plot of {model.title}"
        )

    return run


@register_function(
    menus=MENUS,
    types=TYPES,
    title="Relational Plot (Figure-Level)",
    command_id="himena-seaborn:relplot",
    group="plotting-fig",
)
def relplot(model: WidgetDataModel) -> Parametric:
    """Make a figure-level relational plot"""
    import seaborn as sns

    columns, x_default, y_default = _get_columns_and_defaults(model)

    @configure_gui(
        x={"choices": columns, "value": x_default, "tooltip": Doc.x},
        y={"choices": columns, "value": y_default, "tooltip": Doc.y},
        hue={"choices": columns, "tooltip": Doc.hue},
        col={"choices": columns, "tooltip": Doc.col},
        row={"choices": columns, "tooltip": Doc.row},
        style={"choices": columns, "tooltip": Doc.style},
        size={"choices": columns, "tooltip": Doc.size},
    )
    def run(
        x: str | None = None,
        y: str | None = None,
        hue: str | None = None,
        col: str | None = None,
        row: str | None = None,
        style: str | None = None,
        size: str | None = None,
        kind: Literal["scatter", "line"] = "scatter",
        height: float = 5.0,
        aspect: float = 1.0,
    ) -> WidgetDataModel:
        g = sns.relplot(
            x=x, y=y, hue=hue, col=col, row=row, style=style, size=size,
            data=_norm_data(model), kind=kind, height=height, aspect=aspect,
        )  # fmt: skip
        return WidgetDataModel(
            value=g.figure,
            type=StandardType.MPL_FIGURE,
            title=f"Plot of {model.title}",
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


def _norm_bin_range(binrange: list[float] | None) -> tuple[float, float] | None:
    if binrange is None:
        return None
    if len(binrange) != 2:
        raise ValueError("binrange must have two elements.")
    return binrange


def _get_columns_and_defaults(
    model: WidgetDataModel,
) -> tuple[tuple[str, str], str | None, str]:
    df = wrap_dataframe(_norm_data(model))
    columns = df.column_names()
    if len(columns) == 0:
        raise ValueError("DataFrame is empty.")

    dtypes = df.dtypes
    choices = [
        (f"{cname} ({dtype.name})", cname) for cname, dtype in zip(columns, dtypes)
    ]
    if len(columns) == 1:
        x = None
        y = columns[0]
    else:
        x = columns[0]
        y = columns[1]
    return choices, x, y


def _get_orient(orient: str) -> str:
    if orient == "vertical":
        return "v"
    elif orient == "horizontal":
        return "h"
    else:
        raise ValueError(f"Invalid orient value: {orient!r}")
