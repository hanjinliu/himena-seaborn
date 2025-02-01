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
    title="Strip plot",
    command_id="himena-seaborn:plotting-categorical:stripplot",
)
def stripplot(model: WidgetDataModel) -> Parametric:
    """Make a strip plot"""
    import seaborn as sns

    columns, x_default, y_default = _get_columns_and_defaults(model)

    @configure_gui(
        x={"choices": columns, "value": x_default},
        y={"choices": columns, "value": y_default},
        hue={"choices": columns},
    )
    def run(
        x: str | None = None,
        y: str | None = None,
        hue: str | None = None,
        jitter: float = 0.1,
        dodge: bool = False,
        orient: Literal["vertical", "horizontal"] = "vertical",
        linewidth: float = 0.0,
    ) -> WidgetDataModel:
        fig, ax = figure_and_axes()
        data = _norm_data(model)
        sns.stripplot(
            x=x, y=y, hue=hue, data=data, jitter=jitter, dodge=dodge,
            orient=_get_orient(orient), linewidth=linewidth, ax=ax,
        )  # fmt: skip
        return WidgetDataModel(
            value=fig, type=StandardType.MPL_FIGURE, title=f"Plot of {model.title}"
        )

    return run


@register_function(
    menus=MENUS,
    types=TYPES,
    title="Swarm plot",
    command_id="himena-seaborn:plotting-categorical:swarmplot",
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
        x: str | None = None,
        y: str | None = None,
        hue: str | None = None,
        dodge: bool = False,
        orient: Literal["vertical", "horizontal"] = "vertical",
        size: float = 5,
        linewidth: float = 0.0,
    ) -> WidgetDataModel:
        fig, ax = figure_and_axes()
        data = _norm_data(model)
        sns.swarmplot(
            x=x, y=y, hue=hue, data=data, dodge=dodge, size=size,
            orient=_get_orient(orient), linewidth=linewidth, warn_thresh=1, ax=ax,
        )  # fmt: skip
        return WidgetDataModel(
            value=fig, type=StandardType.MPL_FIGURE, title=f"Plot of {model.title}"
        )

    return run


@register_function(
    menus=MENUS,
    types=TYPES,
    title="Box plot",
    command_id="himena-seaborn:plotting-categorical:boxplot",
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
    title="Violin plot",
    command_id="himena-seaborn:plotting-categorical:violinplot",
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
        x: str | None = None,
        y: str | None = None,
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
            x=x, y=y, hue=hue, data=data, split=split, saturation=saturation, fill=fill,
            orient=_get_orient(orient), linewidth=linewidth, ax=ax,
        )  # fmt: skip
        return WidgetDataModel(
            value=fig, type=StandardType.MPL_FIGURE, title=f"Plot of {model.title}"
        )

    return run


@register_function(
    menus=MENUS,
    types=TYPES,
    title="Boxen plot",
    command_id="himena-seaborn:plotting-categorical:boxenplot",
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
    title="Pair plot",
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
            data=data, vars=vars, hue=hue, kind=kind, diag_kind=diag_kind,
        ).figure  # fmt: skip
        return WidgetDataModel(
            value=fig, type=StandardType.MPL_FIGURE, title=f"Plot of {model.title}"
        )

    return run


@register_function(
    menus=MENUS,
    types=TYPES,
    title="Joint plot",
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
        x: str | None = None,
        y: str | None = None,
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
    title="Bar plot",
    command_id="himena-seaborn:plotting-categorical:barplot",
)
def barplot(model: WidgetDataModel) -> Parametric:
    import seaborn as sns

    columns, x_default, y_default = _get_columns_and_defaults(model)

    @configure_gui(
        x={"choices": columns, "value": x_default},
        y={"choices": columns, "value": y_default},
        hue={"choices": columns},
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
    ) -> WidgetDataModel:
        fig, ax = figure_and_axes()
        data = _norm_data(model)
        sns.barplot(
            x=x, y=y, hue=hue, data=data, n_boot=n_boot, units=units,
            orient=_get_orient(orient), saturation=saturation, dodge=dodge,
            capsize=capsize, ax=ax,
        )  # fmt: skip
        return WidgetDataModel(
            value=fig, type=StandardType.MPL_FIGURE, title=f"Plot of {model.title}"
        )

    return run


@register_function(
    menus=MENUS,
    types=TYPES,
    title="Point plot",
    command_id="himena-seaborn:plotting-categorical:pointplot",
)
def pointplot(model: WidgetDataModel) -> Parametric:
    import seaborn as sns

    columns, x_default, y_default = _get_columns_and_defaults(model)

    @configure_gui(
        x={"choices": columns, "value": x_default},
        y={"choices": columns, "value": y_default},
        hue={"choices": columns},
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
    title="Count plot",
    command_id="himena-seaborn:plotting-categorical:countplot",
)
def countplot(model: WidgetDataModel) -> Parametric:
    import seaborn as sns

    columns, x_default, y_default = _get_columns_and_defaults(model)

    @configure_gui(
        x={"choices": columns, "value": x_default},
        y={"choices": columns, "value": y_default},
        hue={"choices": columns},
    )
    def run(
        x: str | None = None,
        y: str | None = None,
        hue: str | None = None,
        orient: Literal["vertical", "horizontal"] = "vertical",
        saturation: float = 0.75,
        dodge: bool = True,
    ) -> WidgetDataModel:
        fig, ax = figure_and_axes()
        data = _norm_data(model)
        sns.countplot(
            x=x, y=y, hue=hue, data=data, orient=_get_orient(orient),
            saturation=saturation, dodge=dodge, ax=ax,
        )  # fmt: skip
        return WidgetDataModel(
            value=fig, type=StandardType.MPL_FIGURE, title=f"Plot of {model.title}"
        )

    return run


@register_function(
    menus=MENUS,
    types=TYPES,
    title="Categorical plot (figure-level)",
    command_id="himena-seaborn:plotting-fig:catplot",
)
def catplot(model: WidgetDataModel) -> Parametric:
    import seaborn as sns

    columns, x_default, y_default = _get_columns_and_defaults(model)

    @configure_gui(
        x={"choices": columns, "value": x_default},
        y={"choices": columns, "value": y_default},
        hue={"choices": columns},
        col={"choices": columns},
        row={"choices": columns},
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
    title="Distribution plot (figure-level)",
    command_id="himena-seaborn:plotting-distribution:displot",
)
def displot(model: WidgetDataModel) -> Parametric:
    import seaborn as sns

    columns, x_default, y_default = _get_columns_and_defaults(model)

    @configure_gui(
        x={"choices": columns, "value": x_default},
        y={"choices": columns, "value": y_default},
        hue={"choices": columns},
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
    title="Histogram plot",
    command_id="himena-seaborn:plotting-distribution:histplot",
)
def histplot(model: WidgetDataModel) -> Parametric:
    import seaborn as sns

    columns, x_default, y_default = _get_columns_and_defaults(model)

    @configure_gui(
        x={"choices": columns, "value": x_default},
        y={"choices": columns, "value": y_default},
        hue={"choices": columns},
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
    title="KDE plot",
    command_id="himena-seaborn:plotting-distribution:kdeplot",
)
def kdeplot(model: WidgetDataModel) -> Parametric:
    import seaborn as sns

    columns, x_default, y_default = _get_columns_and_defaults(model)

    @configure_gui(
        x={"choices": columns, "value": x_default},
        y={"choices": columns, "value": y_default},
        hue={"choices": columns},
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
    title="Empirical CDF plot",
    command_id="himena-seaborn:plotting-distribution:ecdfplot",
)
def ecdfplot(model: WidgetDataModel) -> Parametric:
    import seaborn as sns

    columns, x_default, y_default = _get_columns_and_defaults(model)

    @configure_gui(
        x={"choices": columns, "value": x_default},
        y={"choices": columns, "value": y_default},
        hue={"choices": columns},
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
    title="Rug plot",
    command_id="himena-seaborn:plotting-distribution:rugplot",
)
def rugplot(model: WidgetDataModel) -> Parametric:
    import seaborn as sns

    columns, x_default, y_default = _get_columns_and_defaults(model)

    @configure_gui(
        x={"choices": columns, "value": x_default},
        y={"choices": columns, "value": y_default},
        hue={"choices": columns},
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
    title="Regression plot (figure-level)",
    command_id="himena-seaborn:plotting-regression:lmplot",
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
    title="Regression plot",
    command_id="himena-seaborn:plotting-regression:regplot",
)
def regplot(model: WidgetDataModel) -> Parametric:
    import seaborn as sns

    columns, x_default, y_default = _get_columns_and_defaults(model)

    @configure_gui(
        x={"choices": columns, "value": x_default},
        y={"choices": columns, "value": y_default},
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
    title="scatter plot",
    command_id="himena-seaborn:plotting-rel:scatterplot",
)
def scatterplot(model: WidgetDataModel) -> Parametric:
    import seaborn as sns

    columns, x_default, y_default = _get_columns_and_defaults(model)

    @configure_gui(
        x={"choices": columns, "value": x_default},
        y={"choices": columns, "value": y_default},
        hue={"choices": columns},
        style={"choices": columns},
        size={"choices": columns},
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
    title="Line plot",
    command_id="himena-seaborn:plotting-rel:lineplot",
)
def lineplot(model: WidgetDataModel) -> Parametric:
    import seaborn as sns

    columns, x_default, y_default = _get_columns_and_defaults(model)

    @configure_gui(
        x={"choices": columns, "value": x_default},
        y={"choices": columns, "value": y_default},
        hue={"choices": columns},
        style={"choices": columns},
        size={"choices": columns},
    )
    def run(
        x: str | None = None,
        y: str | None = None,
        hue: str | None = None,
        style: str | None = None,
        size: str | None = None,
        units: str | None = None,
        estimator: Literal["mean", "median", "count", "sum", "std", "sem", "ci", "sd"]
        | None = "mean",
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
    title="Relational plot (figure-level)",
    command_id="himena-seaborn:plotting-fig:relplot",
)
def relplot(model: WidgetDataModel) -> Parametric:
    import seaborn as sns

    columns, x_default, y_default = _get_columns_and_defaults(model)

    @configure_gui(
        x={"choices": columns, "value": x_default},
        y={"choices": columns, "value": y_default},
        hue={"choices": columns},
        col={"choices": columns},
        row={"choices": columns},
        style={"choices": columns},
        size={"choices": columns},
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
