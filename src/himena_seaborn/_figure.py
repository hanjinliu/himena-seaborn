from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes


def figure_and_axes() -> tuple[Figure, Axes]:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    return fig, ax
