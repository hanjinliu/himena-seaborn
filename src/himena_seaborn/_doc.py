from __future__ import annotations

from types import SimpleNamespace


class Doc(SimpleNamespace):
    x = "Input column to plot on x-axis"
    y = "Input column to plot on y-axis"
    hue = "Input column to group by"
    jitter = "Amount of jitter"
    orient = "Orientation of the plot"
    saturation = "Proportion of the original saturation to draw colors at"
    fill = "If checked, fill in the area (use the solid patch), otherwise only draw the line"
    dodge = "If checked, separate the plots by the hue variable"
    gap = "Distance between the dodged elements in the plot"
    linewidth = "Width of the lines in the plot elements"
    log_scale = "If checked, use a log scale"
    size = "Size of the markers"
    n_boot = "Number of bootstrap iterations to use for computing confidence intervals"
    row = "Input column to use for faceting in the vertical direction"
    col = "Input column to use for faceting in the horizontal direction"
    style = "Input column to use for styling"
