import seaborn as sns
from himena.widgets import MainWindow

def test_categorical(himena_ui: MainWindow):
    df = sns.load_dataset("tips")
    win = himena_ui.add_object(df, type="dataframe")
    for func in [
        "stripplot",
        "swarmplot",
        "violinplot",
        "boxplot",
        "barplot",
        "boxenplot",
        "pointplot",
    ]:
        nwin = len(himena_ui.tabs[0])
        himena_ui.exec_action(
            f"himena-seaborn:{func}",
            model_context=win.to_model(),
            with_params={"x": "day", "y": "total_bill", "hue": "smoker"},
        )
        assert len(himena_ui.tabs[0]) == nwin + 1

def test_relational(himena_ui: MainWindow):
    df = sns.load_dataset("tips")
    win = himena_ui.add_object(df, type="dataframe")
    for func in ["scatterplot", "lineplot"]:
        nwin = len(himena_ui.tabs[0])
        himena_ui.exec_action(
            f"himena-seaborn:{func}",
            model_context=win.to_model(),
            with_params={"x": "total_bill", "y": "tip", "hue": "day"},
        )
        assert len(himena_ui.tabs[0]) == nwin + 1

def test_distribution(himena_ui: MainWindow):
    df = sns.load_dataset("tips")
    win = himena_ui.add_object(df, type="dataframe")
    for func in ["histplot", "kdeplot", "ecdfplot"]:
        nwin = len(himena_ui.tabs[0])
        himena_ui.exec_action(
            f"himena-seaborn:{func}",
            model_context=win.to_model(),
            with_params={"x": "total_bill", "hue": "day"},
        )
        assert len(himena_ui.tabs[0]) == nwin + 1

def test_regression(himena_ui: MainWindow):
    df = sns.load_dataset("tips")
    win = himena_ui.add_object(df, type="dataframe")
    for func in ["regplot", "lmplot"]:
        nwin = len(himena_ui.tabs[0])
        himena_ui.exec_action(
            f"himena-seaborn:{func}",
            model_context=win.to_model(),
            with_params={"x": "total_bill", "y": "tip"},
        )
        assert len(himena_ui.tabs[0]) == nwin + 1
