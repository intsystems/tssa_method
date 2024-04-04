import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl


class EnableJournalStylePlotting:
    """context manager for plotting graphs with journal style
    """
    def __init__(self):
        self._mpl_context = mpl.rc_context()

    def __enter__(self):
        self._mpl_context.__enter__()

        mpl.rcParams['figure.dpi'] = 300
        mpl.rcParams['font.family'] = 'DejaVu Serif'
        mpl.rcParams['lines.linewidth'] = 1.5
        mpl.rcParams['lines.markersize'] = 9
        mpl.rcParams['xtick.labelsize'] = 20
        mpl.rcParams['ytick.labelsize'] = 20
        mpl.rcParams['legend.fontsize'] = 20
        mpl.rcParams['axes.titlesize'] = 30
        mpl.rcParams['axes.labelsize'] = 20

    def __exit__(self, *args, **kwargs):
        self._mpl_context.__exit__(*args, **kwargs)
