import seaborn as sns
from matplotlib import rcParams
from matplotlib.ticker import MaxNLocator


def set_style():
    # This sets reasonable defaults for font size for
    # a figure that will go in a paper
    sns.set_context("paper")

    # Set the font to be serif, rather than sans
    sns.set(font='serif')

    # Make the background white, and specify the
    # specific font family
    sns.set_style("white", {
        "font.family": "serif",
        "font.serif": ["Arial"]
    })

    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42

    rcParams['axes.labelsize'] = 8
    rcParams['xtick.labelsize'] = 8
    rcParams['ytick.labelsize'] = 8
    rcParams['legend.fontsize'] = 8

    sns.set_palette(sns.color_palette('colorblind'))
    sns.set_color_codes('colorblind')


def set_size(width, fraction=1):
    # Width of figure
    fig_width_pt = width * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27
    # Golden ratio to set aesthetic figure height
    golden_ratio = (5 ** .5 - 1) / 2
    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio
    fig_dim = (fig_width_in, fig_height_in)
    return fig_dim


def set_axes_style(axes, **kwargs):
    # Reduce number of ticks
    if 'numticks' in kwargs.keys():
        my_locator = MaxNLocator(kwargs['numticks'])
        axes.yaxis.set_major_locator(my_locator)
        if 'both' in kwargs.keys():
            axes.xaxis.set_major_locator(my_locator)

    axes.tick_params(axis='both', pad=5, length=0)
    # Remove axis spines

    sns.despine(right=True, top=True, bottom=False, left=False)
