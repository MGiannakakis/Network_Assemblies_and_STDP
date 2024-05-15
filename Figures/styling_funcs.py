# -------------------------------------- #
#      Victor Buendia's master file      #
#    for formatting matplotlib graphs in #
#       a cool way. GPLv3 licensed       #
# -------------------------------------- #

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

def master_format(**kwargs):
    """
    Just a function that applies aaaaall the formats below!
    """

    format_axes(**kwargs)
    format_text(**kwargs)
    format_legend(**kwargs)

# -------------------------
# Axes format and styling
# -------------------------

def format_axes(despined=True, tickpad=1.2, capstyle="round", axes_lw = 1.1, tick_maj_size=4.5, tick_min_size=3.0, min_maj_w_ratio = 0.65, **kwargs):
    if despined:
        mpl.rcParams["axes.spines.right"] = False
        mpl.rcParams["axes.spines.top"] = False

    mpl.rcParams["lines.dash_capstyle"] = capstyle 
    mpl.rcParams["lines.solid_capstyle"] = capstyle 

    mpl.rcParams["axes.linewidth"] = axes_lw
    mpl.rcParams["xtick.major.width"] = axes_lw
    mpl.rcParams["ytick.major.width"] = axes_lw
    mpl.rcParams["xtick.minor.width"] = axes_lw * min_maj_w_ratio
    mpl.rcParams["ytick.minor.width"] = axes_lw * min_maj_w_ratio

    mpl.rcParams["xtick.major.size"] = tick_maj_size
    mpl.rcParams["ytick.major.size"] = tick_maj_size
    mpl.rcParams["xtick.minor.size"] = tick_min_size
    mpl.rcParams["ytick.minor.size"] = tick_min_size

    mpl.rcParams["xtick.major.pad"] = tickpad  
    mpl.rcParams["ytick.major.pad"] = tickpad  
    mpl.rcParams["xtick.minor.pad"] = tickpad * 0.75  
    mpl.rcParams["ytick.minor.pad"] = tickpad * 0.75

def despine(axes, bottom=True, left=True):
    """
    Eliminate top and right lines from axes.
    
    Parameters:
    - axes:  Matplotlib axis or iterable of matplotlib axes
        The list of axes to be despined.
    """

    #Check if object is iterable by raising a TypeError (most reliable, asking about type is discouraged by Pythonic style guides)
    if type(axes) is np.ndarray:
        for ax in np.ravel(axes):
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(bottom)
            ax.spines['left'].set_visible(left)

    elif type(axes) is list:
        for ax in axes:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(bottom)
            ax.spines['left'].set_visible(left)
    else:
        axes.spines['right'].set_visible(False)
        axes.spines['top'].set_visible(False)
        axes.spines['bottom'].set_visible(bottom)
        axes.spines['left'].set_visible(left)



def label_axes(axes, textpos, uppercase=False, bracket=True):
    """
    Fast way to label all the diagrams using the alphabet.
    
    Parameters:
    - axs: Matplotlib axes to be labeled
    - textpos: where axes should be positioned, in axes coordinates. Can be a pair of values or a list of pairs
    - uppercase: shall the label be always uppercase (default false)
    - bracket: shall it be between brackets (default true)
    """
    
    #Set the format. Hope I wrote correctly the alphabet.
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    if uppercase:
        alphabet = alphabet.upper()

    if bracket:
        axlabel = "({0})"
    else:
        axlabel = "{0}"
        
    #If we give just a pair of values, convert it into a list for compatibility
    if type(textpos[0]) is float:
        textpos = [[textpos[0], textpos[1]] for i in range(len(axes))]
    
    #Iterate over the axes and set the things
    if type(axes) is np.ndarray:
        for i,ax in enumerate(axes.flatten()):
            ax.text(textpos[i][0], textpos[i][1], axlabel.format(alphabet[i]), color='k', transform=ax.transAxes, weight="bold")
    elif type(axes) is list:
        for i,ax in enumerate(axes):
            ax.text(textpos[i][0], textpos[i][1], axlabel.format(alphabet[i]), color='k', transform=ax.transAxes, weight="bold")
        

def aux_annotate_axes(ax, text, fontsize=18, textpos=[0.5,0.5]):
    """
    Comes from matplotlib documentation, just puts a big label in the middle of the axis with its name.
    """
    ax.text(textpos[0], textpos[1], text, transform=ax.transAxes, ha="center", va="center", fontsize=fontsize, color="darkgrey")


# -------------------------
# Measurements and sizes 
# -------------------------

#Stores some constants used for measuring
class Measures:
    fig_w_1col = 11.0 #cm
    fig_w_2col = 21.0 #cm


def to_inches(cm):
    """
    Convert cm to inches
    """
    return cm/2.54

def one_col_size(ratio=1.618, height=None):
    """
    Returns a tuple (w,h), where w is the width if a single-column graph.
    Height by default is the golden ratio of the width, but can be chosen (in cm)
    
    Parameters:
    - ratio: float
        How large width is with respect to height. This parameter is ignored if height is not None
    - height: float, cm
        Height of the figure. If different from None (default), ratio parameter is ignored.
    """
    width = to_inches(Measures.fig_w_1col)

    if height == None:
        height = width/ratio
    else:
        height = to_inches(height)
    return (width, height)


def two_col_size(ratio=1.618, height=None):
    """
    Returns a tuple (w,h), where w is the width if a double-column graph.
    Height by default is the golden ratio of the width, but can be chosen (in cm)
    
    Parameters:
    - ratio: float
        How large width is with respect to height. This parameter is ignored if height is not None
    - height: float, cm
        Height of the figure. If different from None (default), ratio parameter is ignored.
    """

    width = to_inches(Measures.fig_w_2col)
    if height == None:
        height = width/ratio
    else:
        height = to_inches(height)
    
    return (width, height)


# -------------------------
# Text styles 
# -------------------------

#Does all text formatting at once
def format_text(usetex=False, font="Helvetica", label_fs=12, tick_fs=10, legend_fs=10, pdffonttype=3, **kwargs):
    #Font and sizes
    mpl.rcParams["font.family"] = font

    mpl.rcParams["axes.labelsize"] = label_fs 
    mpl.rcParams["xtick.labelsize"] = tick_fs 
    mpl.rcParams["ytick.labelsize"] = tick_fs 
    mpl.rcParams["legend.fontsize"] = legend_fs 

    #Set TeX mode to beautiful math, even if usetex if false
    mpl.rcParams["text.usetex"] = usetex
    mpl.rcParams["mathtext.fontset"] = "cm"

    #Type 3 is smaller, type 42 is more compliant and sometimes required
    mpl.rcParams["pdf.fonttype"] = pdffonttype

#Handy functions to globally change the fontsizes when needed 
def set_label_fs(self, fontsize):
    mpl.rcParams["axes.labelsize"] = fontsize 

def set_tick_fs(self, fontsize):
    mpl.rcParams["xtick.labelsize"] = fontsize
    mpl.rcParams["ytick.labelsize"] = fontsize

def set_legend_fs(self, fontsize):
    mpl.rcParams["legend.fontsize"] = fontsize 


# -------------------------
# Legend 
# -------------------------

def format_legend(legend_fs=10, frame=False, backcolor="#eeeeee", hdlLength=1.0, hdlText=0.2, hdlHeigth=0.6, labspacing=0.07, colspacing=0.4, **kwargs):
    """
    Apply all format we need to the legend
    """

    mpl.rcParams["legend.fontsize"] = legend_fs 
    mpl.rcParams["legend.facecolor"] = backcolor
    mpl.rcParams["legend.edgecolor"] = backcolor
    mpl.rcParams["legend.frameon"] = frame
    mpl.rcParams["legend.handlelength"] = hdlLength
    mpl.rcParams["legend.handletextpad"] = hdlText
    mpl.rcParams["legend.handleheight"] = hdlHeigth
    mpl.rcParams["legend.labelspacing"] = labspacing 
    mpl.rcParams["legend.columnspacing"] = colspacing 
