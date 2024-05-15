import numpy as np
import cmasher as cmr


colors = {}

#Base colors for lines
colors["tuning"] = "#9B4D89" 
colors["diversity"] = "#B3CB65"
colors["EI_IN"] = "#D4AB6A" 
colors["EI_OUT"] = "#4B688B" 

#Shades for the base colors!
colors["tn0"] = "#9B4D8A"
colors["tn1"] = "#B878AA"
colors["tn2"] = "#D6ABCC"

colors["dv0"] = "#89A336"
colors["dv1"] = "#B3CB65"
colors["dv2"] = "#DDF19E"

colors["in0"] = "#AA7E39"
colors["in1"] = "#D4AB6A"
colors["in2"] = "#FCDAA5"

colors["ot0"] = "#4B688B"
colors["ot1"] = "#7189A5"
colors["ot2"] = "#A5B6CA"

#Excitation and inhibition
colors["EXC"] = "#4962C8"
colors["INH"] = "#FF4747"

#Colormap
colors["cmap"] = cmr.ember