from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path

### set params
script_dir = Path(__file__).parent
os.chdir(script_dir)

#get the vals to run
start=113
end=280.289
step=2
edml_range = list(np.arange(start, end, step))
if edml_range[-1] < end:
    edml_range.append(end)
edml_range = np.array(edml_range)

lim = False

for r in range(len(edml_range)-1):

    xlow = edml_range[r]
    xhigh = edml_range[r+1]

    #pull layer count #######

    edml_layer_count = pd.read_excel('Data Files/EDML Layer Count.xlsx', 
                                     index_col=None, sheet_name=1, skiprows=2, header=None, usecols=[0,1,2,3], 
                                     names=["Depth(m)", "Count", "Year B2K", "MCE"])
    edml_layer_count["Year B1950"] = edml_layer_count["Year B2K"]-50

    #pull the lc data #######

    lc_path1='Data Files/EDML_CFA_113-1443.499m_1mm_resolution.txt'
    lc_path2='Data Files/EDML_CFA_1443.5-2774m_1mm_resolution.txt'

    read_lc1 = pd.read_csv(lc_path1, header=None, skiprows=1, sep='\t', names=['Depth(m)', 'Na(ng/g)','NH4(ng/g)', 'Ca(ng/g)', 'Dust(particles/ml)', 'Cond(mikroS/cm)'])
    read_lc2 = pd.read_csv(lc_path2, header=None, skiprows=1, sep='\t', names=['Depth(m)', 'Na(ng/g)','NH4(ng/g)', 'Ca(ng/g)', 'Dust(particles/ml)', 'Cond(mikroS/cm)'])

    read_lc = pd.concat([read_lc1, read_lc2], ignore_index=True)

    #take compare data to plot volcanic links
    volcanic_path='EDML-GICC Errors/EDML_GICC_Compare.xlsx'
    read_volcanic = pd.read_excel(volcanic_path) #take the vals
    volcanic_wd_meters = read_volcanic["EDML (m)"]

    #create plot
    from matplotlib.ticker import ScalarFormatter

    #setting data ranges
    read_lc = read_lc[read_lc['Depth(m)'] > xlow]
    read_lc = read_lc[read_lc['Depth(m)'] < xhigh]

    # Create vertically stacked plots
    fig, ax = plt.subplots(5, 1, figsize=(24, 10), sharex=True)

    labelfontsize = 12 # Set label font size

    # Plot 1: Na (magenta)
    ax[0].plot(read_lc["Depth(m)"], read_lc["Na(ng/g)"], color='magenta', clip_on=False)
    ax[0].set_ylabel("Na(ng/g)", color='magenta', fontsize=labelfontsize)
    ax[0].tick_params(axis='y', labelcolor='magenta')

    # Plot 2: NH4(ng/g) (red)

    ax[1].step(read_lc["Depth(m)"], read_lc["NH4(ng/g)"], where='mid', color='red', clip_on=False) #allowing spillover
    ax[1].set_ylabel("NH4(ng/g)", color='red', fontsize=labelfontsize) 
    ax[1].tick_params(axis='y', labelcolor='red')
    ax[1].yaxis.set_label_position("right") #move to right
    ax[1].yaxis.tick_right()

    # Plot 3: Ca (blue)
    ax[2].step(read_lc["Depth(m)"], read_lc["Ca(ng/g)"], where='mid', color='blue', clip_on=False)
    ax[2].set_ylabel("Ca(ng/g)", color='blue', fontsize=labelfontsize) 
    ax[2].tick_params(axis='y', labelcolor='blue')

    # Plot 4: Dust (green)
    ax[3].step(read_lc["Depth(m)"], (read_lc["Dust(particles/ml)"]), where='mid', color='green', clip_on=False)
    ax[3].set_ylabel("Dust (particles/ml)", color='green', fontsize=labelfontsize) 
    ax[3].tick_params(axis='y', labelcolor='green')
    ax[3].yaxis.set_label_position("right") #move to right
    ax[3].yaxis.tick_right()

    # Plot 5: Cond(mikroS/cm) (black)
    ax[4].step(read_lc["Depth(m)"], read_lc["Cond(mikroS/cm)"], where='mid', color='black', clip_on=False)
    ax[4].set_ylabel("Cond(mikroS/cm)", color='black', fontsize=labelfontsize) 
    ax[4].tick_params(axis='y', labelcolor='black')

    if lim == True:
        ax[1].set_ylim(0,60) #setting limits for y axis
        ax[2].set_ylim(0,80) #setting limits for y axis
        ax[4].set_ylim(10,60) #setting limits for y axis
    else:
        pass

    #set solid lines for layers, changing the color depending on if they are unknown
    for axes in ax: # all axes
        age_in_bounds=[]
        for i, depth in enumerate(edml_layer_count["Depth(m)"]):
            if xlow < depth < xhigh:
                if edml_layer_count["Count"][i] == 0.5:  
                    age_in_bounds.append(edml_layer_count["Year B2K"][i])
                    axes.axvspan(depth - 0.005, depth + 0.005, alpha=0.25, hatch="///", edgecolor='black')
                elif edml_layer_count["Count"][i] == 1:  
                    age_in_bounds.append(edml_layer_count["Year B2K"][i])
                    axes.axvspan(depth - 0.005, depth + 0.005, color='gray', alpha=0.5)
                elif edml_layer_count["Count"][i] != 1 or edml_layer_count["Count"][i] != 0.5:
                    print("Error in layer count values")
                    sys.exit()

        axes.tick_params(axis='both', labelsize=12) # Set tick label size

    #add black for volcanic links
    for axes in ax: # all axes
        i=0
        for link in volcanic_wd_meters:
            if xlow < link < xhigh:
                axes.axvline(link, color='black', linestyle='--', linewidth=2)
            i += 1

    # set the tick label settings
    for axes in ax[:-1]:  # All except the bottom plot
        axes.tick_params(labelbottom=False)        # Hide x-axis tick labels
        axes.spines['bottom'].set_visible(False)# Hide x-axis spine

    for axes in ax[1:]: #all except top plot
        axes.spines['top'].set_visible(False) 

    #add triangles
    triangle_positions = edml_layer_count["Depth(m)"]
    triangle_positions = triangle_positions[(triangle_positions > xlow) & (triangle_positions < xhigh)]

    # Y position slightly above the top plot's y-limits
    y_top = ax[0].get_ylim()[1] + 0.03  # Adjust as needed
    ax[0].scatter(triangle_positions, [y_top]*len(triangle_positions), marker='v', color='grey', edgecolors='black', s=50, zorder=5, clip_on=False)

    # Add age labels above triangles using axis coordinates
    for depth, year in zip(edml_layer_count["Depth(m)"], edml_layer_count["Year B2K"]):
        if xlow < depth < xhigh:
            ax[0].text(depth, 1.05, f"{np.round((year-50),1)}",
                       rotation=0, ha='center', va='bottom', fontsize=6, color='black',
                       transform=ax[0].get_xaxis_transform())

    # Set shared X axis
    ax[-1].set_xlim(xlow, xhigh)
    ax[-1].set_xlabel("Depth (m)")

    for axes in ax: #add minor ticks
        axes.minorticks_on()
        axes.xaxis.set_minor_locator(MultipleLocator(0.1))  # Minor ticks every 0.1 on x-axis
        axes.tick_params(axis='x', which='minor', length=4, color='gray')

    plt.subplots_adjust(hspace=0)
    plt.suptitle(rf"| Brittle Ice Layer Counting | Depth: $\bf{{{xlow}}}$ to $\bf{{{xhigh}}}$ | Age: $\bf{{{(min(age_in_bounds)-50)}}}$ to $\bf{{{(max(age_in_bounds)-50)}}}$ |", fontsize=16, y=0.935)
    plt.savefig(f"Original Figs/WD_LC_{xlow}_{xhigh}.png", dpi=300, bbox_inches='tight')
    print(f'{np.round((100*r/len(edml_range)),3)}%')
    plt.close()

