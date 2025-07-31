from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
import os

#get the vals to run
os.chdir("/Users/quinnmackay/Documents/GitHub/BICC/Holcene Revision/")
start=0
end=577.172
step=6
shallow_range = list(np.arange(start, end, step))
if shallow_range[-1] < end:
    shallow_range.append(end)
shallow_range = np.array(shallow_range)

lim = False

for r in range(len(shallow_range)-1):

    xlow = shallow_range[r]
    xhigh = shallow_range[r+1]

    #pull layer count #######

    wd_layer_count = pd.read_csv('Updated_WD2014 Layer Count.tab', comment="#", delimiter="\t", names=["Depth ice/snow [m]", "Cal age [ka BP] (ice age)"])
    wd_layer_count["Cal age [ka BP] (ice age)"] = wd_layer_count["Cal age [ka BP] (ice age)"]*1000

    old_wd_layer_count = pd.read_csv('WD2014 Layer Count.tab', comment="#", delimiter="\t", names=["Depth ice/snow [m]", "Cal age [ka BP] (ice age)", "Cal age std e [±] (ice age uncertainty due to an...)", "Cal age std e [±] (ice age uncertainty due to CH...)", "Gas age [ka BP] (gas age)", "Age e [±] (gas age uncertainty (2 sigma))",	"Age diff [ka] (gas age-ice age difference (d...)",	"Age diff e [±] (delta age uncertainty (2 sigma))"])
    old_wd_layer_count["Cal age [ka BP] (ice age)"] = wd_layer_count["Cal age [ka BP] (ice age)"]*1000

    #pull the ion data #######

    early_ion_path='WD Layer Counting Files/Sigl2015_SOM4_Antarctica.xlsx'

    all_cols = cols_to_check = ["BC", "Na", "Sr", "nssS", "nssS_Na_ratio", "Br", "nh4", "nssCa"]
    early_read_ion = pd.read_excel(early_ion_path, header=None, sheet_name='1 - WDC06A_layer_count', skiprows=1, names=["Depth_m", "Depth_mweq",	"Decimal_Year_CE",	"BC",	"Na",	"Sr",	"nssS",	"nssS_Na_ratio", "Br",	"nh4",	"nssCa"])

    for col in all_cols: #remove data > -3
        early_read_ion[col]=early_read_ion[col].mask(early_read_ion[col] < -3)

    early_dep_path = '/Users/quinnmackay/Documents/GitHub/BICC/Holcene Revision/WD Layer Counting Files/DRI_0_577m_032217.txt'
    early_read_dep = pd.read_csv(early_dep_path, delimiter="\t")
    early_read_dep['Cond(uS)']=early_read_dep['Cond(uS)'].mask(early_read_dep['Cond(uS)'] < 0)

    #pull the volcanic layer data
    volcanic_path='/Users/quinnmackay/Documents/GitHub/BICC/Holcene Revision/WDC_GICC21_Compare.tab'
    read_volcanic = pd.read_csv(volcanic_path, delimiter="\t", comment="#") #take the vals
    volcanic_wd_meters = read_volcanic["WDC(m)"]

    #create plot

    #setting ion data ranges
    early_read_ion = early_read_ion[early_read_ion['Depth_m'] > xlow]
    early_read_ion = early_read_ion[early_read_ion['Depth_m'] < xhigh]

    #setting dep data ranges
    early_read_dep = early_read_dep[early_read_dep['Depth(m)'] > xlow]
    early_read_dep = early_read_dep[early_read_dep['Depth(m)'] < xhigh]

    # Create vertically stacked plots
    fig, ax = plt.subplots(9, 1, figsize=(24, 10), sharex=True)

    labelfontsize = 12 # Set label font size

    # Plot 1: Black Carbon (magenta)
    ax[0].step(early_read_ion["Depth_m"], early_read_ion["BC"], where='mid', color='magenta', clip_on=False)
    ax[0].set_ylabel("BC", color='magenta', fontsize=labelfontsize)
    ax[0].tick_params(axis='y', labelcolor='magenta')

    # Plot 2: Na (red)

    ax[1].step(early_read_ion["Depth_m"], early_read_ion["Na"], where='mid', color='red', clip_on=False) #allowing spillover
    ax[1].set_ylabel("Na", color='red', fontsize=labelfontsize)
    ax[1].tick_params(axis='y', labelcolor='red')
    ax[1].yaxis.set_label_position("right") #move to right
    ax[1].yaxis.tick_right()

    # Plot 3: Sr (blue)
    ax[2].step(early_read_ion["Depth_m"], early_read_ion["Sr"], where='mid', color='blue', clip_on=False) #allowing spillover
    ax[2].set_ylabel("Sr (ppb)", color='blue', fontsize=labelfontsize) 
    ax[2].tick_params(axis='y', labelcolor='blue')

    # Plot 4: nssS (blue)
    ax[3].step(early_read_ion["Depth_m"], early_read_ion["nssS"], where='mid', color='green', clip_on=False) #allowing spillover
    ax[3].set_ylabel("Nss-SO4", color='green', fontsize=labelfontsize) 
    ax[3].tick_params(axis='y', labelcolor='green')
    ax[3].yaxis.set_label_position("right") #move to right
    ax[3].yaxis.tick_right()

    # Plot 5: nssS / Na (black)
    ax[4].step(early_read_ion["Depth_m"], early_read_ion["nssS_Na_ratio"], where='mid', color='black', clip_on=False) #allowing spillover
    ax[4].set_ylabel("Nss-SO4 / Na", color='black', fontsize=labelfontsize) 
    ax[4].tick_params(axis='y', labelcolor='black')

    # Plot 6: Br (orange)
    ax[5].step(early_read_ion["Depth_m"], early_read_ion["Br"], where='mid', color='orange', clip_on=False) #allowing spillover
    ax[5].set_ylabel("Br", color='orange', fontsize=labelfontsize) 
    ax[5].tick_params(axis='y', labelcolor='orange')
    ax[5].yaxis.set_label_position("right") #move to right
    ax[5].yaxis.tick_right()

    # Plot 7: NH4 (purple)
    ax[6].step(early_read_ion["Depth_m"], early_read_ion["nh4"], where='mid', color='purple', clip_on=False) #allowing spillover
    ax[6].set_ylabel("NH4", color='purple', fontsize=labelfontsize) 
    ax[6].tick_params(axis='y', labelcolor='purple')

    # Plot 8: nssCa (navy)
    ax[7].step(early_read_ion["Depth_m"], early_read_ion["nssCa"], where='mid', color='navy', clip_on=False) #allowing spillover
    ax[7].set_ylabel("nssCa", color='navy', fontsize=labelfontsize) 
    ax[7].tick_params(axis='y', labelcolor='navy')
    ax[7].yaxis.set_label_position("right") #move to right
    ax[7].yaxis.tick_right()

    # Plot 9: Conductivity (salmon)
    ax[8].step(early_read_dep["Depth(m)"], early_read_dep["Cond(uS)"], color='salmon', where='mid', clip_on=False) #allowing spillover
    ax[8].set_ylabel("Cond (uS)", color='salmon', fontsize=labelfontsize) 
    ax[8].tick_params(axis='y', labelcolor='salmon')


    #set solid lines for layers
    for axes in ax: # all axes
        i=0
        age_in_bounds=[]
        for depth in wd_layer_count["Depth ice/snow [m]"]:
            if xlow < depth < xhigh:
                if not np.any(np.isclose(old_wd_layer_count["Depth ice/snow [m]"].values, depth, atol=1e-3)): #if it is not present in the old layer count, it is a new layer
                    age_in_bounds.append(wd_layer_count["Cal age [ka BP] (ice age)"][i])
                    axes.axvspan(depth - 0.005, depth + 0.005, color='green', alpha=0.5)
                elif np.any(np.isclose(old_wd_layer_count["Depth ice/snow [m]"].values, depth, atol=1e-3)):  #if it is  present in the old layer count, it is a old layer
                    age_in_bounds.append(wd_layer_count["Cal age [ka BP] (ice age)"][i])
                    axes.axvspan(depth - 0.005, depth + 0.005, color='gray', alpha=0.5)
            i += 1

        for depth in old_wd_layer_count["Depth ice/snow [m]"]: #this code should create lines for the layers that were removed
            if xlow < depth < xhigh:
                if not np.any(np.isclose(wd_layer_count["Depth ice/snow [m]"].values, depth, atol=1e-3)):
                    axes.axvspan(depth - 0.005, depth + 0.005, color='red', alpha=0.5)
        
        axes.tick_params(axis='both', labelsize=12) # Set tick label size

    #add black for volcanic links
    for axes in ax: # all axes
        i=0
        for link in volcanic_wd_meters:
            if xlow < link < xhigh:
                axes.axvline(link, color='black', linestyle='--', linewidth=2)
            i += 1

    for axes in ax[:-1]:  # All except the bottom plot
        axes.tick_params(labelbottom=False)        # Hide x-axis tick labels
        axes.spines['bottom'].set_visible(False)# Hide x-axis spine

    for axes in ax[1:]: #all except top plot
        axes.spines['top'].set_visible(False) 

    #add triangles
    triangle_positions = wd_layer_count["Depth ice/snow [m]"]
    triangle_positions = triangle_positions[(triangle_positions > xlow) & (triangle_positions < xhigh)]

    # add missing depths from old_wd_layer_count
    missing_depths = old_wd_layer_count["Depth ice/snow [m]"][~old_wd_layer_count["Depth ice/snow [m]"].isin(wd_layer_count["Depth ice/snow [m]"])]
    triangle_positions = pd.concat([triangle_positions, missing_depths], ignore_index=True)

    # Y position slightly above the top plot's y-limits
    y_top = ax[0].get_ylim()[1] + 0.03  # Adjust as needed
    ax[0].scatter(triangle_positions, [y_top]*len(triangle_positions), marker='v', color='grey', edgecolors='black', s=50, zorder=5, clip_on=False)

    # Set shared X axis
    ax[-1].set_xlim(xlow, xhigh)
    ax[-1].set_xlabel("Depth (m)")

    for axes in ax: #add minor ticks
        axes.minorticks_on()
        axes.xaxis.set_minor_locator(MultipleLocator(0.1))  # Minor ticks every 0.1 on x-axis
        axes.tick_params(axis='x', which='minor', length=4, color='gray')

    plt.subplots_adjust(hspace=0)
    plt.suptitle(rf"| Shallow Ice Layer Counting | Depth: $\bf{{{xlow}}}$ to $\bf{{{xhigh}}}$ | Age: $\bf{{{min(age_in_bounds)}}}$ to $\bf{{{max(age_in_bounds)}}}$ |", fontsize=16, y=0.91)
    plt.savefig(f"Code for creating new LC/Figs/WD_LC_{xlow}_{xhigh}.png", dpi=300, bbox_inches='tight')
    plt.close()