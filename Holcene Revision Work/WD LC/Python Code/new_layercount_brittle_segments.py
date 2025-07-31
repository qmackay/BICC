from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
import os

### set params

#get the vals to run
os.chdir("/Users/quinnmackay/Documents/GitHub/BICC/Holcene Revision/")
start=577.172
end=853.172
step=6
brittle_range = list(np.arange(start, end, step))
if brittle_range[-1] < end:
    brittle_range.append(end)
brittle_range = np.array(brittle_range)

lim = False

for r in range(len(brittle_range)-1):

    xlow = brittle_range[r]
    xhigh = brittle_range[r+1]

    #pull layer count #######

    wd_layer_count = pd.read_csv('Updated_WD2014 Layer Count.tab', comment="#", delimiter="\t", names=["Depth ice/snow [m]", "Cal age [ka BP] (ice age)"])
    wd_layer_count["Cal age [ka BP] (ice age)"] = wd_layer_count["Cal age [ka BP] (ice age)"]*1000

    old_wd_layer_count = pd.read_csv('WD2014 Layer Count.tab', comment="#", delimiter="\t", names=["Depth ice/snow [m]", "Cal age [ka BP] (ice age)", "Cal age std e [±] (ice age uncertainty due to an...)", "Cal age std e [±] (ice age uncertainty due to CH...)", "Gas age [ka BP] (gas age)", "Age e [±] (gas age uncertainty (2 sigma))",	"Age diff [ka] (gas age-ice age difference (d...)",	"Age diff e [±] (delta age uncertainty (2 sigma))"])
    old_wd_layer_count["Cal age [ka BP] (ice age)"] = wd_layer_count["Cal age [ka BP] (ice age)"]*1000

    #pull the DEP data files for the brittle ice #######

    dep_brittle_tab=["0550RA.d50", "0550RB.d50"]
    dep_brittle_tab_paths = [f'WD Layer Counting Files/DEP files brittle ice/{file}' for file in dep_brittle_tab]
    dep_brittle_comma=["0600S.d50", "0650S.d50", "0700S.d50", "0750S.d50", "0800S.d50", "0850S.d50", "0900S.d50", "0950S.d50", "1000S.d50", "1050S.d50", "1100S.d50", "1150S.d50", "1200S.d50", "1250S.d50", "1300S.d50"]
    dep_brittle_comma_paths = [f'WD Layer Counting Files/DEP files brittle ice/{file}' for file in dep_brittle_comma]

    read_dep = []

    for path in dep_brittle_tab_paths: #appends data with comma delims
        with open(path) as f: #determine how big header
            for i, line in enumerate(f):
                if "END HEADER" in line:
                    start_line = i + 1  # Start after this line + 0 for no header
                    break
        data = pd.read_csv(path, skiprows=start_line, header=None, sep=r'\s+', names=["Depth(m)", "Conductance(uS)"])
        read_dep.append(data)

    for path in dep_brittle_comma_paths: #appends data with comma delims
        with open(path) as f: #determine how big header
            for i, line in enumerate(f):
                if "END HEADER" in line:
                    start_line = i + 2  # Start after this line + 1 for header
                    break
        data = pd.read_csv(path, skiprows=start_line, header=None, sep=',', names=["Depth(m)", "Conductance(uS)"])
        read_dep.append(data)

    read_dep = pd.concat(read_dep, ignore_index=True)

    #pull the ion data #######

    ion_path='WD Layer Counting Files/WDC06A 577-1300 m Chemistry.xlsx'

    read_ion = pd.read_excel(ion_path, header=None, skiprows=5, names=['Depth(m)', "Cl(ng/g)", "NO3(ng/g)", "SO4(ng/g)", "Na(ng/g)", "K(ng/g)", "Mg(ng/g)", "Ca(ng/g)"])

    #take compare data to plot volcanic links
    volcanic_path='/Users/quinnmackay/Documents/GitHub/BICC/Holcene Revision/WDC_GICC21_Compare.tab'
    read_volcanic = pd.read_csv(volcanic_path, delimiter="\t", comment="#") #take the vals
    volcanic_wd_meters = read_volcanic["WDC(m)"]

    #create plot
    from matplotlib.ticker import ScalarFormatter

    #setting DEP data ranges
    read_dep = read_dep[read_dep['Depth(m)'] > xlow]
    read_dep = read_dep[read_dep['Depth(m)'] < xhigh]

    #setting ion data ranges
    read_ion = read_ion[read_ion['Depth(m)'] > xlow]
    read_ion = read_ion[read_ion['Depth(m)'] < xhigh]

    # Create vertically stacked plots
    fig, ax = plt.subplots(5, 1, figsize=(24, 10), sharex=True)

    labelfontsize = 12 # Set label font size

    # Plot 1: DEP (magenta)
    ax[0].plot(read_dep["Depth(m)"], read_dep["Conductance(uS)"], color='magenta', clip_on=False)
    ax[0].set_ylabel("DEP (µS)", color='magenta', fontsize=labelfontsize)
    ax[0].tick_params(axis='y', labelcolor='magenta')

    # Plot 2: nss-SO4 (red)

    read_ion.loc[read_ion["SO4(ng/g)"] > 70, "SO4(ng/g)"] = 70 # this is just needed so other parts are still visible.

    ax[1].step(read_ion["Depth(m)"], read_ion["SO4(ng/g)"], where='mid', color='red', clip_on=False) #allowing spillover
    ax[1].set_ylabel("nss-SO₄ (ng/g)", color='red', fontsize=labelfontsize) #assuming it is nss
    ax[1].tick_params(axis='y', labelcolor='red')
    ax[1].yaxis.set_label_position("right") #move to right
    ax[1].yaxis.tick_right()

    # Plot 3: Na (blue)
    ax[2].step(read_ion["Depth(m)"], read_ion["Na(ng/g)"], where='mid', color='blue', clip_on=False)
    ax[2].set_ylabel("Na (ng/g)", color='blue', fontsize=labelfontsize) 
    ax[2].tick_params(axis='y', labelcolor='blue')

    # Plot 4: nss-SO4/Na (blue)
    ax[3].step(read_ion["Depth(m)"], (read_ion["SO4(ng/g)"]/read_ion["Na(ng/g)"]), where='mid', color='green', clip_on=False)
    ax[3].set_ylabel("Nss-SO4 / Na (ng/g)", color='green', fontsize=labelfontsize) 
    ax[3].tick_params(axis='y', labelcolor='green')
    ax[3].set_yscale("log")  # Apply log scale
    ax[3].yaxis.set_major_formatter(ScalarFormatter()) # change to normal numbers
    ax[3].set_yticks([1, 10])
    ax[3].set_yticklabels(["1", "10"]) #set as normal numbers
    ax[3].yaxis.set_label_position("right") #move to right
    ax[3].yaxis.tick_right()

    # Plot 4: NO3 (black)
    ax[4].step(read_ion["Depth(m)"], read_ion["NO3(ng/g)"], where='mid', color='black', clip_on=False)
    ax[4].set_ylabel("NO3 (ng/g)", color='black', fontsize=labelfontsize) 
    ax[4].tick_params(axis='y', labelcolor='black')

    if lim == True:
        ax[1].set_ylim(0,60) #setting limits for y axis
        ax[2].set_ylim(0,80) #setting limits for y axis
        ax[4].set_ylim(10,60) #setting limits for y axis
    else:
        pass

    #set solid lines for layers, changing the color depending on if they are unknown
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

    # set the tick label settings
    for axes in ax[:-1]:  # All except the bottom plot
        axes.tick_params(labelbottom=False)        # Hide x-axis tick labels
        axes.spines['bottom'].set_visible(False)# Hide x-axis spine

    for axes in ax[1:]: #all except top plot
        axes.spines['top'].set_visible(False) 

    #add triangles
    triangle_positions = wd_layer_count["Depth ice/snow [m]"]

    # add missing depths from old_wd_layer_count
    missing_depths = old_wd_layer_count["Depth ice/snow [m]"][~old_wd_layer_count["Depth ice/snow [m]"].isin(wd_layer_count["Depth ice/snow [m]"])]
    triangle_positions = pd.concat([triangle_positions, missing_depths], ignore_index=True)


    triangle_positions = triangle_positions[(triangle_positions > xlow) & (triangle_positions < xhigh)]
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
    plt.suptitle(rf"| Brittle Ice Layer Counting | Depth: $\bf{{{xlow}}}$ to $\bf{{{xhigh}}}$ | Age: $\bf{{{min(age_in_bounds)}}}$ to $\bf{{{max(age_in_bounds)}}}$ |", fontsize=16, y=0.91)
    plt.savefig(f"Code for creating new LC/Figs/WD_LC_{xlow}_{xhigh}.png", dpi=300, bbox_inches='tight')
    plt.close()

