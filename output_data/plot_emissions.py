import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import re
import os

farben = {
    'CO2': 'green',
    'CO': 'lightgreen',
    'H2O': 'blue',
    'N2': 'orange',
    'Al2O3': 'silver',
    'NO': 'red',
    'BC': 'grey',
    'OH': 'purple',
    'Other': 'brown',
}

# --- Read CSV File ---
def read_emissions_csv(file_path):
    df = pd.read_csv(file_path)
    emission_start_index = df.columns.get_loc('total_sum_species') + 1
    emission_columns = df.columns[emission_start_index:]
    clean_columns = [re.sub(r'\*', '', col).strip() for col in emission_columns]
    df.rename(columns=dict(zip(emission_columns, clean_columns)), inplace=True)
    return df, clean_columns

# --- Aggregate Emissions by Altitude Bins ---
def aggregate_by_altitude(df, emissions, bin_size=5, max_altitude=100):
    df = df[df['h'] <= max_altitude]  # Limit to max altitude
    df['Altitude_bin'] = pd.cut(df['h'], bins=range(0, max_altitude + bin_size, bin_size))
    binned = df.groupby('Altitude_bin')[emissions].sum()

    total_emissions = binned.sum()
    total_sum = total_emissions.sum()
    shares = total_emissions / total_sum

    major_species = shares[shares >= 0.005].index
    minor_species = shares[shares < 0.005].index

    binned_major = binned[major_species].copy()
    binned_major['Other'] = binned[minor_species].sum(axis=1)

    return binned_major

# --- Plot Top 5 Emissions Per Bin ---
def plot_top5_emissions_per_bin(binned_emissions, save_name):
    
    top5_per_bin = []
    bin_labels = []

    for bin_label, emissions in binned_emissions.iterrows():
        top5 = emissions.sort_values(ascending=False).head(5)

        # Print top 5 emissions for this bin
        bin_range = f"{int(bin_label.left)}–{int(bin_label.right)} km"
        print(f"\nTop 5 emissions for {bin_range}:")
        for species, value in top5.items():
            print(f"  {species}: {value:.4f} kg")

        top5_per_bin.append(top5)
        bin_labels.append(bin_range)

    # Reverse bins so 0–5 km is at the top
    top5_per_bin = top5_per_bin[::-1]
    bin_labels = bin_labels[::-1]

    plt.figure(figsize=(10, 10))
    y_positions = np.arange(len(bin_labels))

    bar_height = 0.8  # Full height for all bars at same bin position

    fallback_cmap = mpl.colormaps['tab20']

    color_pool = {}
    species_seen = []

    for i, (y, top5) in enumerate(zip(y_positions, top5_per_bin)):
        for species, value in list(top5.items()):
            # Use predefined color if available
            if species not in color_pool:
                color_pool[species] = farben.get(species, fallback_cmap(len(color_pool) % 20))
                species_seen.append(species)

            plt.barh(y=y, width=value, height=bar_height,
                 color=color_pool[species], label=species)

    plt.yticks(y_positions, bin_labels)
    plt.gca().invert_yaxis()
    plt.xlabel('Emissions [kg]')
    plt.ylabel('Altitude')
    plt.title('Top 5 Emissions per 5 km Bin ')

    # Unique legend
    handles, labels = plt.gca().get_legend_handles_labels()
    seen = {}
    unique_handles = []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = True
            unique_handles.append((h, l))
    unique_handles = sorted(unique_handles, key=lambda x: species_seen.index(x[1]))
    plt.legend([h for h, _ in unique_handles], [l for _, l in unique_handles],
               bbox_to_anchor=(1.05, 1), loc='upper left', title='Species')

    plt.tight_layout()
    plt.savefig(f"output_data/plots/{save_name}_overlapping_top5_bar.png")
    plt.show()



# --- Plot Pie Chart for Total Emissions ---
def plot_emissions_pie(df, emissions, save_name):
    fallback_cmap = mpl.colormaps['tab20']

    # Calculate shares
    total_emissions = df[emissions].sum()
    total_sum = total_emissions.sum()
    shares = total_emissions / total_sum

    # Separate major and minor contributors
    major_species = shares[shares >= 0.02]
    other = shares[shares < 0.02].sum()
    major_species['Other'] = other

    # Assign colors in order of the species in the pie
    colors = [
        farben.get(species, fallback_cmap(i % 20))
        for i, species in enumerate(major_species.index)
    ]

    # Plot
    fig, ax = plt.subplots()
    wedges, texts, autotexts = ax.pie(
        major_species,
        labels=major_species.index,
        autopct='%1.1f%%',
        pctdistance=0.8,
        labeldistance=1.05,
        startangle=140,
        colors=colors  # Apply custom colors here
    )

    for autotext in autotexts:
        autotext.set_fontsize(9)

    ax.set_title('Total Emission Share per Species')
    plt.tight_layout()
    plt.savefig(f"output_data/plots/{save_name}_pie_chart.png")
    plt.show()

# --- Save Total Emissions to Excel ---
def save_emissions_summary(df, emissions, save_name):
    total_emissions = df[emissions].sum()
    total_sum = total_emissions.sum()
    summary_df = total_emissions.to_frame(name='Total Emissions [kg]')
    summary_df.loc['TOTAL'] = total_sum
    summary_df.to_excel(f"output_data/plots/{save_name}_emissions_summary.xlsx")

# --- Main Execution ---
def main():
    file_path = 'output_data\emissions\EMIS_2023-013_Starlink5492_NASA_CEA_Equilibrium(2).csv'  # Replace with your CSV file path
    save_name = os.path.splitext(os.path.basename(file_path))[0]  # Extract filename without extension
    df, emission_cols = read_emissions_csv(file_path)
    binned_emissions = aggregate_by_altitude(df, emission_cols)
    plot_top5_emissions_per_bin(binned_emissions, save_name)
    plot_emissions_pie(df, emission_cols, save_name)
    save_emissions_summary(df, emission_cols, save_name)

if __name__ == '__main__':
    main()
