import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import re
import os

# --- Read CSV File ---
def read_emissions_csv(file_path):
    df = pd.read_csv(file_path)
    emission_start_index = df.columns.get_loc('time_interval') + 1
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
    top_5_emissions_per_bin = pd.DataFrame()

    for bin_label, emissions in binned_emissions.iterrows():
        top_5_species = emissions.sort_values(ascending=False).head(5)
        top_5_emissions_per_bin = pd.concat([top_5_emissions_per_bin, top_5_species.to_frame().T], ignore_index=True)

    altitude_labels = [f'{int(bin.right)} km' for bin in binned_emissions.index.categories]
    top_5_emissions_per_bin.index = altitude_labels

    color_list = list(mcolors.CSS4_COLORS.values())
    num_colors = len(top_5_emissions_per_bin.columns)
    colors = plt.cm.get_cmap('tab20', num_colors)

    plt.figure(figsize=(10, 8))
    for i, species in enumerate(top_5_emissions_per_bin.columns):
        plt.barh(top_5_emissions_per_bin.index, top_5_emissions_per_bin[species],
                 color=colors(i), label=species, log=False)

    plt.title('Emissions for each 5 km Interval (up to 100 km)')
    plt.xlabel('Emissions [kg]')
    plt.ylabel('Altitude')
    plt.legend(title='Species', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{save_name}_bar_chart.png")
    plt.show()

# --- Plot Pie Chart for Total Emissions ---
def plot_emissions_pie(df, emissions, save_name):
    total_emissions = df[emissions].sum()
    total_sum = total_emissions.sum()
    shares = total_emissions / total_sum

    major_species = shares[shares >= 0.02]
    other = shares[shares < 0.02].sum()
    major_species['Other'] = other

    fig, ax = plt.subplots()
    wedges, texts, autotexts = ax.pie(
        major_species,
        labels=major_species.index,
        autopct='%1.1f%%',
        pctdistance=0.8,  # Push percentages outward
        labeldistance=1.05,
        startangle=140
    )
    for autotext in autotexts:
        autotext.set_fontsize(9)
    ax.set_title('Total Emission Share per Species')
    plt.tight_layout()
    plt.savefig(f"{save_name}_pie_chart.png")
    plt.show()

# --- Save Total Emissions to Excel ---
def save_emissions_summary(df, emissions, save_name):
    total_emissions = df[emissions].sum()
    total_sum = total_emissions.sum()
    summary_df = total_emissions.to_frame(name='Total Emissions [kg]')
    summary_df.loc['TOTAL'] = total_sum
    summary_df.to_excel(f"{save_name}_emissions_summary.xlsx")

# --- Main Execution ---
def main():
    file_path = 'output_data\emissions\EMIS_2023-010_Starlink5277_Emission_Factors.csv'  # Replace with your CSV file path
    save_name = os.path.splitext(os.path.basename(file_path))[0]  # Extract filename without extension
    df, emission_cols = read_emissions_csv(file_path)
    binned_emissions = aggregate_by_altitude(df, emission_cols)
    plot_top5_emissions_per_bin(binned_emissions, save_name)
    plot_emissions_pie(df, emission_cols, save_name)
    save_emissions_summary(df, emission_cols, save_name)

if __name__ == '__main__':
    main()
