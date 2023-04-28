import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from matplotlib_venn import venn2

from ipywidgets import interact

# user written modules
import dataproject as dp # import dataproject.py as dp
import datetime

import pandas_datareader # install with `pip install pandas-datareader`
import pydst # install with `pip install git+https://github.com/elben10/pydst`

# Setup data loader with the langauge 'english'
Dst = pydst.Dst(lang='en') 

# Get all tables in subject '1' (people)
tables = Dst.get_tables(subjects=['1']) 


# Define function to update plot based on selected municipality
def update_plot(municipality, df):
    # Filter data to selected municipality
    subset = df[df['Municipality'] == municipality]
    
    # Group data by year and sum columns for selected municipality
    grouped = subset.groupby('Year').sum()
    
    # Calculate proportion of residents aged 0-6 years receiving either type of subsidy to total residents aged 0-6 years
    grouped['proportion_private'] = grouped['pri_child'] / grouped['Residents aged 0-6 years']
    grouped['proportion_home'] = grouped['own_child'] / grouped['Residents aged 0-6 years']
    
    # Create line plot of proportion over time for selected municipality
    fig, ax = plt.subplots(figsize=(6, 4))
    grouped[['proportion_private', 'proportion_home']].plot(kind='line', ax=ax, linewidth=2)
    
    # Set plot title and axis labels
    ax.set_title(f"Proportion of residents aged 0-6 years who receive subsidies for private or at-home daycare in {municipality}", fontsize=14)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Proportion', fontsize=12)
    
    # Customize tick labels and grid lines
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Customize legend
    ax.legend(['Private daycare', 'At-home daycare'], fontsize=14)
    
    plt.show()