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


def DAGTIL4():
    # Importing the data from DAGTIL4
    Rec_vars = Dst.get_variables(table_id='DAGTIL4')
    Rec_vars

    # Make the Dataframe
    variables_rec = {'OMRÅDE':['*'],'TILSKUDSART':['*'],'BERORT':['*'], 'TID':['*']}
    rec_gen= Dst.get_data(table_id = 'DAGTIL4', variables=variables_rec)
    rec_gen.sort_values(by=['TID', 'OMRÅDE'], inplace=True)

    # On municipality and year level, sum the number of recipients of each subsidy type
    rec_gen = rec_gen.groupby(['OMRÅDE', 'TID', 'BERORT','TILSKUDSART']).sum().reset_index()

    # Generate 1 new column for OMRÅDE AND TID with INDHOLD for BERORT = 'Children' and TILSKUDSART = 'Subsidy for day-care of own children'
    rec_gen['own_child'] = rec_gen[(rec_gen['BERORT'] == 'Children') & (rec_gen['TILSKUDSART'] == 'Subsidy for day-care of own children')].groupby(['OMRÅDE', 'TID'])['INDHOLD'].transform('sum')
    # Generate 1 new column for OMRÅDE AND TID with INDHOLD for BERORT = 'Children' and TILSKUDSART = 'Subsidy to parents who choose private day-care'
    rec_gen['pri_child'] = rec_gen[(rec_gen['BERORT'] == 'Children') & (rec_gen['TILSKUDSART'] == 'Subsidy to parents who choose private day-care')].groupby(['OMRÅDE', 'TID'])['INDHOLD'].transform('sum')

    # Generate 1 new column for OMRÅDE AND TID with INDHOLD for BERORT = 'Families' and TILSKUDSART = 'Subsidy to parents who choose private day-care'
    rec_gen['pri_fam'] = rec_gen[(rec_gen['BERORT'] == 'Families') & (rec_gen['TILSKUDSART'] == 'Subsidy to parents who choose private day-care')].groupby(['OMRÅDE', 'TID'])['INDHOLD'].transform('sum')
    # Generate 1 new column for OMRÅDE AND TID with INDHOLD for BERORT = 'Families' and TILSKUDSART = 'Subsidy to parents who choose private day-care'
    rec_gen['own_fam'] = rec_gen[(rec_gen['BERORT'] == 'Families') & (rec_gen['TILSKUDSART'] == 'Subsidy for day-care of own children')].groupby(['OMRÅDE', 'TID'])['INDHOLD'].transform('sum')

    # Replace NaN with 0
    rec_gen = rec_gen.fillna(0)
    # Replace .. with 0
    rec_gen = rec_gen.replace('..', 0)

    # drop the INDHOLD, BERORT and TILSKUDSART columns
    rec_gen = rec_gen.drop(columns=['INDHOLD','BERORT', 'TILSKUDSART'])

    # Convert the own_child, pri_child, pri_fam and own_fam columns to integers
    rec_gen['own_child'] = rec_gen['own_child'].astype(int)
    rec_gen['pri_child'] = rec_gen['pri_child'].astype(int)
    rec_gen['pri_fam'] = rec_gen['pri_fam'].astype(int)
    rec_gen['own_fam'] = rec_gen['own_fam'].astype(int)

    # For each OMRÅDE AND TID, sum the 'own_child', 'pri_child', 'pri_fam' and 'own_fam' columns
    rec_gen = rec_gen.groupby(['OMRÅDE', 'TID']).sum().reset_index()

    # Rename the columns
    rec_gen = rec_gen.rename(columns={'OMRÅDE':'Municipality', 'TID':'Year'})

    # return the dataframe 
    return rec_gen 



def BY2():
    # Importing the data from BY2
    By_vars = Dst.get_variables(table_id='BY2')
    By_vars

    # Make the Dataframe
    variables_by = {'KOMK':['*'],'BYST':['*'],'ALDER':['*'], 'Tid':['*']}
    by_gen= Dst.get_data(table_id = 'BY2', variables=variables_by)
    by_gen.sort_values(by=['TID', 'KOMK'], inplace=True)

    # Rename the columns
    by_gen = by_gen.rename(columns={'KOMK':'Municipality','ALDER':'Age', 'TID':'Year'})

    # Remove non-numeric characters from the "Age" column (the observations in 'Age' are of the type "51 years" and not "51") 
    by_gen['Age'] = by_gen['Age'].str.replace(' years?', '', regex=True).astype(int)

    # Filter the dataset to select rows where Age is between 0 and 6 years
    age_0_to_6 = by_gen[(by_gen['Age'] >= 0) & (by_gen['Age'] <= 6)]

    # Group the data by Municipality and Year and sum the number of persons
    grouped_data = age_0_to_6.groupby(['Municipality', 'Year'])['INDHOLD'].sum()

    # Convert the "Age" column to string type
    by_gen['Age'] = by_gen['Age'].astype(str)

    # Remove non-numeric characters from the "Age" column
    by_gen['Age'] = by_gen['Age'].str.replace(' years?', '', regex=True).astype(int)

    # Filter the dataset to select rows where Age is between 0 and 6 years
    age_0_to_6 = by_gen[(by_gen['Age'] >= 0) & (by_gen['Age'] <= 6)]

    # Group the data by Municipality and Year and sum the number of persons
    grouped_data = age_0_to_6.groupby(['Municipality', 'Year'])['INDHOLD'].sum().reset_index()
    grouped_data = grouped_data.rename(columns={'INDHOLD': 'Residents aged 0-6 years'})

    # Add a new column for the total number of residents
    total_residents = by_gen.groupby(['Municipality', 'Year'])['INDHOLD'].sum().reset_index()
    total_residents = total_residents.rename(columns={'INDHOLD': 'Total number of residents'})

    # Merge the two dataframes
    data2 = pd.merge(grouped_data, total_residents, on=['Municipality', 'Year'])

    # return the dataframe 
    return data2