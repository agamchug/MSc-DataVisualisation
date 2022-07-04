#%%
from operator import index
from os import name
from mercantile import Bbox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.core.reshape.reshape import stack_multiple
import seaborn as sns
import geopandas as gpd
from seaborn.palettes import dark_palette
import plotly.express as px
import json
import contextily as ctx
from contextily import Place
from shapely.geometry import box, Point
import geopy
import xyzservices.providers as xyz
import pyproj
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import plotly.graph_objects as go

%cd /Users/agamchug/Downloads
# %%
data = pd.read_csv('company_data.csv')
data = data.iloc[:,0:32] # removing the insignificant variables
#%% 
data.isna().sum()
data_clean=data[data['RegAddress.Country']=='ENGLAND'] # focussing ony on England
# %% More cleaning
data_clean = data.drop(columns=['SICCode.SicText_4', 'SICCode.SicText_3', 'SICCode.SicText_2', 'DissolutionDate', 'Returns.LastMadeUpDate', 'Accounts.LastMadeUpDate', 'RegAddress.County', 'RegAddress.POBox', 'RegAddress.CareOf' ])
data_clean = data_clean[data_clean['Accounts.AccountCategory']!='DORMANT']
data_clean = data_clean[data_clean['SICCode.SicText_1']!='None Supplied']
data_clean['IncorporationDate'] = pd.to_datetime(data_clean['IncorporationDate'], errors = 'coerce')
data_clean = data_clean[data_clean['IncorporationDate'] >= '01/01/2019'] # data only from 2019
# %%
data_clean.dropna(axis=0, subset=[' RegAddress.AddressLine2', 'RegAddress.Country'], inplace=True)
data_clean.dropna(axis=0,subset=['RegAddress.PostCode', 'Accounts.AccountRefDay', 'Accounts.AccountRefMonth', 'Accounts.NextDueDate', 'Returns.NextDueDate'], inplace=True)
# %% Attaching Latitudes and Longitudes based on the postcodes
postcodes= pd.read_csv('postcodes.csv')
data_coords = data_clean[['CompanyName', 'IncorporationDate', 'RegAddress.PostCode','RegAddress.Country','SICCode.SicText_1']].reset_index()
data_coords = pd.merge(data_coords, postcodes[['Postcode', 'Latitude', 'Longitude','ITL level 2']], how = 'left', left_on = 'RegAddress.PostCode', right_on = 'Postcode')
# %%
data_coords
# %% Splitting on pre and post 20/07//2020. This middle date was chosen to reflect the actualised impacts of the pandemic much later.
coords_pre = data_coords[data_coords['IncorporationDate']<'2020-07-01'].dropna(subset=['Postcode']).reset_index()
coords_post = data_coords[data_coords['IncorporationDate']>='2020-07-01'].dropna(subset=['Postcode']).reset_index()
# %% first 3 characters of postcode
coords_pre['postcode_3'] = [i[0:3] for i in coords_pre['Postcode']]
coords_post['postcode_3'] = [i[0:3] for i in coords_post['Postcode']]
# %%
# %% dataframe to get the 9 England EER regions
eer_data = pd.read_csv('NSPL_NOV_2019_UK.csv')
eer_data = eer_data[['pcd','pcd2', 'pcds','eer']]
coords_pre = pd.merge(coords_pre, eer_data, how='left', left_on = 'Postcode', right_on = 'pcds')
coords_post = pd.merge(coords_post, eer_data, how='left', left_on = 'Postcode', right_on = 'pcds')
coords_pre.drop(columns=['level_0', 'index','pcd', 'pcd2'], inplace=True)
coords_post.drop(columns=['level_0', 'index','pcd', 'pcd2'], inplace=True)
# %% function to add regions to the 2 pre and post dataframes
def add_regions(df):

    Region = []
    for i in df['eer']:
        if i == 'E15000001':
    	    Region.append('North East')
        elif i == 'E15000002':
            Region.append('North West')
        elif i == 'E15000003':
            Region.append('Yorkshire and The Humber')
        elif i == 'E15000004':
            Region.append('East Midlands')
        elif i == 'E15000005':
            Region.append('West Midlands')
        elif i == 'E15000006':
            Region.append('Eastern')
        elif i == 'E15000007':
            Region.append('London')
        elif i == 'E15000008':
            Region.append('South East')
        elif i == 'E15000009':
            Region.append('South West')
        else:
            Region.append(np.NAN)

    df.insert(loc=0,column='Regions', value=Region)
# %%
add_regions(coords_post)
add_regions(coords_pre)
# %%
coords_pre.dropna(axis=0,subset=['Regions'],inplace=True)
coords_post.dropna(axis=0,subset=['Regions'],inplace=True)
coords_post.reset_index(inplace=True, drop=True)
coords_pre.reset_index(inplace=True, drop=True)
#%%
#Leading industries with the SIC codes 
#68 real estate 
#47 retail 
#70 consultancy
#56 food and beverage
#55 Accommodation 
#62 IT 
#43 41 Construction 
#46 Wholesale
#64 and 65 Financial services
#96 Wellness 
#82 Admin
#49 50 51 52 transport and logistics 
#86 healthcare 
#45 motor vehicles
#85 education
#81 cleaning and sanitation
#59 Television 

# %% Attaching industry names
def SIC_to_ind(df):
    SIC_2 = [df['SICCode.SicText_1'][i].split(' -')[0][0:2] for i in range(len(df))] #first 2 digits of SIC
    ind = []
    for i in SIC_2:
        if i == '68':
            ind.append('Real Estate')
        elif i == '47':
            ind.append('Retail')
        elif i == '70':
            ind.append('Consultancy')
        elif i in ['56','55']:
            ind.append('Hospitality')
        elif i == '62':
            ind.append('IT')
        elif i in ['43','41']:
            ind.append('Construction')
        elif i == '46':
            ind.append('Wholesale')
        elif i in ['64','65']:
            ind.append('Financial Services')
        elif i == '96':
            ind.append('Wellness')
        elif i == '82':
            ind.append('Admin')
        elif i in ['49', '50', '51', '52']:
            ind.append('Transport and Logistics')
        elif i == '86':
            ind.append('Healthcare')
        elif i == '45':
            ind.append('Motor')
        elif i == '85':
            ind.append('Education')
        elif i == '81':
            ind.append('Cleaning and Sanitation')
        elif i == '59':
            ind.append('Television') 
        else:
            ind.append('Not Available')
    df.insert(loc=0,column='SIC_2',value=SIC_2)
    df.insert(loc=0,column='Industry',value=ind)
# %%
SIC_to_ind(coords_post)
SIC_to_ind(coords_pre)

# %% obtaining year and month only
coords_post['Inc_Y_M'] = coords_post['IncorporationDate'].apply(lambda x: x.strftime('%Y-%m'))
coords_pre['Inc_Y_M'] = coords_pre['IncorporationDate'].apply(lambda x: x.strftime('%Y-%m'))

# %% saving changes to csv
#coords_post.to_csv('coords_post.csv')
#coords_pre.to_csv('coords_pre.csv')

# %% reading the above files if needed 
#coords_pre = pd.read_csv('coords_pre.csv')
#coords_post = pd.read_csv('coords_post.csv')
#oords_pre['IncorporationDate'] = pd.to_datetime(coords_pre['IncorporationDate'])
#coords_post['IncorporationDate'] = pd.to_datetime(coords_post['IncorporationDate'])
# %% Chart 1: Geospatial pre and post covid distribution of companies
eng = gpd.read_file('topo_eer.json', epsg=3857) #for the layout of England based on the regions
# %% converting to geopandas
points_pre = gpd.points_from_xy(coords_pre.Longitude,coords_pre.Latitude)
companies_pre = gpd.GeoDataFrame(coords_pre,geometry=points_pre,crs='EPSG:3857')
points_post = gpd.points_from_xy(coords_post.Longitude,coords_post.Latitude)
companies_post = gpd.GeoDataFrame(coords_post,geometry=points_post,crs='EPSG:3857')
# %% plotting
fig = plt.figure(figsize=(20,20),frameon=True)
fig.patch.set_facecolor('ivory')
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)
eng.plot(ax=ax1, alpha=0.5, edgecolor='black', color='palegreen')
companies_pre.plot(ax=ax1,alpha=0.1,marker='.',markersize=7,color='teal')
eng.plot(ax=ax2,alpha=0.5, edgecolor='black',color='palegreen')
companies_post.plot(ax=ax2,alpha=0.1,marker='.',markersize=7,color='teal')
ax1.set_axis_off()
ax2.set_axis_off()
#ctx.add_basemap(ax=[ax1,ax2],source=ctx.providers.CartoDB.Voyager)
#ctx.add_basemap(ax=ax2,source=ctx.providers.CartoDB.Voyager)
fig.suptitle('Businesses incorporated in England before and after July 2020: guaging the \n impact of peak Covid-19 lockdowns.',ha='center', fontsize='15',color ='black')
ax1.text(-2.5,57, 'Before', alpha=0.7, color ='#1e3d59',fontsize=20)
ax2.text(-2.5,57, 'After',alpha=0.7, color ='#1e3d59',fontsize=20)
ax1.text(-2.45, 55, 'North East', fontweight='bold')
ax1.text(-3, 53.3, 'North West',fontweight='bold')
ax1.text(-2.25, 54, 'Yorkshire & The Humber',fontweight='bold')
ax1.text(-1.8, 53, 'East Midlands',fontweight='bold')
ax1.text(-2.55, 52.2, 'West Midlands',fontweight='bold')
ax1.text(1, 52.3, 'Eastern',fontweight='bold')
ax1.text(-0.3, 51.5, 'London',fontweight='bold')
ax1.text(-0.1, 51, 'South East',fontweight='bold')
ax1.text(-3, 51, 'South West',fontweight='bold')

ax2.text(-2.45, 55, 'North East', fontweight='bold')
ax2.text(-3, 53.3, 'North West',fontweight='bold')
ax2.text(-2.25, 54, 'Yorkshire & The Humber',fontweight='bold')
ax2.text(-1.8, 53, 'East Midlands',fontweight='bold')
ax2.text(-2.55, 52.2, 'West Midlands',fontweight='bold')
ax2.text(1, 52.3, 'Eastern',fontweight='bold')
ax2.text(-0.3, 51.5, 'London',fontweight='bold')
ax2.text(-0.1, 51, 'South East',fontweight='bold')
ax2.text(-3, 51, 'South West',fontweight='bold')

# %% Plot2: Location and industry-wise changes pre and post July 2020
coords_pre['Regions'].unique()
coords_pre.groupby(['Regions'])['SIC_2'].count()
# %% choosing the 5 most dense regions
# %%
London_pre = coords_pre[coords_pre['Regions']=='London'] 
London_pre = London_pre.groupby(['Industry']).count().loc[:,['SIC_2']].sort_values(['SIC_2'],ascending=False).loc[['Retail','Real Estate', 'Construction', 'IT', 'Consultancy', 'Hospitality', 'Healthcare']].reset_index(level=['Industry'])
SE_pre = coords_pre[coords_pre['Regions']=='South East'].groupby(['Industry']).count().loc[:,['SIC_2']].sort_values(['SIC_2'],ascending=False).loc[['Retail','Real Estate', 'Construction', 'IT', 'Consultancy', 'Hospitality', 'Healthcare']].reset_index(level=['Industry'])
E_pre = coords_pre[coords_pre['Regions']=='Eastern'].groupby(['Industry']).count().loc[:,['SIC_2']].sort_values(['SIC_2'],ascending=False).loc[['Retail','Real Estate', 'Construction', 'IT', 'Consultancy', 'Hospitality', 'Healthcare']].reset_index(level=['Industry'])
NW_pre = coords_pre[coords_pre['Regions']=='North West'].groupby(['Industry']).count().loc[:,['SIC_2']].sort_values(['SIC_2'],ascending=False).loc[['Retail','Real Estate', 'Construction', 'IT', 'Consultancy', 'Hospitality', 'Healthcare']].reset_index(level=['Industry'])
WM_pre = coords_pre[coords_pre['Regions']=='West Midlands'].groupby(['Industry']).count().loc[:,['SIC_2']].sort_values(['SIC_2'],ascending=False).loc[['Retail','Real Estate', 'Construction', 'IT', 'Consultancy', 'Hospitality', 'Healthcare']].reset_index(level=['Industry'])


# %%
London_post = coords_post[coords_post['Regions']=='London'] 
London_post = London_post.groupby(['Industry']).count().loc[:,['SIC_2']].sort_values(['SIC_2'],ascending=False).loc[['Retail','Real Estate', 'Construction', 'IT', 'Consultancy', 'Hospitality', 'Healthcare']].reset_index(level=['Industry'])
SE_post = coords_post[coords_post['Regions']=='South East'].groupby(['Industry']).count().loc[:,['SIC_2']].sort_values(['SIC_2'],ascending=False).loc[['Retail','Real Estate', 'Construction', 'IT', 'Consultancy', 'Hospitality', 'Healthcare']].reset_index(level=['Industry'])
E_post = coords_post[coords_post['Regions']=='Eastern'].groupby(['Industry']).count().loc[:,['SIC_2']].sort_values(['SIC_2'],ascending=False).loc[['Retail','Real Estate', 'Construction', 'IT', 'Consultancy', 'Hospitality', 'Healthcare']].reset_index(level=['Industry'])
NW_post = coords_post[coords_post['Regions']=='North West'].groupby(['Industry']).count().loc[:,['SIC_2']].sort_values(['SIC_2'],ascending=False).loc[['Retail','Real Estate', 'Construction', 'IT', 'Consultancy', 'Hospitality', 'Healthcare']].reset_index(level=['Industry'])
WM_post = coords_post[coords_post['Regions']=='West Midlands'].groupby(['Industry']).count().loc[:,['SIC_2']].sort_values(['SIC_2'],ascending=False).loc[['Retail','Real Estate', 'Construction', 'IT', 'Consultancy', 'Hospitality', 'Healthcare']].reset_index(level=['Industry'])

# %%
coords_post['Pre_Post'] = ['After' for i in range(len(coords_post))]
coords_pre['Pre_Post'] = ['Before' for i in range(len(coords_pre))]
# %%
London_post['Pre_Post'] = ['After' for i in range(len(London_post))]
London_post['Region'] = ['London' for i in range(len(London_post))]
SE_post['Pre_Post'] = ['After' for i in range(len(SE_post))]
SE_post['Region'] = ['South East' for i in range(len(SE_post))]
E_post['Pre_Post'] = ['After' for i in range(len(E_post))]
E_post['Region'] = ['East' for i in range(len(E_post))]
WM_post['Pre_Post'] = ['After' for i in range(len(WM_post))]
WM_post['Region'] = ['West Midlands' for i in range(len(WM_post))]
NW_post['Pre_Post'] = ['After' for i in range(len(NW_post))]
NW_post['Region'] = ['North West' for i in range(len(NW_post))]
# %%
London_pre['Pre_Post'] = ['Before' for i in range(len(London_post))]
London_pre['Region'] = ['London' for i in range(len(London_post))]
SE_pre['Pre_Post'] = ['Before' for i in range(len(SE_post))]
SE_pre['Region'] = ['South East' for i in range(len(SE_post))]
E_pre['Pre_Post'] = ['Before' for i in range(len(E_post))]
E_pre['Region'] = ['East' for i in range(len(E_post))]
WM_pre['Pre_Post'] = ['Before' for i in range(len(WM_post))]
WM_pre['Region'] = ['West Midlands' for i in range(len(WM_post))]
NW_pre['Pre_Post'] = ['Before' for i in range(len(NW_post))]
NW_pre['Region'] = ['North West' for i in range(len(NW_post))]
# %% combined data frame for the bar chart
df_bar = pd.concat([London_post, London_pre, SE_post, SE_pre, E_post, E_pre, WM_post, WM_pre, NW_post, NW_pre])
df_bar.rename(columns={'SIC_2':'Count'},inplace=True)
# %% Plot2: Interactive stacked and grouped bar chart. Hover over stacks to see industry sizes.
fig = px.bar(df_bar, x="Region", y="Count", color='Pre_Post',
             barmode='group', opacity=0.6, hover_name='Industry', hover_data= {'Pre_Post':False, 'Region':False})
fig.update_layout(title_text='Region and industry wise incorporation of companies, before and after July 2020', titlefont_size=12, yaxis=dict(title='Count'), legend = dict(title=' '))
fig.show()
fig.write_html('Graph2.html')

# %% Plot3: timeline of incorporation for the 5 industries.
df_line = pd.concat([coords_pre,coords_post])
# %% subsetting the data by industries
retail = df_line[df_line['Industry']=='Retail'].groupby(['Inc_Y_M']).count().loc[:,'Industry'].reset_index(level='Inc_Y_M').rename(columns={'Industry':'Count'}).iloc[0:35,:]
hosp = df_line[df_line['Industry']=='Hospitality'].groupby(['Inc_Y_M']).count().loc[:,'Industry'].reset_index(level='Inc_Y_M').rename(columns={'Industry':'Count'}).iloc[0:35,:]
health = df_line[df_line['Industry']=='Healthcare'].groupby(['Inc_Y_M']).count().loc[:,'Industry'].reset_index(level='Inc_Y_M').rename(columns={'Industry':'Count'}).iloc[0:35,:]
real_est = df_line[df_line['Industry']=='Real Estate'].groupby(['Inc_Y_M']).count().loc[:,'Industry'].reset_index(level='Inc_Y_M').rename(columns={'Industry':'Count'}).iloc[0:35,:]
it = df_line[df_line['Industry']=='IT'].groupby(['Inc_Y_M']).count().loc[:,'Industry'].reset_index(level='Inc_Y_M').rename(columns={'Industry':'Count'}).iloc[0:35,:]
# %% plotting
fig = plt.figure(figsize=(22,12))
fig.patch.set_facecolor('lavender')
ax=fig.add_subplot(111)
ax.grid(True, axis= 'y', ls='--', color = 'lightgrey')
ax.spines['bottom'].set_color("#D9D9D9")
ax.spines['left'].set_color("#D9D9D9")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.scatter(retail['Inc_Y_M'], retail['Count'],marker='+',color='c')
ax.plot(retail['Inc_Y_M'], retail['Count'],alpha=0.7,color='c')
ax.scatter(health['Inc_Y_M'], health['Count'],marker='+',color='crimson')
ax.plot(health['Inc_Y_M'], health['Count'],alpha=0.7,color='crimson')
ax.scatter(real_est['Inc_Y_M'], real_est['Count'],marker='+',color='royalblue')
ax.plot(real_est['Inc_Y_M'], real_est['Count'],alpha=0.7,color='royalblue')
ax.scatter(hosp['Inc_Y_M'], hosp['Count'],marker='+',color='lime')
ax.plot(hosp['Inc_Y_M'], hosp['Count'],alpha=0.7,color='lime')
ax.scatter(it['Inc_Y_M'], it['Count'],marker='+',color='dimgrey')
ax.plot(it['Inc_Y_M'], it['Count'],alpha=0.7,color='dimgrey')
ax.set_facecolor('lavender')


x_labs = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
        'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
        'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']
ax.set_yticklabels(['0','500', '','1500', '', '2500', '', '3500', ''], alpha=1, fontsize=14)
ax.set_ylabel('Number of companies (in thousands)', color="#424242", fontsize=16)
ax.set_xticklabels(x_labs, fontsize=14, rotation=0, alpha=1, color ='#1e3d59')
ax.tick_params(colors ='#1e3d59', which='both')
ax.set_title('New companies incorporated in England: Before and through the phases of Covid-19.',fontsize=20, y=1.02, loc='left', color='#424242',)
ax.text('2019-01', 280, '2019',alpha=1, color ='#1e3d59', fontsize='14')
ax.text('2020-01', 280, '2020', alpha=1, color ='#1e3d59', fontsize='14')
ax.text('2021-01', 280, '2021', alpha=1, color ='#1e3d59', fontsize='14')
ax.axvline(14, alpha=0.45, ymin=0.08, ymax=0.87, color='grey', ls='--')
ax.axvline(21, alpha=0.45, ymin=0.08, ymax=0.87, color='grey', ls='--')
ax.axvline(25, alpha=0.45, ymin=0.08, ymax=0.87, color='grey', ls='--')
ax.axvline(30, alpha=0.45, ymin=0.08, ymax=0.87, color='grey', ls='--')
legendElements = [Line2D([0],[0], color='c', lw=3, label='Retail', alpha=0.8),
                Line2D([0],[0], color='crimson', lw=3,  label='Healthcare', alpha=0.8),
                Line2D([0],[0], color='royalblue', lw=3,  label='Real Estate', alpha=0.8), 
                Line2D([0],[0], color='lime', lw=3,  label='Hospitality', alpha=0.8),
                Line2D([0],[0], color='dimgrey', lw=3,  label='IT', alpha=0.8)
                ]
ax.legend(handles = legendElements, loc=(0.88,0.95), frameon=False, fontsize=16, labelcolor='#424242')
ax.text('2020-02', 2200, 'Lockdown 1', alpha=0.7, color ='#1e3d59', fontsize='10')
ax.text('2020-09', 2650, 'Lockdown 2', alpha=0.7, color ='#1e3d59', fontsize='10')
ax.text('2021-01', 2800, 'Lockdown 3', alpha=0.7, color ='#1e3d59', fontsize='10')
ax.text('2021-06', 1900, 'Lockdowns Lifted', alpha=0.7, color ='#1e3d59', fontsize='10')

plt.show()
# %%
