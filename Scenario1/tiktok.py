tiktok.py
#%%
from matplotlib import colors
from numpy.core.shape_base import hstack
import pandas as pd
import json
import datetime
import numpy as np
from pandas._libs.tslibs.period import Period
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pandas.core import frame
#%%
#  Open JSON file
file = open('/Users/agamchug/Desktop/Bayes/trending.json', encoding="utf8")

# Parse JSON
data = json.load(file)

# Close file
file.close()

# Show amount of objects
len(data['collector'])
# %%
print(json.dumps(data['collector'][4], indent=4, sort_keys=True))
# %%
# This will split the objects to separate columns and store everything as a DataFrame
df = pd.json_normalize(data['collector'])

# Explode the fields containing lists, to separate rows
df = df.explode('hashtags').explode('mentions')

# Converting the dataframe back to JSON format, so we can normalize again
df = df.to_json(orient='records')

# Parse the JSON data
parsed_json = json.loads(df)

# Normalize again and recreate the dataframe
df = pd.json_normalize(parsed_json)
#%cd '/Users/agamchug/Desktop/' 
#df.to_csv('Tiktok.csv', index=False)

# Drop unused column
df = df.drop('hashtags', axis=1)

# %%inspecting dataset
df.head()
#%% 
df.info()
# %% checking for NaN
df.isna().sum(axis=0)
#%% Removing insignificant columns or columns with high number of NaNs
df = df.drop(columns=['videoUrl', 'mentions', 'webVideoUrl', 'videoUrlNoWaterMark', 'downloaded', 'authorMeta.id', 'authorMeta.secUid', 'authorMeta.nickName', 'authorMeta.signature', 'authorMeta.avatar', 'musicMeta.musicId', 'musicMeta.playUrl', 'musicMeta.coverThumb', 'musicMeta.coverMedium', 'musicMeta.coverLarge', 'covers.default', 'covers.origin', 'covers.dynamic'])
#%% checking for duplicates
df.duplicated(), df.duplicated().sum(axis=0)
#shows false for most of them but as we saw in line 38, indices 2,3, and 4 appear to be the same.
# %%
#seems like the last 4 columns have different values for these otherwise duplicates.
#inspecting duplicates exclusing these 4 columns 
print(df.columns)
cols_withDupes = ['id', 'text', 'createTime', 'diggCount',
       'shareCount', 'playCount', 'commentCount', 'authorMeta.name',
       'authorMeta.verified', 'musicMeta.musicName',
       'musicMeta.musicAuthor', 'musicMeta.musicOriginal', 'videoMeta.height',
       'videoMeta.width', 'videoMeta.duration']
df.duplicated(subset=cols_withDupes).sum()
#4693 duplicates
#%% calculating how many tiktoks were duplicates
justDupes = df[df.duplicated(subset=cols_withDupes)==True]
justDupes.groupby(['id']).count() #to get an idea of how many times there is a duplicate for each tiktok
df = df.drop_duplicates(subset=cols_withDupes) #removing the duplicates
#final dataset with 1000 entries only. 
#df.drop([5436], inplace=True)
df.reset_index(inplace=True, drop=True)
# %%
converted = []
for i in df["createTime"]:
    conv = datetime.datetime.fromtimestamp(i).isoformat()
    converted.append(conv)
list(converted)
# %%
# split the date
dateval = [i[:10] for i in converted]
timeval = [i[11:] for i in converted]
# %%
# append onto tiktok df
df["Date_posted"] = dateval
df["Time_posted"] = timeval

postdate = list(df["Date_posted"])
split_date = list(df["Time_posted"].str.split())

day_number = []
for i in postdate:
    dfm = pd.Timestamp(i)
    dyn = dfm.dayofweek
    day_number.append(dyn)

day_of_week = []
for i in day_number:
    if i == 0:
        day_of_week.append("Monday")
    if i == 1:
        day_of_week.append("Tuesday")
    if i == 2:
        day_of_week.append("Wednesday")
    if i == 3:
        day_of_week.append("Thursday")
    if i == 4:
        day_of_week.append("Friday")
    if i == 5:
        day_of_week.append("Saturday")
    if i == 6:
        day_of_week.append("Sunday")

Hour_posted = []
for i in range(len(converted)):
       Hr = pd.Period(converted[i]).hour
       Hour_posted.append(Hr)
df['Hour_posted'] = Hour_posted

Period_posted = []
for i in Hour_posted:
       if i > 0 and i <= 6:
              Period_posted.append('Midnight to 6AM')
       elif i > 6 and i <= 12:
              Period_posted.append('6AM to Noon')
       elif i > 12 and i <= 18:
              Period_posted.append('Noon to 6PM')
       else:
              Period_posted.append('6PM to Midnight')
df['Period_posted'] = Period_posted
df['playCount'].max()
# %%
df['Day_posted'] = day_of_week
#%cd '/Users/agamchug/Desktop/' 
#df.to_csv('Tiktok cleand.csv', index=False)
# %%
df['playCount'] = np.log(df['playCount'])
Mon_posted_v = df[(df['Day_posted']=='Monday') & (df['authorMeta.verified'] == True)]
Mon_posted_nv = df[(df['Day_posted']=='Monday') & (df['authorMeta.verified'] == False)]
Tue_posted_v = df[(df['Day_posted']=='Tuesday') & (df['authorMeta.verified'] == True)]
Tue_posted_nv = df[(df['Day_posted']=='Tuesday') & (df['authorMeta.verified'] == False)]
Wed_posted_v = df[(df['Day_posted']=='Wednesday') & (df['authorMeta.verified'] == True)]
Wed_posted_nv = df[(df['Day_posted']=='Wednesday') & (df['authorMeta.verified'] == False)]
Thu_posted_v = df[(df['Day_posted']=='Thursday') & (df['authorMeta.verified'] == True)]
Thu_posted_nv = df[(df['Day_posted']=='Thursday') & (df['authorMeta.verified'] == False)]
Fri_posted_v = df[(df['Day_posted']=='Friday') & (df['authorMeta.verified'] == True)]
Fri_posted_nv = df[(df['Day_posted']=='Friday') & (df['authorMeta.verified'] == False)]
Sat_posted_v = df[(df['Day_posted']=='Saturday') & (df['authorMeta.verified'] == True)]
Sat_posted_nv = df[(df['Day_posted']=='Saturday') & (df['authorMeta.verified'] == False)]
Sun_posted_v = df[(df['Day_posted']=='Sunday') & (df['authorMeta.verified'] == True)]
Sun_posted_nv = df[(df['Day_posted']=='Sunday') & (df['authorMeta.verified'] == False)]


Mto6 = {'Period_posted': 'Midnight to 6AM', 'playCount': 0}
_6toM = {'Period_posted': '6PM to Midnight', 'playCount': 0}
_6toN = {'Period_posted': '6AM to Noon', 'playCount': 0}


Mon_avg_v = Mon_posted_v.groupby(['Period_posted']).mean()[['playCount']].reset_index()
Mon_avg_v = Mon_avg_v.append(Mto6, ignore_index=True, sort=True).sort_values(['Period_posted']).reset_index(drop=True)
Mon_avg_nv = Mon_posted_nv.groupby(['Period_posted']).mean()[['playCount']].reset_index()

Tue_avg_v = Tue_posted_v.groupby(['Period_posted']).mean()[['playCount']].reset_index()
Tue_avg_v = Tue_avg_v.append(Mto6, ignore_index=True, sort=True).sort_values(['Period_posted']).reset_index(drop=True)
Tue_avg_nv = Tue_posted_nv.groupby(['Period_posted']).mean()[['playCount']].reset_index()

Wed_avg_v = Wed_posted_v.groupby(['Period_posted']).mean()[['playCount']].reset_index().sort_values(['Period_posted'])
Wed_avg_v = Wed_avg_v.append(_6toM, ignore_index=True, sort=True).sort_values(['Period_posted']).reset_index(drop=True)
Wed_avg_nv = Wed_posted_nv.groupby(['Period_posted']).mean()[['playCount']].reset_index()

Thu_avg_v = Thu_posted_v.groupby(['Period_posted']).mean()[['playCount']].reset_index()
Thu_avg_nv = Thu_posted_nv.groupby(['Period_posted']).mean()[['playCount']].reset_index()

Fri_avg_v = Fri_posted_v.groupby(['Period_posted']).mean()[['playCount']].reset_index()
Fri_avg_v = Fri_avg_v.append(Mto6, ignore_index=True, sort=True).sort_values(['Period_posted']).reset_index(drop=True)
Fri_avg_nv = Fri_posted_nv.groupby(['Period_posted']).mean()[['playCount']].reset_index()

Sat_avg_v = Sat_posted_v.groupby(['Period_posted']).mean()[['playCount']].reset_index()
Sat_avg_v = Sat_avg_v.append(_6toN, ignore_index=True, sort=True).sort_values(['Period_posted']).reset_index(drop=True)
Sat_avg_nv = Sat_posted_nv.groupby(['Period_posted']).mean()[['playCount']].reset_index()

Sun_avg_v = Sun_posted_v.groupby(['Period_posted']).mean()[['playCount']].reset_index()
Sun_avg_v = Sun_avg_v.append(Mto6, ignore_index=True, sort=True).sort_values(['Period_posted']).reset_index(drop=True)
Sun_avg_nv = Sun_posted_nv.groupby(['Period_posted']).mean()[['playCount']].reset_index()
#%%
Verified_days = [Mon_avg_v, Tue_avg_v, Wed_avg_v, Thu_avg_v, Fri_avg_v, Sat_avg_v, Sun_avg_v]
NonVerified_days = [Mon_avg_nv, Tue_avg_nv, Wed_avg_nv, Thu_avg_nv, Fri_avg_nv, Sat_avg_nv, Sun_avg_nv]
Verified_days_sorted = []
NonVerified_days_sorted = []
l = [1,3,0,2]
for i in range(len(Verified_days)):
    Verified_days[i]['order'] = l
    Verified_days[i].sort_values(by='order', ascending=True, inplace=True)
    Verified_days_sorted.append(Verified_days[i])
    NonVerified_days[i]['order'] = l
    NonVerified_days[i].sort_values(by='order', ascending=True, inplace=True)
    NonVerified_days_sorted.append(NonVerified_days[i])
# %%

# %%
# %%

import seaborn as sns
sns.lineplot(Mon_avg_v['Period_posted'], Mon_avg_v['playCount'])
# %%
fig = plt.figure(figsize=(15,5))
fig.patch.set_facecolor('azure')
gs = gridspec.GridSpec(
                    1,13, figure=fig, 
                    width_ratios = [10,1,10,1,10,1,10,1,10,1,10,1,10]
                    )               

ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,2])
ax3 = fig.add_subplot(gs[0,4])
ax4 = fig.add_subplot(gs[0,6])
ax5 = fig.add_subplot(gs[0,8])
ax6 = fig.add_subplot(gs[0,10])
ax7 = fig.add_subplot(gs[0,12])
ax8 = fig.add_subplot(gs[0,1]) 
ax9 = fig.add_subplot(gs[0,3])
ax10 = fig.add_subplot(gs[0,5])
ax11 = fig.add_subplot(gs[0,7])
ax12 = fig.add_subplot(gs[0,9])
ax13 = fig.add_subplot(gs[0,11])

plots = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13]
for i in range(7,13):
    plots[i].axvline(0.5, alpha=0.2, color='grey')
    plots[i].axis('off')
for i in plots:
    i.grid(True, axis= 'y', ls='--', color = 'pink')
    i.set_facecolor('azure')

for i in range(len(plots)):
    plots[i].spines['top'].set_visible(False)
    plots[i].spines['right'].set_visible(False)
    plots[i].spines['left'].set_visible(False)
    plots[i].spines['bottom'].set_color("#D9D9D9")
ax1.spines['left'].set_visible(True)
ax1.spines['left'].set_color("#D9D9D9")

for x,y in zip(plots, Verified_days_sorted):
    x.scatter(y['Period_posted'], y['playCount'], color='#1e3d59', marker= '.')
for x,y in zip(plots, NonVerified_days_sorted):
    x.plot(y['Period_posted'], y['playCount'], color='#ffc13b')
    x.scatter(y['Period_posted'], y['playCount'], color='#ffc13b', marker='.')

ax1.plot(['6AM to Noon', 'Noon to 6PM', '6PM to Midnight'], [14.695660, 13.771781, 13.148031], color='#1e3d59')
ax2.plot(['6AM to Noon', 'Noon to 6PM', '6PM to Midnight'], [12.575398, 13.870184, 15.295984], color='#1e3d59')
ax3.plot(['Midnight to 6AM', '6AM to Noon', 'Noon to 6PM'], [12.418063, 13.815511, 12.519973], color='#1e3d59')
ax4.plot(Thu_avg_v['Period_posted'], Thu_avg_v['playCount'], color='#1e3d59')
ax5.plot(['6AM to Noon', 'Noon to 6PM', '6PM to Midnight'], [12.248409, 13.226163, 11.862135], color='#1e3d59')
ax6.plot(['Midnight to 6AM', 'Noon to 6PM'], [14.220976, 12.994629], color='#1e3d59', ls='--', alpha=0.3)
ax6.plot(['Noon to 6PM', '6PM to Midnight'], [12.994629, 15.623665], color='#1e3d59')
ax7.plot(['6AM to Noon', 'Noon to 6PM', '6PM to Midnight'], [14.331648, 14.584947, 13.807085], color='#1e3d59')
ax1.scatter(['Midnight to 6AM'], [10.2], marker= 'x', color = '#1e3d59')
ax2.scatter(['Midnight to 6AM'], [10.2], marker= 'x', color = '#1e3d59')
ax3.scatter(['6PM to Midnight'], [10.2], marker= 'x', color = '#1e3d59')
ax5.scatter(['Midnight to 6AM'], [10.2], marker= 'x', color = '#1e3d59')
ax6.scatter(['6AM to Noon'], [10.2], marker= 'x', color = '#1e3d59')
ax7.scatter(['Midnight to 6AM'], [10.2], marker= 'x', color = '#1e3d59')

for i in plots:
    i.set_ylim(10,16.5)

x_labs = ['Midnight', 'Morning', 'Afternoon', 'Evening']    
for i in range(1,7):
    plots[i].set_yticks(np.arange(10,17,2))
    plots[i].set_yticklabels([])
    plots[i].yaxis.set_ticks_position('none')
    plots[i].set_xticklabels(x_labs, fontsize=8, rotation=30, alpha=0.7, color ='#1e3d59')
    plots[i].tick_params(colors ='#1e3d59', which='both')

ax1.set_yticks(np.arange(10,17,2))
ax1.set_yticklabels(['10','', '','16'], alpha=0.8)
ax1.set_ylabel('log(average views received)', color="#424242")
ax1.set_xticklabels(x_labs, fontsize=8, rotation=30, alpha=0.7, color ='#1e3d59')
ax1.tick_params(colors ='#1e3d59', which='both')

legendElements = [Line2D([0],[0], color='#ffc13b', lw=2.5, label='Unverified Users', alpha=0.8),
                Line2D([0],[0], color='#1e3d59', lw=2.5,  label='Verified Users', alpha=0.8), 
                Line2D([0],[0],marker='x', markerfacecolor='#1e3d59', markeredgecolor='#1e3d59', color='w', label='No Posts')
                ]

ax1.text(1, 15, 'Monday', alpha=0.7, color ='#1e3d59')
ax2.text(0.9, 15, 'Tuesday', alpha=0.7, color ='#1e3d59')
ax3.text(0.5, 15, 'Wednesday', alpha=0.7, color ='#1e3d59')
ax4.text(0.8, 15, 'Thursday', alpha=0.7, color ='#1e3d59')
ax5.text(1, 15, 'Friday', alpha=0.7, color ='#1e3d59')
ax6.text(0.8, 15, 'Saturday', alpha=0.7, color ='#1e3d59')
ax7.text(1, 15, 'Sunday', alpha=0.7, color ='#1e3d59')
fig.suptitle('Variation in average views received by videos, based on period and day of posting.',  color="#424242")
fig.legend(handles = legendElements, loc=(0.84,0.87), frameon=False, fontsize=8, labelcolor='#424242')
plt.show()
