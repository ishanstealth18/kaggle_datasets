import pandas as pd

original_df = pd.read_csv('Metro_Interstate_Traffic_Volume.csv')

original_df['date_time'] = pd.to_datetime(original_df['date_time'])

print(original_df.info())


day_mask = (original_df['date_time'].dt.hour >= 6) & (original_df['date_time'].dt.hour < 18)

day_df= original_df[day_mask]
night_df = original_df[~day_mask]

day_df = day_df.drop(columns=['holiday'], axis=1)
night_df = night_df.drop(columns=['holiday'], axis=1)

day_df['year'] = day_df['date_time'].dt.year
day_df['month'] = day_df['date_time'].dt.month
day_df['day'] = day_df['date_time'].dt.day
day_df['hour'] = day_df['date_time'].dt.hour

day_df = day_df.sort_values(by=['day', 'hour'])
day_only_df = day_df.groupby(['day']).agg(avg_traffic_volume=('traffic_volume', 'mean'))
hour_only_df = day_df.groupby(['hour'], as_index=False).agg(avg_traffic_volume=('traffic_volume', 'mean'))


night_df['year'] = night_df['date_time'].dt.year
night_df['month'] = night_df['date_time'].dt.month
night_df['day'] = night_df['date_time'].dt.day
night_df['hour'] = night_df['date_time'].dt.hour

#print(night_df['hour'].unique())

#night_df = night_df.sort_values(by=['day'])

night_only_df = night_df.groupby(['day']).agg(avg_traffic_volume=('traffic_volume', 'mean'))
night_hour_only_df = night_df.groupby(['hour'], as_index=False).agg(avg_traffic_volume=('traffic_volume', 'mean'))
combined_hour_df = pd.concat([hour_only_df, night_hour_only_df], axis=0)
combined_hour_df['hour'] = pd.to_datetime(combined_hour_df['hour'], format='%H')
combined_hour_df['hour'] = combined_hour_df['hour'].dt.strftime("%H:%M")
#print(combined_hour_df.head())


# day and night time with most traffic volume
import seaborn as sns
from matplotlib import pyplot as plt

#sns.lineplot(data=day_only_df, x='day', y='avg_traffic_volume', estimator=None, sort=False)
#plt.show()
#sns.lineplot(data=hour_only_df, x='hour', y='avg_traffic_volume', estimator=None, sort=False)
#plt.show()


#sns.lineplot(data=night_only_df, x='day', y='avg_traffic_volume', estimator=None, sort=False)
#plt.show()
#sns.lineplot(data=combined_hour_df, x='hour', y='avg_traffic_volume', estimator=None, sort=False)
#plt.show()

# Effect of weather
# day rain
day_rain_outlier = list(day_df[day_df['rain_1h'] > 30].index)
day_df = day_df.drop(index=day_rain_outlier)
day_rain_df = day_df.groupby(['rain_1h'], as_index=False).agg(avg_traffic_volume=('traffic_volume', 'mean'))
#sns.lineplot(data=day_rain_df, x='rain_1h', y='avg_traffic_volume', estimator=None, sort=False)
#plt.title('Day Time Rain Traffic Volume')
#plt.xlabel('Rain (mm)')
#plt.show()

# night rain

night_rain_outlier = list(night_df[night_df['rain_1h'] > 30].index)
night_df = night_df.drop(index=night_rain_outlier)
night_rain_df = night_df.groupby(['rain_1h'], as_index=False).agg(avg_traffic_volume=('traffic_volume', 'mean'))
#sns.lineplot(data=night_rain_df, x='rain_1h', y='avg_traffic_volume', estimator=None, sort=False)
#plt.title('Night Time Rain Traffic Volume')
#plt.xlabel('Rain (mm)')
#plt.show()

# Day snow
day_snow_df = day_df.groupby(['snow_1h'], as_index=False).agg(avg_traffic_volume=('traffic_volume', 'mean'))
sns.lineplot(data=day_snow_df, x='snow_1h', y='avg_traffic_volume', estimator=None, sort=False)
plt.title('Day Time Snow Traffic Volume')
plt.xlabel('Snow (mm)')
plt.show()

# Night snow
night_snow_df = night_df.groupby(['snow_1h'], as_index=False).agg(avg_traffic_volume=('traffic_volume', 'mean'))
sns.lineplot(data=night_snow_df, x='snow_1h', y='avg_traffic_volume', estimator=None, sort=False)
plt.title('Night Time Snow Traffic Volume')
plt.xlabel('Snow (mm)')
plt.show()


fig,axes = plt.subplots(2,2)
sns.lineplot(data=night_rain_df, x='rain_1h', y='avg_traffic_volume', estimator=None, sort=False, ax=axes[0][0])
plt.title('Night Time Rain Traffic Volume')
plt.xlabel('Rain (mm)')

sns.lineplot(data=night_rain_df, x='rain_1h', y='avg_traffic_volume', estimator=None, sort=False, ax=axes[0][1])
plt.title('Night Time Rain Traffic Volume')
plt.xlabel('Rain (mm)')

sns.lineplot(data=day_snow_df, x='snow_1h', y='avg_traffic_volume', estimator=None, sort=False, ax=axes[1][0])
plt.title('Day Time Snow Traffic Volume')
plt.xlabel('Snow (mm)')

sns.lineplot(data=night_snow_df, x='snow_1h', y='avg_traffic_volume', estimator=None, sort=False, ax=axes[1][1])
plt.title('Night Time Snow Traffic Volume')
plt.xlabel('Snow (mm)')

plt.tight_layout()
plt.show()

# heatmap
day_heatmap_data = day_df[['rain_1h', 'snow_1h', 'traffic_volume', 'day','hour']]
day_heatmap_filter = day_heatmap_data.groupby(['rain_1h', 'snow_1h', 'day', 'hour']).agg(avg_traffic_volume=('traffic_volume', 'mean'))
print(day_heatmap_filter.head())
traffic_pivot = day_heatmap_data.pivot_table(
    values="traffic_volume", index="hour", columns=['day'], aggfunc="mean")
print(traffic_pivot)

sns.heatmap(traffic_pivot, cmap="coolwarm",  # Light colors for low traffic, deep blue for heavy traffic

    fmt=".0f",  # Round values to whole integers
    linewidths=0.5,  # Add a subtle grid line between cells
    cbar_kws={"label": "Average Traffic Volume"},)
plt.show()