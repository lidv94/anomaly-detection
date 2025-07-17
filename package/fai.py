# example data 
#data = data.groupby([x]).agg(y1 = pd.NamedAgg(y1,aggfunc='sum'),y2 = pd.NamedAgg(y2,aggfunc='sum')).reset_index()
#data
#   year	sum_cnt	sum_chequeamount
#	2018	256	    8270384.00
#	2019	1445	55569017.00
#	2020	1186	39513231.00
#	2021	844	    16066940.00
#	2022	555	    14396905.08
#	2023	385	    24893947.00
#	2024	135	    5801244.23

################################################
#plot_dual_axis(data=data, x='year', y1='sum_cnt', y2='sum_chequeamount'
#                ,title='cheque p paytype [chanel agent]'
#                ,xlable='transaction year'
#                ,ylable1='cheque count',ylable2='sum cheque amount'
#                ,width=12, height=6, alpha=0.3)
                
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_dual_axis(data:pd.DataFrame, x:str, y1:str, y2:str, title:str,xlable:str,ylable1:str,ylable2:str, width=12, height=6, alpha=0.3)->None:
    """
        Args:
        data (pd.DataFrame) : sum amount and sum count each agent and each year
        x (str) : year -> columns name
        y1 (str) : count of each year (y1 bar chart) -> columns name
        y2 (str) : amount of each year (y2 line chart) -> columns name
        title (str) : title
        xlable (str) : x title
        ylable (str) : y title
        width (int) : a graph width
        height (int) : a graph height
        alpha (float) : a bar opacity
    """
    fig, ax1 = plt.subplots(figsize=(width, height))
    fig.suptitle(title, fontsize=30)
    color1='tab:gray'
    color2='tab:blue'
    ax2 = ax1.twinx()
    ax1.bar(data[x], data[y1], color=color1, alpha=alpha)
    ax1.set_ylabel(ylable1, color=color1, fontsize=20)
    ax1.set_xlabel(xlable, fontsize=20)
    ax2.plot(data[x], data[y2], color=color2)
    ax2.scatter(data[x], data[y2], color=color2, marker='x')
    ax2.set_ylabel(ylable2, color=color2, fontsize=20)
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    fig.autofmt_xdate(rotation=50)
    
#############################################
#plot_dual_axis_plotly(data=data, x='year', y1='sum_cnt', y2='sum_chequeamount'
#                ,title='cheque p paytype [chanel agent]'
#                ,xlable='transaction year'
#                ,ylable1='cheque count',ylable2='sum cheque amount'
#                ,width=1200, height=600)

import plotly.graph_objects as go

def plot_dual_axis_plotly(data:pd.DataFrame, x:str, y1:str, y2:str, title:str,xlable:str,ylable1:str,ylable2:str, width=1200, height=600):
    """
        Args:
        data (pd.DataFrame) : sum amount and sum count each agent and each year
        x (str) : year -> columns name
        y1 (str) : count of each year (y1 bar chart) -> columns name
        y2 (str) : amount of each year (y2 line chart) -> columns name
        title (str) : title
        xlable (str) : x title
        ylable (str) : y title
        width (int) : a graph width
        height (int) : a graph height
    """
    fig = go.Figure(
        data=go.Bar(
            x=data[x],
            y=data[y1],
            name=ylable1,
            marker=dict(color="royalblue"),
            # hover_data = [hover],
        )
    )

    fig.add_trace(
        go.Scatter(
            x=data[x],
            y=data[y2],
            yaxis="y2",
            name=ylable2,
            marker=dict(color="darkorange"),
            # hover_data = [hover],
        )
    )

    fig.update_layout(
        legend=dict(
            orientation="h",
            x=0.8,
            xanchor="center",
            y=1.1,        # Just below the top of the chart area
            yanchor="top"),
        xaxis_title=xlable,  # Set the x-axis label
        yaxis=dict(
            title=dict(text=ylable1),
            side="left",
            range=[min(0,data[y1].min()), data[y1].max()*1.1],
        ),
        yaxis2=dict(
            title=dict(text=ylable2),
            side="right",
            range=[min(0,data[y2].min()), data[y2].max()*1.1],
            overlaying="y",
            tickmode="sync",
        ),
        title_text=title,
        title={'text': title,'x': 0.5, 'xanchor': 'center','font': {'size': 30}}, # Add figure title
        width=width,  # Set the width of the graph in pixels
        height=height,  # Set the height of the graph in pixels
    )

    fig.show()
    
def month_to_group(day_diff):
    if (day_diff >= 0) & (day_diff <= 3):
        return 3
    elif (day_diff > 3) & (day_diff <= 6):
        return 6
    elif (day_diff > 6) & (day_diff <= 9):
        return 9
    elif (day_diff > 9) & (day_diff <= 12):
        return 12
    elif (day_diff > 12) & (day_diff <= 24):
        return 24
    elif (day_diff > 24) & (day_diff <= 36):
        return 36
    else: 
        return 37
    
def df_unit_time(data:pd.core.frame.DataFrame,event_date:str,ids:str,value:str,agg:str, prefix:str,current_dt_input:str):
    current_dt = pd.to_datetime(current_dt_input)
    data['diffday_current'] = (current_dt - data[event_date]).dt.days
    data['diffmonth_current'] = (data['diffday_current']/30).apply(lambda x: np.ceil(x))
    data['indicator_month'] = data['diffmonth_current'].apply(month_to_group)
    indicator_month_list = sorted(data['indicator_month'].unique())
    sales_id_list = data[ids].unique()
    
# sale id : all month
    all_indicator_month = pd.MultiIndex.from_product([sales_id_list, indicator_month_list], names=[ids, 'indicator_month'])
    all_indicator_month = pd.DataFrame(index=all_indicator_month).reset_index()
# count   
    agent_rolling = data.groupby([ids,'indicator_month']).agg(
        values = pd.NamedAgg(column=value, aggfunc=agg),
        ).sort_values([ids,'indicator_month'], ascending=True).reset_index()
# merge sale id all month vs count
    alls = all_indicator_month.merge(agent_rolling,how='left',on=[ids,'indicator_month'])
    alls['values'] = alls['values'].fillna(0)
    alls['cumsum_'+'values'] = alls.groupby(ids)['values'].transform(pd.Series.cumsum)
# add prefix
    proxy = pd.pivot_table(data=alls,
                               values='cumsum_'+'values',
                               columns='indicator_month',
                               index=ids,
                              ).add_prefix(prefix)
    proxy.index.name = ids
    proxy.columns.name = None
    proxy = proxy.reset_index()
    return alls,proxy

# data: transaction data
# chequeno     policyno     transactiondate amount      sales_id
# 0070001919    22405448    2022-12-30      500000.0    0000000010
# 0070001920    22405448    2022-12-30      135413.0    0000000010

# df1,df2 = df_unit_time(data=agent,event_date='transactiondate',ids='sales_id',value='chequeno',agg='count', prefix='L_cnt_',current_dt_input='2024-06-30')

# df1
# sales_id  indicator_month     values  cumsum_values
# 0000000010    3                0.0    0.0
# 0000000010    6                0.0    0.0
# 0000000010    9                0.0    0.0
# 0000000010    12               0.0    0.0
# 0000000010    24               2.0    2.0
# 0000000010    36               0.0    2.0
# 0000000010    37               0.0    2.0

# df2
# sales_id    L_cnt_3   L_cnt_6 L_cnt_9 L_cnt_12    L_cnt_24    L_cnt_36    L_cnt_37
# 0000000010    0.0      0.0     0.0     0.0          2.0        2.0          2.0

def prep(df_raw, col_date_event, df_map, on_map_left, on_map_right,dict_date1):
    start_dt = '2020-01-01'
    proxy_all = pd.DataFrame()
    
    for i in range(len(dict_date1)):
        # Filter data based on date range
        proxy = df_raw[(df_raw[col_date_event] >= start_dt) & 
                       (df_raw[col_date_event] <= dict_date1.loc[i, 'val_dt'])]

        # Ensure index is reset to avoid index-related issues
        proxy = proxy.reset_index(drop=True)
        
        # Assign scalar value to the entire column
        proxy['yyyy_qq'] = dict_date1.loc[i, 'yyyy_qq']

        # Merge with df_map on specified columns
        proxy = proxy.merge(df_map, how='left', left_on=on_map_left, right_on=on_map_right)
        
        # Filter rows where 'l6_agent_id' is not null
        proxy = proxy[proxy['l6_agent_id'].notna()]
        
        # Concatenate with the overall proxy_all DataFrame
        proxy_all = pd.concat([proxy_all, proxy], axis=0, ignore_index=True)
    
    return proxy_all