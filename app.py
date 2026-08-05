import pandas as pd
import streamlit as st
import plotly.express as px
import os
os.chdir(os.path.join(os.path.dirname(__file__), 'scripts'))
from scripts.live_forecast import forecast

st.set_page_config(page_title='Green Hydrogen Forecast', layout='wide',
                   initial_sidebar_state='collapsed'
                   )
st.markdown('---')

def load_data():
    df = forecast()
    return df.reset_index()

df = load_data()

col_header, col_selector = st.columns([7, 3])
with col_header:
    st.title("Green Hydrogen Forecast Dashboard")

with col_selector:
    days = {0 : "Today (Day 0)",
            1 : "Tomorrow (Day 1)",
            2 : "Day 2",
            3 : "Day 3",
            4 : "Day 4",
            5 : "Day 5",
            6 : "Day 6"
            }
    select_day = st.selectbox("Select Forecast Day",
                              options=list(days.keys()),
                              format_func=days.get
                              )
st.markdown('---')

df_day = df[df['Relative Day'] == select_day].copy()

total_yield = df_day['Predicted_Hydrogen_yield(kg)'].sum()
shutdown_hrs = df_day['Predicted_Shutdown'].sum()

col1, col2, col3 = st.columns(3)
with col1:
    with st.container(border=True):
        st.metric(f"Predicted Yield (Day {select_day}):", f"{total_yield:.2f} kg")   

with col2:
    with st.container(border=True):
        st.metric("Shutdown Hours:", f"{shutdown_hrs} hrs")
with col3:
    with st.container(border=True):
        if shutdown_hrs == 0:
            st.success("Full Operations")
        else:
            st.error(f"{shutdown_hrs} Offline hours predicted")

st.subheader(f"Hourly Green Hydrogen Production (Day {select_day})")

threshold_val = 3.8 * 1000 / 53.2
marker_colors = []

for i in df_day['Predicted_Shutdown']:
    if i == 1:
        marker_colors.append('red')
    else:
        marker_colors.append('green')

fig = px.line(df_day, x='Timestamp',
              y='Predicted_Hydrogen_yield(kg)',
              markers=True, labels={'Predicted_Hydrogen_yield(kg)' : 'Yield(kg)' , 'Timestamp' : 'Time of Day'}
              )

fig.update_traces(line_width=3, marker=dict(size=9, color=marker_colors))

fig.add_hline(y=threshold_val, line_dash='dash',
              line_color='red', line_width=2,
              annotation_text="Shutdown Threshold",
              annotation_position='bottom right',
              annotation_font_color='red'
              )

fig.add_hrect(y0=0, y1=threshold_val,
              fillcolor='red', opacity=0.1, line_width=0.1
              )

fig.update_layout(height=450, hovermode='x unified',
                  margin=dict(l=30, r=30, t=40, b=30)
                  )

st.plotly_chart(fig, use_container_width=True)
st.info("Chart Legend:\n\n"
        "Red Marker: Predicted Plant Shutdown\n\n"
        "Green Marker: Predicted Plant Operating Safely"
        )

with st.expander("View Weather Telemetry (GHI and Windspeed)"):
    tab1, tab2 = st.tabs(["Solar Irradiance (GHI)", "Windspeed"])
    
    with tab1:
        fig_ghi = px.line(
            df_day, x='Timestamp', y='GHI(W/m2)', 
            title="Hourly GHI (W/m2)"
        )
        fig_ghi.update_traces(line_color='orange')
        fig_ghi.update_layout(height=300, hovermode='x unified', margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_ghi, use_container_width=True)
        
    with tab2:
        fig_wind = px.line(
            df_day, x='Timestamp', y='Windspeed(m/s)', 
            title="Hourly Windspeed (m/s)",
            labels={'Windspeed(m/s)': 'Windspeed (m/s)'}
        )
        fig_wind.update_traces(line_color='lightgreen')
        fig_wind.update_layout(height=300, hovermode='x unified', margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_wind, use_container_width=True)