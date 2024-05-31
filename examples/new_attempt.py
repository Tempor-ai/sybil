import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Generate dummy data
date_rng = pd.date_range(start='2022-01-01', end='2023-01-01', freq='D')
dummy_values = np.random.randint(0, 100, size=(len(date_rng)))
dataset = pd.DataFrame(date_rng, columns=['Date'])
dataset['Value'] = dummy_values

# Streamlit interface
st.write("### Dataset Line Plot")

# Creating a Plotly figure
fig = go.Figure()

# Adding the time series data to the plot
fig.add_trace(go.Scatter(
    x=dataset['Date'],
    y=dataset['Value'],
    mode='lines+markers',
    name='Data',
    marker=dict(size=5),
    line=dict(width=2)
))

# Enhancements
fig.update_layout(
    title='Univariate Time Series Data',
    xaxis_title='Date',
    yaxis_title='Value',
    template='plotly_white',
    xaxis=dict(
        showgrid=True,
        gridcolor='lightgrey',
        tickformat='%b %Y',
        rangeslider=dict(visible=True)
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor='lightgrey'
    ),
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    )
)

# Adding annotations for some points
for i in range(0, len(dataset), 30):
    fig.add_annotation(
        x=dataset['Date'][i],
        y=dataset['Value'][i],
        text=str(dataset['Value'][i]),
        showarrow=True,
        arrowhead=2,
        ax=0,
        ay=-30
    )

# Display the plot in Streamlit
st.plotly_chart(fig)
