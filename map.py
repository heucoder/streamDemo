
import numpy as np
import pandas as pd
import pydeck
import streamlit as st

df = pd.DataFrame(
   np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
   columns=['lat', 'lon'])


INITIAL_VIEW_STATE = pydeck.ViewState(
  latitude=37.76,
  longitude=-122.4,
  zoom=11,
  max_zoom=16,
  pitch=45,
  bearing=0
)

layer = pydeck.Layer(
            'HexagonLayer',
            data=df,
            get_position='[lon, lat]',
            # get_color='[200, 30, 0, 160]',
            radius=100,
            pickable=True,
            elevation_scale = 4,
            elevation_range=[0, 1000],
            extruded=True,
)

r = pydeck.Deck(
    layers=[layer],
    initial_view_state=INITIAL_VIEW_STATE,
    tooltip={
        'html': '<b>Elevation Value:</b> {elevationValue}',
        'style': {
            'color': 'white'
        }
    }
)

st.pydeck_chart(r)


