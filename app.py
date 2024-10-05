import streamlit as st 

import os
import zipfile

import pandas as pd
import geopandas as gpd

import folium
from streamlit_folium import st_folium

import os

st.set_page_config(page_title="Explore model output", layout="wide")

st.sidebar.title("Road detection in satellite imagery")

st.sidebar.page_link("app.py", label="Explore model output")
st.sidebar.page_link("pages/test_prediction.py", label="Test model prediction")

st.subheader("Explore model output")

@st.cache_data(show_spinner='Loading data...')
def load_geojson():
    print('Reading GeoJSON files...')

    # Read GeoJSON data into GeoDataFrames
    gdf_train = gpd.read_file("data/predictions/geojson/train.json")
    gdf_train['dataset'] = 'train'
    gdf_train['color'] = 'blue'

    gdf_val = gpd.read_file("data/predictions/geojson/val.json")
    gdf_val['dataset'] = 'val'
    gdf_val['color'] = 'orange'

    gdf_test = gpd.read_file("data/predictions/geojson/test.json")
    gdf_test['dataset'] = 'test'
    gdf_test['color'] = 'yellow'

    all_gdfs = {
        'train': gdf_train,
        'val': gdf_val,
        'test': gdf_test,
    }

    print('Reading GeoJSON files... Done')
    
    return all_gdfs

all_gdfs = load_geojson()

st.write("The map below shows the road detection results of the model on the satellite imagery.")

all_fgs = []

with st.spinner("Preparing data..."):
    print("Create feature groups...")

    symbols = {'train':'&#128309;', 'val':'&#128992;', 'test':'&#128993;'}

    for key, gdf in all_gdfs.items(): 
        folium_shapes = folium.GeoJson(data=gdf,
            style_function=lambda feature: {
                "color": feature["properties"]["color"],
                "weight": 2,
                "opacity": 1,
            })

        fg = folium.FeatureGroup(name=f'{key} {symbols[key]}')
        fg.add_child(folium_shapes)

        all_fgs.append(fg)

    print("Create feature groups... Done")

with st.spinner("Loading map..."):
    print("Loading map...")

    m = folium.Map(min_zoom = 10, max_zoom = 17)

    tile = folium.TileLayer(
        tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr = 'Esri',
        name = 'Esri Satellite',
        max_zoom = 17,
        overlay = False,
        control = True
        ).add_to(m)

    control = folium.LayerControl(collapsed=False)

    folium.plugins.Fullscreen(
        position="topright",
        title="Enter fullscreen",
        title_cancel="Exit fullscreen",
        force_separate_button=True,
    ).add_to(m)

    st_folium(m, 
        use_container_width=True,
        feature_group_to_add=all_fgs,
        returned_objects=[],
        zoom=10,
        center=(-11.1, -69.2),
        layer_control=control)

    print("Loading map... Done")
