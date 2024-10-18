import streamlit as st 

from helper.helper_functions import *

import os
import random
import re
import shutil
from datetime import datetime

from tqdm import tqdm

import numpy as np

from IPython.display import Image, display, clear_output
from PIL import ImageOps

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, imshow, axis
from matplotlib.image import imread
from matplotlib.ticker import MaxNLocator

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import models
from tensorflow.python import keras
from tensorflow import data as tf_data
from tensorflow import image as tf_image
from tensorflow import io as tf_io
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

from keras.models import load_model
from keras.callbacks import ModelCheckpoint

os.environ['SM_FRAMEWORK'] = 'tf.keras'

import segmentation_models as sm
from segmentation_models import Unet
from segmentation_models import get_preprocessing
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score

import mlflow

from pyquadkey2 import quadkey
import cv2
from skimage import measure

import shapely
from shapely.geometry import LineString, Polygon, MultiPolygon

import geopandas as gpd
import pygeoops

import folium
from streamlit_folium import st_folium

import urllib
from io import BytesIO

st.set_page_config(
  page_title="Test model prediction",
  layout="wide"
)

st.sidebar.title("Road detection in satellite imagery")

st.sidebar.page_link("app.py", label="Explore model output")
st.sidebar.page_link("pages/test_prediction.py", label="Test model prediction")

st.subheader("Test model prediction")

MLFLOW_REPO_URL = "https://dagshub.com/pbvaras/mvp-mbit.mlflow"
mlflow.set_tracking_uri(MLFLOW_REPO_URL)

custom_objects={
    'CustomJaccardLoss': CustomJaccardLoss,
    'CustomIOUScore': CustomIOUScore
    }

@st.cache_resource(show_spinner='Loading model...')
def load_model():
    model_uri = f"models:/mvp-mbit/1"

    model = mlflow.keras.load_model(model_uri, custom_objects=custom_objects)

    return model

model = load_model()

st.warning("Select a point in the map to get the predicted geometries for the tile")

col1, col2 = st.columns([2,1], gap='medium')

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

fg = []

if 'fg' in st.session_state:
   fg = st.session_state['fg']

with col1:
  st_data = st_folium(m, 
    use_container_width=True,
    feature_group_to_add=fg,
    returned_objects=['last_clicked'],
    zoom=10,
    center=(-11.1, -69.2),
    layer_control=control)

def refresh_map(fg=None):
   st_data = st_folium(m, 
    width=1200,
    height=400,
    feature_group_to_add=fg,
    layer_control=control)
   
   return st_data

def process_coordinates(lat, lon):

    if 'last_clicked' in st.session_state and st.session_state['last_clicked']['lat'] == lat and st.session_state['last_clicked']['lng'] == lon:
        print('return')
        return
    
    print('process coordinates')
    st.session_state['last_clicked'] = {'lat': lat, 'lng': lon}

    zoom_level = 17

    tile = quadkey.from_geo((lat, lon), zoom_level).to_tile()[0]
    tile_url = f'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom_level}/{tile[1]}/{tile[0]}'

    with urllib.request.urlopen(tile_url) as url:
        img = load_image(BytesIO(url.read()))
        st.session_state['img'] = img

    with col2:
        st.write('Satellite image')
        st.image(img / 255)

    pred = model.predict(img)
    st.session_state['pred'] = pred

    with col2:
        st.write('Geometry prediction')
        st.image(pred)

    gdf = reconstruct_geometry(tile, pred[0])

    fg = folium.FeatureGroup('Predictions')

    if gdf is not None:
      folium_shapes = folium.GeoJson(data=gdf,
          style_function=lambda feature: {
              "color": "red",
              "weight": 4,
              "opacity": 1,
          })

      fg.add_child(folium_shapes)

    st.session_state['fg'] = fg

    st.rerun()

saved_last_clicked = None

# Capture the clicked position
if st_data and 'last_clicked' in st_data and st_data['last_clicked'] and st_data['last_clicked'] != saved_last_clicked:

    clicked_lat = st_data['last_clicked']['lat']
    clicked_lon = st_data['last_clicked']['lng']

    # Show a message with the clicked coordinates
    with col2:
      st.success(f"Satellite image and predicted raster will show below. Find the reconstructed geometry directly on the map")

    # Further process the coordinates in a function
    process_coordinates(clicked_lat, clicked_lon)


if 'img' in st.session_state and 'pred' in st.session_state:
    with col2:
        st.write('Satellite image')
        st.image(st.session_state['img'][0] / 255)

        st.write('Geometry prediction')
        st.image(st.session_state['pred'][0])