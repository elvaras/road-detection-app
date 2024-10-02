import streamlit as st 

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

@tf.keras.utils.register_keras_serializable(name="CustomJaccardLoss")
class CustomJaccardLoss(sm.losses.JaccardLoss):
    @classmethod
    def from_config(cls, config):
        if isinstance(config, dict):
            return cls(**config)
        else:
            return cls()

    def get_config(self):
        return {}

@tf.keras.utils.register_keras_serializable(name="CustomIOUScore")
class CustomIOUScore(sm.metrics.IOUScore):
    @classmethod
    def from_config(cls, config):
        if isinstance(config, dict):
            return cls(**config)
        else:
            return cls()

    def get_config(self):
        return {}

def load_image(input_img_path):
  img = load_img(input_img_path, target_size=(256,256))
  img_array = img_to_array(img)
  img_array = np.expand_dims(img_array, axis=0)

  return img_array

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
    overlay = False,
    control = True
    ).add_to(m)

control = folium.LayerControl(collapsed=False)

m.add_child(folium.LatLngPopup())

fg = []

if 'fg' in st.session_state:
   fg = st.session_state['fg']

with col1:
  st_data = st_folium(m, 
    use_container_width=True,
    feature_group_to_add=fg,
    # returned_objects=["last_clicked"],
    zoom=10,
    center=(-11.1, -69.2),
    layer_control=control)

def reconstruct_geometry(tile, pred):
    zoom_level = 17

    tile_min_lat, tile_min_lon = quadkey.from_tile(tile, zoom_level).to_geo(anchor=quadkey.TileAnchor.ANCHOR_SW)
    tile_max_lat, tile_max_lon = quadkey.from_tile(tile, zoom_level).to_geo(anchor=quadkey.TileAnchor.ANCHOR_NE)

    binary_mask = (pred > 0.5).astype(np.uint8)

    # Find contours using OpenCV. RETR_CCOMP retrieves external and internal contours (holes)
    contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
      return

    lines = []

    contour_polygons = []
    for contour_polygon in contours:
      contour_polygons.append(contour_polygon[:, 0, :])

    child_contours_dict = {}

    # Reconstruct contour hierarchy from findContours results. Parents represent
    # external contours, children are the internal contours (holes in the external hull)
    # See hierarchy description for details
    # https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#ga17ed9f5d79ae97bd4c7cf18403e1689a
    for i, contour_hierarchy in enumerate(hierarchy[0]):
      parent = contour_hierarchy[3]
      if parent != -1:
        if parent in child_contours_dict:
          child_contours_dict[parent].append(i)
        else:
          child_contours_dict[parent] = [i]

    polygons = []

    # Create polygons or multi-polygons for contours
    for i, contour_hierarchy in enumerate(hierarchy[0]):
      if contour_hierarchy[3] != -1: # This is a child contour
        continue

      contour_polygon = contour_polygons[i]

      if len(contour_polygon) < 4:
        continue

      if i in child_contours_dict:
        child_contour_polygons = [contour_polygons[child_idx] for child_idx in child_contours_dict[i]]

        # If the contour has holes, create a multi-polygon object
        # The first element in the tuple represents the external contour
        # The second element contains all internal contours (holes)
        multi_polygon = MultiPolygon([ (contour_polygon, child_contour_polygons) ])
        polygons.append(multi_polygon)

      else:
        polygons.append(Polygon(contour_polygon))

    # Convert the contours to lines by extracting their centerline
    for polygon in polygons:
      try:
        centerline = pygeoops.centerline(polygon)
        lines.append(centerline)
      except Exception as e:
        print('Failed to find centerline for polygon:', polygon)
        print(e)

    # Convert to geo-referenced coordinates
    geo_lines = []

    height, width = (256, 256)

    for line in lines:
      geo_line = shapely.transform(line, lambda x: ([tile_min_lon, tile_max_lat] + x / [width, height] * [tile_max_lon - tile_min_lon, tile_min_lat - tile_max_lat]))
      geo_lines.append(geo_line)

    geo_lines.append(LineString([(tile_min_lon, tile_min_lat), (tile_min_lon, tile_max_lat), (tile_max_lon, tile_max_lat), (tile_max_lon, tile_min_lat), (tile_min_lon, tile_min_lat)]))

    gdf = gpd.GeoDataFrame(geometry=geo_lines, crs='EPSG:4326')

    return gdf

def refresh_map(fg=None):
   st_data = st_folium(m, 
    width=1200,
    height=400,
    feature_group_to_add=fg,
    layer_control=control)
   
   return st_data

def process_coordinates(lat, lon):
    zoom_level = 17

    tile = quadkey.from_geo((lat, lon), zoom_level).to_tile()[0]
    tile_url = f'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom_level}/{tile[1]}/{tile[0]}'

    with urllib.request.urlopen(tile_url) as url:
        img = load_image(BytesIO(url.read()))

    with col2:
        st.write('Satellite image')
        st.image(img / 255)

    pred = model.predict(img)

    with col2:
        st.write('Geometry prediction')
        st.image(pred)

    gdf = reconstruct_geometry(tile, pred[0])

    fg = folium.FeatureGroup('Predictions')

    if gdf is not None:
      folium_shapes = folium.GeoJson(data=gdf,
          style_function=lambda feature: {
              "color": "red",
              "weight": 2,
              "opacity": 1,
          })

      fg.add_child(folium_shapes)

    st.session_state['fg'] = fg

# Capture the clicked position
if st_data and 'last_clicked' in st_data and st_data['last_clicked']:
    clicked_lat = st_data['last_clicked']['lat']
    clicked_lon = st_data['last_clicked']['lng']

    # Show a message with the clicked coordinates
    with col2:
      st.success(f"You can find the satellite image and predicted raster below. The reconstructed geometry is shown directly on the map")
      st.warning("If the geometry doesn't show on the map, please drag it to refresh")

    # Further process the coordinates in a function
    process_coordinates(clicked_lat, clicked_lon)


