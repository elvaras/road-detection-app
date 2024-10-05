# Road detection on satellite imagery

This is a Python application built using Streamlit. The application provides an interface for exploring and testing the output of a deep learning model on satellite imagery.

<!-- ## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Pages](#pages)
- [Contributing](#contributing)
- [License](#license) -->

## Installation

Install the required packages:
```
pip install -r requirements.txt
```

## Usage

To run the application, execute the following command:
```
streamlit run app.py
```

## Pages

### Explore Model Output

- **File:** `app.py`
- **Description:** This page displays a map with the road detection results of the deep learning model on the satellite imagery.

### Test Model Prediction

- **File:** `page/test_prediction.py`
- **Description:** This page allows the user to select a point on the map to run the model on the selected area.
