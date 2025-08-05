import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Airbnb Price Predictor", layout="wide")

# --- City mapping from notebook ---
city_map = {
    'Paris': 0,
    'Rome': 1,
    'Amsterdam': 2,
    'Berlin': 3,
    'Prague': 4,
    'Barcelona': 5,
    'Budapest': 6,
    'Vienna': 7,
    'Athens': 8,
    'Istanbul': 9,
    'Dublin': 10,
    'Oslo': 11,
    'Stockholm': 12,
    'Copenhagen': 13,
    'Brussels': 14
}

# City coordinates for map initialization
city_coords = {
    "Paris": (48.8566, 2.3522),
    "Rome": (41.9028, 12.4964),
    "Amsterdam": (52.3676, 4.9041),
    "Berlin": (52.5200, 13.4050),
    "Prague": (50.0755, 14.4378),
    "Barcelona": (41.3851, 2.1734),
    "Budapest": (47.4979, 19.0402),
    "Vienna": (48.2082, 16.3738),
    "Athens": (37.9838, 23.7275),
    "Istanbul": (41.0082, 28.9784),
    "Dublin": (53.3498, -6.2603),
    "Oslo": (59.9139, 10.7522),
    "Stockholm": (59.3293, 18.0686),
    "Copenhagen": (55.6761, 12.5683),
    "Brussels": (50.8503, 4.3517)
}

# Room type mapping from notebook
room_type_map = {
    'Entire home/apt': 0,
    'Private room': 1,
    'Shared room': 2,
    'Hotel room': 3
}

@st.cache_resource
def load_model_and_resources():
    """Load the trained model, scaler, and neighbourhood frequency mapping"""
    try:
        model = joblib.load("lightgbm_airbnb_model.pkl")
        scaler = joblib.load("feature_scaler.pkl")
        neigh_freq_map = joblib.load("neigh_freq_map.pkl")
        return model, scaler, neigh_freq_map
    except FileNotFoundError as e:
        st.error(f"Required files not found: {e}")
        st.error("Please ensure the following files are in the same directory as this app:")
        st.error("- lightgbm_airbnb_model.pkl")
        st.error("- feature_scaler.pkl") 
        st.error("- neigh_freq_map.pkl")
        st.stop()

model, scaler, neigh_freq_map = load_model_and_resources()

# Get available neighbourhoods from the frequency map
available_neighbourhoods = list(neigh_freq_map.keys())

feature_order = [
    'latitude', 'longitude', 'room_type', 'minimum_nights', 
    'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count', 
    'availability_365', 'number_of_reviews_ltm', 'city', 'neighbourhood_freq_enc'
]

# --- Streamlit UI ---
st.title("🏠 Airbnb Price Predictor")
st.markdown("Enter Airbnb listing details to predict the nightly price for major European cities.")
st.markdown("---")

# Create columns for better layout
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Listing Details")

    # City selection with automatic coordinate update
    city = st.selectbox("🌍 City", list(city_map.keys()))

    # Initialize session state for coordinates
    if ( 'latitude_input' not in st.session_state or 
         'longitude_input' not in st.session_state or 
         st.session_state.get('last_city', None) != city ):
        default_lat, default_lon = city_coords[city]
        st.session_state['latitude_input'] = default_lat
        st.session_state['longitude_input'] = default_lon
        st.session_state['last_city'] = city

    latitude = st.number_input(
        "📍 Latitude", 
        value=st.session_state['latitude_input'], 
        format="%.6f",
        key="latitude_input",
        help="Geographic latitude coordinate"
    )

    longitude = st.number_input(
        "📍 Longitude", 
        value=st.session_state['longitude_input'], 
        format="%.6f",
        key="longitude_input",
        help="Geographic longitude coordinate"
    )

    # Neighbourhood selection
    neighbourhood = st.selectbox(
        "🏘️ Neighbourhood", 
        available_neighbourhoods,
        help="Select the neighbourhood for frequency encoding"
    )

    room_type = st.selectbox("🏠 Room Type", list(room_type_map.keys()))

    minimum_nights = st.slider("🌙 Minimum Nights", 1, 30, 1)

    number_of_reviews = st.slider("⭐ Total Reviews", 0, 500, 0)

    reviews_per_month = st.number_input(
        "📅 Reviews per Month", 
        value=0.0, 
        min_value=0.0, 
        max_value=50.0, 
        step=0.1, 
        format="%.2f"
    )

    calculated_host_listings_count = st.slider("🏘️ Host Listings Count", 1, 10, 1)

    availability_365 = st.slider("📆 Availability (days/year)", 0, 365, 180)

    number_of_reviews_ltm = st.slider("📈 Reviews Last 12 Months", 0, 500, 0)

with col2:
    st.header("Location Preview")

    # Create a map centered on the selected coordinates
    m = folium.Map(location=[latitude, longitude], zoom_start=13)

    # Add marker for the listing location
    folium.Marker(
        [latitude, longitude],
        popup=f"{city} - {neighbourhood}<br>{room_type}",
        tooltip=f"Click for details",
        icon=folium.Icon(color='red', icon='home')
    ).add_to(m)

    # Display the map
    st_folium(m, height=400, width=700)

# --- Prediction Section ---
st.markdown("---")
st.header("💰 Price Prediction")

if st.button("Predict Price", type="primary", use_container_width=True):
    # Get neighbourhood frequency encoding
    neighbourhood_freq_enc = neigh_freq_map.get(neighbourhood, 0)  # Default to 0 if not found

    # Prepare input data with correct feature order and label encoding
    input_dict = {
        'latitude': latitude,
        'longitude': longitude,
        'room_type': room_type_map[room_type],  # Encode room type
        'minimum_nights': minimum_nights,
        'number_of_reviews': number_of_reviews,
        'reviews_per_month': reviews_per_month,
        'calculated_host_listings_count': calculated_host_listings_count,
        'availability_365': availability_365,
        'number_of_reviews_ltm': number_of_reviews_ltm,
        'city': city_map[city],  # Encode city
        'neighbourhood_freq_enc': neighbourhood_freq_enc  # Add neighbourhood encoding
    }

    # Create DataFrame with correct column order
    input_df = pd.DataFrame([input_dict], columns=feature_order)

    # Debug: Show the input data
    with st.expander("🔍 Debug: Input Data"):
        st.write("Input DataFrame shape:", input_df.shape)
        st.write("Input DataFrame columns:", list(input_df.columns))
        st.write("Scaler expects features:", list(scaler.feature_names_in_) if hasattr(scaler, 'feature_names_in_') else "Feature names not available")
        st.dataframe(input_df)

    try:
        # Scale features (as done in notebook)
        X_scaled = scaler.transform(input_df)

        # Make prediction (model returns log-transformed price)
        pred_price_log = model.predict(X_scaled)[0]

        # Transform back to original scale using exp() (as in notebook)
        pred_price = np.exp(pred_price_log)

        # Display results
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label="Predicted Price", 
                value=f"€{pred_price:.0f}",
                help="Estimated nightly price in Euros"
            )

        with col2:
            st.metric(
                label="Weekly Price", 
                value=f"€{pred_price * 7:.0f}",
                help="Estimated weekly price"
            )

        with col3:
            st.metric(
                label="Monthly Price", 
                value=f"€{pred_price * 30:.0f}",
                help="Estimated monthly price"
            )

        # Additional insights
        st.markdown("### 📊 Listing Summary")

        summary_data = {
            "Feature": ["City", "Neighbourhood", "Room Type", "Location", "Minimum Stay", "Host Listings", "Availability"],
            "Value": [
                city,
                neighbourhood,
                room_type,
                f"{latitude:.4f}, {longitude:.4f}",
                f"{minimum_nights} night(s)",
                f"{calculated_host_listings_count} listing(s)",
                f"{availability_365} days/year"
            ]
        }

        summary_df = pd.DataFrame(summary_data)
        st.table(summary_df)

        # Show encoded values for debugging
        with st.expander("🔧 Encoded Values"):
            st.write(f"City encoded: {city_map[city]}")
            st.write(f"Room type encoded: {room_type_map[room_type]}")
            st.write(f"Neighbourhood frequency encoded: {neighbourhood_freq_enc}")

    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")
        st.error("Please check that all required model files are present and compatible.")

        # Show detailed error info
        with st.expander("🔍 Error Details"):
            st.write("Error type:", type(e).__name__)
            st.write("Error message:", str(e))
            if hasattr(scaler, 'feature_names_in_'):
                st.write("Scaler expects features:", list(scaler.feature_names_in_))
            st.write("Provided features:", list(input_df.columns))

# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    <small>
    🤖 Powered by LightGBM Machine Learning Model<br>
    📊 Trained on European Airbnb data with neighbourhood frequency encoding<br>
    💡 Predictions are estimates and may vary from actual market prices
    </small>
    </div>
    """, 
    unsafe_allow_html=True
)

# --- Sidebar Information ---
with st.sidebar:
    st.markdown("---")
    st.markdown("### ℹ️ Model Information")
    st.markdown(f"**Available neighbourhoods:** {len(available_neighbourhoods)}")
    st.markdown(f"**Cities supported:** {len(city_map)}")
    st.markdown(f"**Features used:** {len(feature_order)}")

    with st.expander("📋 Feature List"):
        for i, feature in enumerate(feature_order, 1):
            st.write(f"{i}. {feature}")

    with st.expander("🏙️ City Encodings"):
        for city_name, encoding in city_map.items():
            st.write(f"{city_name}: {encoding}")
