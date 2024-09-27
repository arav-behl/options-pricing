import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
from numpy import log, sqrt, exp
import matplotlib.pyplot as plt
import seaborn as sns
from blackscholes import BlackScholes

#######################
# Page configuration
st.set_page_config(
    page_title="Black-Scholes Option Pricing Model",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded")


# Custom CSS to inject into Streamlit
st.markdown("""
<style>
/* Adjust the size and alignment of the CALL and PUT value containers */
.metric-container {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 8px; /* Adjust the padding to control height */
    width: auto; /* Auto width for responsiveness, or set a fixed width if necessary */
    margin: 0 auto; /* Center the container */
}

/* Custom classes for CALL and PUT values */
.metric-call {
    background-color: #90ee90; /* Light green background */
    color: black; /* Black font color */
    margin-right: 10px; /* Spacing between CALL and PUT */
    border-radius: 10px; /* Rounded corners */
}

.metric-put {
    background-color: #ffcccb; /* Light red background */
    color: black; /* Black font color */
    border-radius: 10px; /* Rounded corners */
}

/* Style for the value text */
.metric-value {
    font-size: 1.5rem; /* Adjust font size */
    font-weight: bold;
    margin: 0; /* Remove default margins */
}

/* Style for the label text */
.metric-label {
    font-size: 1rem; /* Adjust font size */
    margin-bottom: 4px; /* Spacing between label and value */
}

</style>
""", unsafe_allow_html=True)

# Sidebar for User Inputs
with st.sidebar:
    st.title("📊 Black-Scholes Model")
    st.write("`Created by:`")
    linkedin_url = "https://www.linkedin.com/in/arav-behl-0524a6230/"
    st.markdown(f'<a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">`Arav Behl`</a>', unsafe_allow_html=True)

    current_price = st.number_input("Current Asset Price", value=100.0)
    strike = st.number_input("Strike Price", value=100.0)
    time_to_maturity = st.number_input("Time to Maturity (Years)", value=1.0)
    volatility = st.number_input("Volatility (σ)", value=0.2)
    interest_rate = st.number_input("Risk-Free Interest Rate", value=0.05)
    call_purchase_price = st.number_input("Call Purchase Price", value=10.0)
    put_purchase_price = st.number_input("Put Purchase Price", value=10.0)

    st.markdown("---")
    calculate_btn = st.button('Heatmap Parameters')
    spot_min = st.number_input('Min Spot Price', min_value=0.01, value=current_price*0.8, step=0.01)
    spot_max = st.number_input('Max Spot Price', min_value=0.01, value=current_price*1.2, step=0.01)
    vol_min = st.slider('Min Volatility for Heatmap', min_value=0.01, max_value=1.0, value=volatility*0.5, step=0.01)
    vol_max = st.slider('Max Volatility for Heatmap', min_value=0.01, max_value=1.0, value=volatility*1.5, step=0.01)
    
    spot_range = np.linspace(spot_min, spot_max, 10)
    vol_range = np.linspace(vol_min, vol_max, 10)

    st.markdown("---")
    st.subheader("Market Prices for Implied Volatility")
    market_call_price = st.number_input("Market Call Price", value=call_purchase_price, step=0.01)
    market_put_price = st.number_input("Market Put Price", value=put_purchase_price, step=0.01)


def plot_heatmap(bs_model, spot_range, vol_range, strike, call_purchase_price, put_purchase_price):
    call_pnl = np.zeros((len(vol_range), len(spot_range)))
    put_pnl = np.zeros((len(vol_range), len(spot_range)))
    
    for i, vol in enumerate(vol_range):
        for j, spot in enumerate(spot_range):
            bs_temp = BlackScholes(
                time_to_maturity=bs_model.time_to_maturity,
                strike=strike,
                current_price=spot,
                volatility=vol,
                interest_rate=bs_model.interest_rate
            )
            bs_temp.run()
            call_pnl[i, j] = bs_temp.calculate_pnl(call_purchase_price, option_type='call')
            put_pnl[i, j] = bs_temp.calculate_pnl(put_purchase_price, option_type='put')
    
    # Plotting Call P&L Heatmap
    fig_call, ax_call = plt.subplots(figsize=(10, 8))
    sns.heatmap(call_pnl, xticklabels=np.round(spot_range, 2), yticklabels=np.round(vol_range, 2), annot=True, fmt=".2f", cmap="RdYlGn", ax=ax_call)
    ax_call.set_title('CALL P&L')
    ax_call.set_xlabel('Spot Price')
    ax_call.set_ylabel('Volatility')
    
    # Plotting Put P&L Heatmap
    fig_put, ax_put = plt.subplots(figsize=(10, 8))
    sns.heatmap(put_pnl, xticklabels=np.round(spot_range, 2), yticklabels=np.round(vol_range, 2), annot=True, fmt=".2f", cmap="RdYlGn", ax=ax_put)
    ax_put.set_title('PUT P&L')
    ax_put.set_xlabel('Spot Price')
    ax_put.set_ylabel('Volatility')
    
    return fig_call, fig_put


# Main Page for Output Display
st.title("Black-Scholes Pricing Model")

# Table of Inputs
input_data = {
    "Current Asset Price": [current_price],
    "Strike Price": [strike],
    "Time to Maturity (Years)": [time_to_maturity],
    "Volatility (σ)": [volatility],
    "Risk-Free Interest Rate": [interest_rate],
    "Call Purchase Price": [call_purchase_price],
    "Put Purchase Price": [put_purchase_price],
}
input_df = pd.DataFrame(input_data)
st.table(input_df)

# Calculate Call and Put values
bs_model = BlackScholes(time_to_maturity, strike, current_price, volatility, interest_rate)
bs_model.run()
call_price, put_price = bs_model.call_price, bs_model.put_price

# Display Call and Put Values in colored tables
col1, col2 = st.columns([1,1], gap="small")

with col1:
    # Using the custom class for CALL value
    st.markdown(f"""
        <div class="metric-container metric-call">
            <div>
                <div class="metric-label">CALL Value</div>
                <div class="metric-value">${call_price:.2f}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    # Using the custom class for PUT value
    st.markdown(f"""
        <div class="metric-container metric-put">
            <div>
                <div class="metric-label">PUT Value</div>
                <div class="metric-value">${put_price:.2f}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("")
st.title("Options Price - Interactive Heatmap")
st.info("Explore how option prices fluctuate with varying 'Spot Prices and Volatility' levels using interactive heatmap parameters, all while maintaining a constant 'Strike Price'.")

# Interactive Sliders and Heatmaps for Call and Put Options
col1, col2 = st.columns([1,1], gap="small")

with col1:
    st.subheader("Call P&L Heatmap")
    heatmap_fig_call, _ = plot_heatmap(bs_model, spot_range, vol_range, strike, call_purchase_price, put_purchase_price)
    st.pyplot(heatmap_fig_call)

with col2:
    st.subheader("Put P&L Heatmap")
    _, heatmap_fig_put = plot_heatmap(bs_model, spot_range, vol_range, strike, call_purchase_price, put_purchase_price)
    st.pyplot(heatmap_fig_put)

# Calculate and display implied volatilities
st.markdown("---")
st.subheader("Implied Volatility")

col1, col2 = st.columns(2)

with col1:
    implied_vol_call = BlackScholes.calculate_implied_volatility(
        market_call_price, 'call', current_price, strike, time_to_maturity, interest_rate
    )
    st.metric("Implied Volatility (Call)", f"{implied_vol_call:.4f}" if implied_vol_call else "N/A")

with col2:
    implied_vol_put = BlackScholes.calculate_implied_volatility(
        market_put_price, 'put', current_price, strike, time_to_maturity, interest_rate
    )
    st.metric("Implied Volatility (Put)", f"{implied_vol_put:.4f}" if implied_vol_put else "N/A")

st.info("Implied volatility is calculated based on the market prices of options. It represents the market's expectation of future volatility.")

# Add a new section for Implied Volatility Surface
st.markdown("---")
st.subheader("Implied Volatility Surface")

# Create a range of strike prices and times to maturity
strike_range = np.linspace(strike * 0.8, strike * 1.2, 10)
time_range = np.linspace(0.1, 2, 10)

# Calculate implied volatility surface for call options
iv_surface = np.zeros((len(strike_range), len(time_range)))

for i, k in enumerate(strike_range):
    for j, t in enumerate(time_range):
        iv = BlackScholes.calculate_implied_volatility(
            market_call_price, 'call', current_price, k, t, interest_rate
        )
        iv_surface[i, j] = iv if iv else np.nan

# Plot the implied volatility surface
fig, ax = plt.subplots(figsize=(10, 8))
X, Y = np.meshgrid(time_range, strike_range)
surf = ax.pcolormesh(X, Y, iv_surface, cmap='viridis', shading='auto')
ax.set_xlabel('Time to Maturity')
ax.set_ylabel('Strike Price')
ax.set_title('Implied Volatility Surface (Call Options)')
fig.colorbar(surf, ax=ax, label='Implied Volatility')

st.pyplot(fig)

st.info("The Implied Volatility Surface shows how implied volatility varies with different strike prices and times to maturity.")
