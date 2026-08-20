"""
E-commerce Orders — Interactive EDA Dashboard
Run with:  streamlit run app.py
Place `ecommerce_orders_dataset.csv` in the same folder as this file
(or update DATA_PATH below).
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="E-commerce Orders Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = "ecommerce_orders_dataset.csv"

# ----------------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------------
@st.cache_data
def load_data(path):
    df = pd.read_csv(path, parse_dates=["Order_Date"])
    return df

try:
    df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(f"Could not find `{DATA_PATH}`. Upload the CSV or update DATA_PATH in app.py.")
    uploaded = st.file_uploader("Or upload the CSV here", type="csv")
    if uploaded is not None:
        df = pd.read_csv(uploaded, parse_dates=["Order_Date"])
    else:
        st.stop()

# ----------------------------------------------------------------------------
# Sidebar filters
# ----------------------------------------------------------------------------
st.sidebar.header("🔎 Filters")

min_date, max_date = df["Order_Date"].min(), df["Order_Date"].max()
date_range = st.sidebar.date_input(
    "Order Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date
)

countries = st.sidebar.multiselect(
    "Country", sorted(df["Country"].unique()), default=[]
)
categories = st.sidebar.multiselect(
    "Product Category", sorted(df["Product_Category"].unique()), default=[]
)
segments = st.sidebar.multiselect(
    "Customer Segment", sorted(df["Customer_Segment"].unique()), default=[]
)
statuses = st.sidebar.multiselect(
    "Order Status", sorted(df["Order_Status"].unique()), default=[]
)

filtered = df.copy()

if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = date_range
    filtered = filtered[
        (filtered["Order_Date"] >= pd.to_datetime(start))
        & (filtered["Order_Date"] <= pd.to_datetime(end))
    ]
if countries:
    filtered = filtered[filtered["Country"].isin(countries)]
if categories:
    filtered = filtered[filtered["Product_Category"].isin(categories)]
if segments:
    filtered = filtered[filtered["Customer_Segment"].isin(segments)]
if statuses:
    filtered = filtered[filtered["Order_Status"].isin(statuses)]

st.sidebar.markdown(f"**Rows after filtering:** {len(filtered):,} / {len(df):,}")

if filtered.empty:
    st.warning("No data matches the selected filters. Adjust filters in the sidebar.")
    st.stop()

# ----------------------------------------------------------------------------
# Header + KPIs
# ----------------------------------------------------------------------------
st.title("🛒 E-commerce Orders — Exploratory Dashboard")
st.caption("Interactive EDA over the ecommerce orders dataset. Use the sidebar to filter.")

total_revenue = filtered["Order_Amount"].sum()
total_orders = filtered["Order_ID"].nunique()
avg_order_value = filtered["Order_Amount"].mean()
total_profit = filtered["Profit_Amount"].sum()
return_rate = (filtered["Returned"] == "Yes").mean() * 100
avg_rating = filtered["Review_Rating"].mean()

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Total Revenue", f"${total_revenue:,.0f}")
k2.metric("Total Orders", f"{total_orders:,}")
k3.metric("Avg Order Value", f"${avg_order_value:,.2f}")
k4.metric("Total Profit", f"${total_profit:,.0f}")
k5.metric("Return Rate", f"{return_rate:.1f}%")
k6.metric("Avg Review Rating", f"{avg_rating:.2f} ⭐")

st.divider()

# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------
tab_overview, tab_products, tab_customers, tab_geo, tab_ops, tab_data = st.tabs(
    ["📈 Sales Trends", "📦 Products", "👥 Customers", "🌍 Geography",
     "🚚 Operations & Returns", "🗂 Raw Data"]
)

# ---------------- Sales Trends ----------------
with tab_overview:
    st.subheader("Revenue & Orders Over Time")
    granularity = st.radio("Aggregate by", ["Day", "Week", "Month"], horizontal=True, index=2)
    freq_map = {"Day": "D", "Week": "W", "Month": "ME"}
    ts = (
        filtered.set_index("Order_Date")
        .resample(freq_map[granularity])
        .agg(Revenue=("Order_Amount", "sum"), Orders=("Order_ID", "count"))
        .reset_index()
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts["Order_Date"], y=ts["Revenue"], name="Revenue",
                              mode="lines", yaxis="y1", line=dict(color="#2E86AB")))
    fig.add_trace(go.Scatter(x=ts["Order_Date"], y=ts["Orders"], name="Orders",
                              mode="lines", yaxis="y2", line=dict(color="#F18F01")))
    fig.update_layout(
        yaxis=dict(title="Revenue ($)"),
        yaxis2=dict(title="Orders", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow = filtered["Day_Of_Week"].value_counts().reindex(dow_order).reset_index()
        dow.columns = ["Day", "Orders"]
        fig_dow = px.bar(dow, x="Day", y="Orders", title="Orders by Day of Week",
                          color="Orders", color_continuous_scale="Blues")
        st.plotly_chart(fig_dow, use_container_width=True)
    with c2:
        season = filtered["Season"].value_counts().reset_index()
        season.columns = ["Season", "Orders"]
        fig_season = px.pie(season, names="Season", values="Orders", title="Orders by Season", hole=0.4)
        st.plotly_chart(fig_season, use_container_width=True)

# ---------------- Products ----------------
with tab_products:
    st.subheader("Product Category Performance")
    c1, c2 = st.columns(2)
    with c1:
        cat_rev = filtered.groupby("Product_Category")["Order_Amount"].sum().sort_values(ascending=False).reset_index()
        fig_cat = px.bar(cat_rev, x="Order_Amount", y="Product_Category", orientation="h",
                          title="Revenue by Product Category", color="Order_Amount",
                          color_continuous_scale="Tealgrn")
        fig_cat.update_layout(yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(fig_cat, use_container_width=True)
    with c2:
        cat_profit = filtered.groupby("Product_Category")["Profit_Amount"].sum().sort_values(ascending=False).reset_index()
        fig_profit = px.bar(cat_profit, x="Profit_Amount",
                             y="Product_Category", orientation="h",
                             title="Profit by Product Category", color="Profit_Amount",
                             color_continuous_scale="Purples")
        fig_profit.update_layout(yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(fig_profit, use_container_width=True)

    st.subheader("Top Brands & Subcategories")
    c3, c4 = st.columns(2)
    with c3:
        top_brands = filtered.groupby("Brand")["Order_Amount"].sum().sort_values(ascending=False).head(10).reset_index()
        fig_brand = px.bar(top_brands, x="Brand", y="Order_Amount", title="Top 10 Brands by Revenue",
                            color="Order_Amount", color_continuous_scale="Oranges")
        st.plotly_chart(fig_brand, use_container_width=True)
    with c4:
        top_sub = filtered.groupby("Product_Subcategory")["Order_Amount"].sum().sort_values(ascending=False).head(10).reset_index()
        fig_sub = px.bar(top_sub, x="Order_Amount", y="Product_Subcategory", orientation="h",
                          title="Top 10 Subcategories by Revenue", color="Order_Amount",
                          color_continuous_scale="Reds")
        fig_sub.update_layout(yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(fig_sub, use_container_width=True)

    st.subheader("Price vs Quantity")
    fig_scatter = px.scatter(
        filtered.sample(min(3000, len(filtered)), random_state=1),
        x="Unit_Price", y="Quantity", color="Product_Category", opacity=0.6,
        title="Unit Price vs Quantity Purchased (sampled)"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# ---------------- Customers ----------------
with tab_customers:
    st.subheader("Customer Segments & Demographics")
    c1, c2 = st.columns(2)
    with c1:
        seg = filtered.groupby("Customer_Segment")["Order_Amount"].agg(["sum", "mean", "count"]).reset_index()
        seg.columns = ["Segment", "Total Revenue", "Avg Order Value", "Orders"]
        fig_seg = px.bar(seg, x="Segment", y="Total Revenue", title="Revenue by Customer Segment",
                          color="Segment")
        st.plotly_chart(fig_seg, use_container_width=True)
    with c2:
        fig_age = px.histogram(filtered, x="Customer_Age", nbins=30, title="Customer Age Distribution",
                                color_discrete_sequence=["#6A4C93"])
        st.plotly_chart(fig_age, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        gender = filtered["Customer_Gender"].value_counts().reset_index()
        gender.columns = ["Gender", "Count"]
        fig_gender = px.pie(gender, names="Gender", values="Count", title="Customer Gender Split", hole=0.4)
        st.plotly_chart(fig_gender, use_container_width=True)
    with c4:
        fig_clv = px.box(filtered, x="Customer_Segment", y="Customer_Lifetime_Value",
                          title="Customer Lifetime Value by Segment", color="Customer_Segment")
        st.plotly_chart(fig_clv, use_container_width=True)

    st.subheader("Device & Traffic Source")
    c5, c6 = st.columns(2)
    with c5:
        device = filtered["Device_Type"].value_counts().reset_index()
        device.columns = ["Device", "Orders"]
        fig_device = px.bar(device, x="Device", y="Orders", title="Orders by Device Type",
                             color="Orders", color_continuous_scale="Blues")
        st.plotly_chart(fig_device, use_container_width=True)
    with c6:
        traffic = filtered["Traffic_Source"].value_counts().reset_index()
        traffic.columns = ["Source", "Orders"]
        fig_traffic = px.bar(traffic, x="Source", y="Orders", title="Orders by Traffic Source",
                              color="Orders", color_continuous_scale="Greens")
        st.plotly_chart(fig_traffic, use_container_width=True)

# ---------------- Geography ----------------
with tab_geo:
    st.subheader("Revenue by Country")
    country_rev = filtered.groupby("Country")["Order_Amount"].sum().reset_index()

    # Plotly's choropleth is most reliable with ISO-3 country codes.
    COUNTRY_ISO3 = {
        "Germany": "DEU",
        "France": "FRA",
        "India": "IND",
        "United States": "USA",
        "Australia": "AUS",
        "Canada": "CAN",
        "Saudi Arabia": "SAU",
        "United Kingdom": "GBR",
        "Pakistan": "PAK",
        "UAE": "ARE",
    }
    country_rev["ISO3"] = country_rev["Country"].map(COUNTRY_ISO3)

    fig_map = px.choropleth(country_rev, locations="ISO3", locationmode="ISO-3",
                             color="Order_Amount", title="Revenue by Country",
                             color_continuous_scale="Viridis",
                             hover_name="Country")
    st.plotly_chart(fig_map, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig_country_bar = px.bar(country_rev.sort_values("Order_Amount", ascending=False),
                                  x="Country", y="Order_Amount", title="Revenue by Country (Bar)",
                                  color="Order_Amount", color_continuous_scale="Viridis")
        st.plotly_chart(fig_country_bar, use_container_width=True)
    with c2:
        city_rev = filtered.groupby("City")["Order_Amount"].sum().sort_values(ascending=False).reset_index()
        fig_city = px.bar(city_rev, x="City", y="Order_Amount", title="Revenue by City",
                           color="Order_Amount", color_continuous_scale="Cividis")
        st.plotly_chart(fig_city, use_container_width=True)

    st.subheader("Warehouse Region Performance")
    region = filtered.groupby("Warehouse_Region").agg(
        Orders=("Order_ID", "count"), AvgDeliveryDays=("Delivery_Days", "mean")
    ).reset_index()
    fig_region = px.bar(region, x="Warehouse_Region", y="Orders", title="Orders by Warehouse Region",
                         color="AvgDeliveryDays", color_continuous_scale="RdYlGn_r",
                         labels={"AvgDeliveryDays": "Avg Delivery Days"})
    st.plotly_chart(fig_region, use_container_width=True)

# ---------------- Operations & Returns ----------------
with tab_ops:
    st.subheader("Order Status & Returns")
    c1, c2 = st.columns(2)
    with c1:
        status = filtered["Order_Status"].value_counts().reset_index()
        status.columns = ["Status", "Count"]
        fig_status = px.pie(status, names="Status", values="Count", title="Order Status Distribution", hole=0.4)
        st.plotly_chart(fig_status, use_container_width=True)
    with c2:
        ret_by_cat = (
            filtered.groupby("Product_Category")["Returned"]
            .apply(lambda x: (x == "Yes").mean() * 100)
            .sort_values(ascending=False)
            .reset_index()
        )
        ret_by_cat.columns = ["Category", "Return Rate (%)"]
        fig_ret = px.bar(ret_by_cat, x="Return Rate (%)", y="Category", orientation="h",
                          title="Return Rate % by Product Category", color="Return Rate (%)",
                          color_continuous_scale="Reds")
        fig_ret.update_layout(yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(fig_ret, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig_delivery = px.histogram(filtered, x="Delivery_Days", nbins=13, title="Delivery Days Distribution",
                                     color_discrete_sequence=["#2A9D8F"])
        st.plotly_chart(fig_delivery, use_container_width=True)
    with c4:
        payment = filtered.groupby("Payment_Method")["Order_ID"].count().sort_values(ascending=False).reset_index()
        payment.columns = ["Payment Method", "Orders"]
        fig_pay = px.bar(payment, x="Payment Method", y="Orders", title="Orders by Payment Method",
                          color="Orders", color_continuous_scale="Purples")
        st.plotly_chart(fig_pay, use_container_width=True)

    st.subheader("Correlation Heatmap (Numeric Features)")
    corr_cols = ["Unit_Price", "Quantity", "Discount_Percent", "Discount_Amount",
                 "Shipping_Cost", "Tax_Amount", "Order_Amount", "Delivery_Days",
                 "Review_Rating", "Customer_Lifetime_Value", "Profit_Margin_Percent", "Profit_Amount"]
    corr = filtered[corr_cols].corr()
    fig_corr = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r",
                          title="Correlation Heatmap")
    st.plotly_chart(fig_corr, use_container_width=True)

# ---------------- Raw Data ----------------
with tab_data:
    st.subheader("Filtered Raw Data")
    st.dataframe(filtered, use_container_width=True, height=500)
    st.download_button(
        "⬇️ Download filtered data as CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_orders.csv",
        mime="text/csv",
    )

    st.subheader("Summary Statistics")
    st.dataframe(filtered.describe().T, use_container_width=True)

st.divider()
st.caption("Built with Streamlit + Plotly · Data: ecommerce_orders_dataset.csv")
