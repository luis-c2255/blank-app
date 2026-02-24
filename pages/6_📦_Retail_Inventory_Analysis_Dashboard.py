import streamlit as st  
import pandas as pd  
import numpy as np  
import plotly.express as px  
import plotly.graph_objects as go  
from plotly.subplots import make_subplots  
from sklearn.ensemble import RandomForestRegressor  
from sklearn.preprocessing import LabelEncoder  
from sklearn.model_selection import train_test_split  
import seaborn as sns  
import matplotlib.pyplot as plt  

from utils.theme import Components, Colors, apply_chart_theme, init_page

init_page("Retail Inventory Analysis Dashboard", "📦")

# Load custom CSS
try:
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("Custom CSS file not found. Using default styling.")

@st.cache_data  
def load_data():  
    df = pd.read_csv('retail_store_inventory.csv')  
    df['Date'] = pd.to_datetime(df['Date'])  
  
    # Feature engineering  
    df['Year'] = df['Date'].dt.year  
    df['Month'] = df['Date'].dt.month  
    df['Quarter'] = df['Date'].dt.quarter
    df['DayOfWeek'] = df['Date'].dt.dayofweek  
    df['Month_Name'] = df['Date'].dt.strftime('%B')   
    df['Revenue'] = df['Units Sold'] * df['Price'] * (1 - df['Discount']/100)  
    df['Forecast_Error'] = df['Units Sold'] - df['Demand Forecast']
    df['Forecast_Error_Pct'] = (df['Forecast_Error'] / df['Demand Forecast']) * 100 
    df['Forecast_Accuracy'] = 1 - abs(df['Forecast_Error']) / df['Demand Forecast']  
    df['Stock_to_Sales_Ratio'] = df['Inventory Level'] / (df['Units Sold'] + 1)  
    df['Price_vs_Competitor'] = df['Price'] - df['Competitor Pricing']  
    df['Stockout_Risk'] = (df['Inventory Level'] < df['Demand Forecast']).astype(int)  
    df['Overstock_Risk'] = (df['Inventory Level'] > 2 * df['Demand Forecast']).astype(int)
    df['Stock_Coverage_Days'] = df['Inventory Level'] / df['Units Sold'].replace(0, 1) 

    # Risk categories  
    df['Stockout_Risk'] = np.where(df['Stock_Coverage_Days'] < 3, 'High Risk',  
    np.where(df['Stock_Coverage_Days'] < 7, 'Medium Risk', 'Low Risk'))  
    df['Overstock_Risk'] = np.where(df['Stock_Coverage_Days'] > 30, 'Overstock', 'Normal')  
    df['Price_Position'] = np.where(df['Price'] < df['Competitor Pricing'], 'Below Competition',  
    np.where(df['Price'] > df['Competitor Pricing'], 'Above Competition', 'At Par')) 
  
    return df  
  
df = load_data()  

# Title
st.markdown(
    Components.page_header(
        "📦  Retail Inventory Analysis Dashboard"
    ), unsafe_allow_html=True
)
st.markdown("""
<div style='text-align: center; font-size: 1.8rem'>
    <p>Comprehensive insights into sales, inventory, and demand forecasting</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

st.markdown(
    Components.section_header(
        "Filters", "🔍"
    ), unsafe_allow_html=True
)

# Date range filter
min_date = df['Date'].min()
max_date = df['Date'].max()
date_range = st.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Category filter
categories = st.multiselect(
    "Select Categories",
    options=df['Category'].unique(),
    default=df['Category'].unique()
)

# Region filter
regions = st.multiselect(
    "Select Regions",
    options=df['Region'].unique(),
    default=df['Region'].unique()
)

# Store filter
stores = st.multiselect(
    "Select Stores (Optional)",
    options=sorted(df['Store ID'].unique()),
    default = []
)

# Weather filter
weather = st.multiselect(
    "Weather Conditions",
    options=df['Weather Condition'].unique(),
    default=df['Weather Condition'].unique()
)

# Apply filters
if len(date_range) == 2:
    filtered_df = df[
        (df['Date'] >= pd.to_datetime(date_range[0])) &
        (df['Date'] <= pd.to_datetime(date_range[1])) &
        (df['Category'].isin(categories)) &
        (df['Region'].isin(regions)) &
        (df['Weather Condition'].isin(weather))
    ]
if stores:
    filtered_df = filtered_df[filtered_df['Store ID'].isin(stores)]
else:
    filtered_df = df.copy()


st.markdown(
    Components.section_header("Executive Summary", "📊"),
    unsafe_allow_html=True
)

# KPI Metrics Row
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    total_revenue = filtered_df['Revenue'].sum()  
    st.markdown(
        Components.metric_card(
            title="Total Revenue",
            value=f"${total_revenue:,.0f}",
            delta="💵",
            card_type="success"
        ), unsafe_allow_html=True
    )
with col2:
    total_sales = filtered_df['Units Sold'].sum() 
    st.markdown(
        Components.metric_card(
            title="Total Units Sold",
            value=f"{total_sales:,.0f}",
            delta="📦",
            card_type="success"
        ), unsafe_allow_html=True
    )
with col3:
    total_stores = filtered_df['Store ID'].nunique() 
    st.markdown(
        Components.metric_card(
            title="Active Stores",
            value=f"{total_stores}",
            delta="🏪",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col4:
    avg_inventory = filtered_df['Inventory Level'].mean()
    st.markdown(
        Components.metric_card(
            title="Avg Inventory",
            value=f"{avg_inventory:.0f}",
            delta="📊",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col5:
    avg_forecast_error = abs(filtered_df['Forecast_Error_Pct']).mean() 
    st.markdown(
        Components.metric_card(
            title="Forecast Error",
            value=f"{avg_forecast_error:.1f}%",
            delta="🎯",
            card_type="error"
        ), unsafe_allow_html=True
    )
st.markdown("---") 
st.markdown(
    Components.section_header("Sales Trend & Category Performance", "📈"),
    unsafe_allow_html=True
)

with st.container():
    # Daily sales trend  
    daily_sales = filtered_df.groupby('Date').agg({  
    'Units Sold': 'sum',  
    'Revenue': 'sum'  
    }).reset_index()   
    fig = go.Figure()  
    fig.add_trace(go.Scatter(  
        x=daily_sales['Date'],  
        y=daily_sales['Units Sold'],  
        mode='lines',  
        name='Units Sold',  
        line=dict(color='#1f77b4', width=2),  
        fill='tozeroy'  
    ))  
   
    fig.update_layout(  
        height=400,  
        hovermode='x unified',  
        xaxis_title="Date",  
        yaxis_title="Units Sold",  
        showlegend=True  
    ) 
    st.plotly_chart(fig, width="stretch")  

with st.container():
    # Top categories 
    category_revenue = filtered_df.groupby('Category')['Revenue'].sum().sort_values(ascending=False).head(5)  
    fig2 = px.bar(
        x=category_revenue.values,
        y=category_revenue.index,
        orientation='h',
        color=category_revenue.values,
        color_continuous_scale='Blues',
        labels={'x': 'Revenue ($)', 'y': 'Category'},
        title='Top Categories')
          
    fig2.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig2, width="stretch") 

st.markdown("---") 
st.markdown(
    Components.section_header("Regional Performance & Weather Impact", "🗺️"),
    unsafe_allow_html=True
)

with st.container():
    # Regional performance  
    regional_sales = filtered_df.groupby('Region').agg({  
    'Revenue': 'sum',  
    'Units Sold': 'sum'  
    }).reset_index()  
  
    fig3 = px.pie(
        regional_sales, 
        values='Revenue',
        names='Region',
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Set3,
        title='Regional Performance')            
    fig3.update_traces(textinfo='percent+label', textposition='inside')  
    fig3.update_layout(height=400)
    st.plotly_chart(fig3, width="stretch")  

with st.container():
    # weather impact
    weather_data = filtered_df.groupby('Weather Condition')['Units Sold'].mean().sort_values(ascending=False) 
  
    fig4 = px.bar(
        x=weather_data.index,
        y=weather_data.values,
        color=weather_data.values,
        color_continuous_scale="Viridis",
        labels={'x': 'Weather Condition', 'y': 'Avg Units Sold'},
        title='Weather Impact on Sales') 
    fig4.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig4, width="stretch")  

st.markdown("---") 
st.markdown(
    Components.section_header("Promotional Impact Analysis", "🎉"),
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)
with col1:
    promo_data = filtered_df.groupby('Holiday/Promotion').agg({
        'Units Sold': 'mean',
        'Revenue': 'mean',
        'Discount': 'mean'
    }).reset_index()
    promo_data['Holiday/Promotion'] = promo_data['Holiday/Promotion'].map({0: 'No Promotion', 1: 'With Promotion'})
  
    fig5 = px.bar(
        promo_data,
        x='Holiday/Promotion',
        y='Units Sold',
        color='Holiday/Promotion',
        color_discrete_map={'No Promotion': '#ff7f0e', 'With Promotion': '#2ca02c'},
        title='Average Units Sold')  
    fig5.update_layout(height=300, showlegend=False)
    st.plotly_chart(fig5, width="stretch")

with col2:
    fig6 = px.bar(
        promo_data,
        x='Holiday/Promotion',
        y='Revenue',
        color='Holiday/Promotion',
        color_discrete_map={'No Promotion': '#ff7f0e', 'With Promotion': '#2ca02c'},
        title='Average Revenue')
    fig6.update_layout(height=300, showlegend=False)
    st.plotly_chart(fig6, width="stretch")

with col3:
    fig_bar = px.bar(
        promo_data,
        x='Holiday/Promotion',
        y='Discount',
        color='Holiday/Promotion',
        color_discrete_map={'No Promotion': '#ff7f0e', 'With Promotion': '#2ca02c'},
        title='Average Discount (%)')
    fig_bar.update_layout(height=300, showlegend=False)
    st.plotly_chart(fig_bar, width="stretch")

st.markdown("---") 
st.markdown(
    Components.page_header("📦 Inventory Health & Optimization"),
    unsafe_allow_html=True
)

# KPI Metrics Row
col1, col2, col3, col4 = st.columns(4)

with col1:
    high_risk_count = len(filtered_df[filtered_df['Stockout_Risk'] == 'High Risk'])  
    st.markdown(
        Components.metric_card(
            title="High Stockout Risk",
            value=f"{high_risk_count:,}",
            delta=f"{(high_risk_count/len(filtered_df)*100):.1f}%",
            card_type="warning"
        ), unsafe_allow_html=True
    )
with col2:
    overstock_count = len(filtered_df[filtered_df['Overstock_Risk'] == 'Overstock']) 
    st.markdown(
        Components.metric_card(
            title="Overstock Items",
            value=f"{overstock_count:,}",
            delta=f"{(overstock_count/len(filtered_df)*100):.1f}%",
            card_type="success"
        ), unsafe_allow_html=True
    )
with col3:
    avg_coverage = filtered_df['Stock_Coverage_Days'].mean()
    st.markdown(
        Components.metric_card(
            title="Avg Stock Coverage",
            value=f"{avg_coverage:.1f} days",
            delta="⏱️",
            card_type="warning"
        ), unsafe_allow_html=True
    )
with col4:
    total_inventory_value = (filtered_df['Inventory Level'] * filtered_df['Price']).sum() 
    st.markdown(
        Components.metric_card(
            title="Total Inventory Value",
            value=f"${total_inventory_value:,.0f}",
            delta="💵",
            card_type="success"
        ), unsafe_allow_html=True
    )
st.markdown("---") 
st.markdown(
    Components.section_header("Stockout Risk Analysis", "🚨"),
    unsafe_allow_html=True
)
with st.container():
    stockout_by_store = filtered_df.groupby(['Store ID', 'Stockout_Risk']).size().reset_index(name='Count')  
    stockout_pivot = stockout_by_store.pivot(index='Store ID', columns='Stockout_Risk', values='Count').fillna(0)  
  
    # Top 15 stores with highest risk  
    top_stores = stockout_pivot.nlargest(15, 'High Risk') 
    fig8 = go.Figure()
    fig8.add_trace(go.Bar(name='High Risk', x=top_stores.index, y=top_stores['High Risk'], marker_color='#d62728'))
    fig8.add_trace(go.Bar(name='Medium Risk', x=top_stores.index, y=top_stores['Medium Risk'], marker_color='#ff7f0e'))
    fig8.add_trace(go.Bar(name='Low Risk', x=top_stores.index, y=top_stores['Low Risk'], marker_color='#2ca02c'))

    fig8.update_layout(
        barmode='stack',
        height=400,
        xaxis_title='Store ID',
        yaxis_title='Number of Items',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    st.plotly_chart(fig8, width='stretch')

with st.container():
    risk_dist = filtered_df['Stockout_Risk'].value_counts().reset_index()  
    risk_dist.columns = ['Risk Level', 'Count']  
  
    colors = {'High Risk': '#d62728', 'Medium Risk': '#ff7f0e', 'Low Risk': '#2ca02c'}  
    risk_dist['Color'] = risk_dist['Risk Level'].map(colors) 

    fig_pie = px.pie(
        risk_dist,
        values='Count',
        names='Risk Level',
        color='Risk Level',
        color_discrete_map=colors,
        hole=0.4,
        title="Risk Distribution")
    
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    fig_pie.update_layout(height=400)
    st.plotly_chart(fig_pie, width="stretch")

st.markdown("---") 
st.markdown(
    Components.section_header("Inventory Turnover & Overstock Analysis", "🔄"),
    unsafe_allow_html=True
)

with st.container():
    turnover_data = filtered_df.groupby('Category').agg({
        'Units Sold': 'sum',
        'Inventory Level': 'mean'
    }).reset_index()
    turnover_data['Turnover_Ratio'] = (turnover_data['Units Sold'] / turnover_data['Inventory Level']).round(2)
    turnover_data = turnover_data.sort_values('Turnover_Ratio', ascending=True)

    fig_bar2 = px.bar(
        turnover_data,
        x='Turnover_Ratio',
        y='Category',
        orientation='h',
        color='Turnover_Ratio',
        color_continuous_scale='RdYlGn',
        title="Inventory Turnover by Category",
        labels={'Turnover_Ratio': 'Turnover Ratio'}
    )
    fig_bar2.update_layout(height=400)
    st.plotly_chart(fig_bar2, width="stretch")

with st.container():
    overstock_data = filtered_df[filtered_df['Overstock_Risk'] == 'Overstock'].groupby('Category').agg({
        'Product ID': 'count',
        'Inventory Level': 'mean'
    }).reset_index()
    overstock_data.columns = ['Category', 'Overstock_Count', 'Avg_Inventory']
    overstock_data = overstock_data.sort_values('Overstock_Count', ascending=False)

    fig_bar3 = px.bar(
        overstock_data,
        x='Category',
        y='Overstock_Count',
        color='Avg_Inventory',
        color_continuous_scale='Reds',
        labels={'Overstock_Count': 'Number of Overstock Items'}
    )
    fig_bar3.update_layout(height=400)
    st.plotly_chart(fig_bar3, width="stretch")

st.markdown("---") 
st.markdown(
    Components.section_header("Critical Inventory Items", "📋"),
    unsafe_allow_html=True
)

with st.container():
    stock_ratio = filtered_df.groupby('Category')['Stock_to_Sales_Ratio'].mean().reset_index()
    fig_stock = px.bar(
        stock_ratio,
        x='Category',
        y='Stock_to_Sales_Ratio',
        color='Category',
        color_continuous_scale='RdYlGn_r')
    fig_stock.update_traces(texttemplate='%{text:.2f}', textposition='inside')
    fig_stock.add_hline(y=1.5, line_dash='dash', line_color='green',
    annotation_text='Optimal Min (1.5x)')
    fig_stock.add_hline(y=2.0, line_dash='dash', line_color='red',
    annotation_text='Optimal Max (2.0x)')
    st.plotly_chart(fig_stock, width="stretch", height=600) 

st.markdown("---") 
col1, col2 = st.columns(2)
with col1:
    st.markdown("### 🚨 High Stockout Risk")
    high_risk_items = filtered_df[filtered_df['Stockout_Risk'] == 'High Risk'][
        ['Store ID', 'Product ID', 'Category', 'Inventory Level', 'Units Sold', 'Stock_Coverage_Days']
    ].sort_values('Stock_Coverage_Days').head(20)

    st.dataframe(
        high_risk_items.style.background_gradient(subset=['Stock_Coverage_Days'], cmap='Reds_r'),
        width="stretch", height=400
    )
    csv = high_risk_items.to_csv(index=False)
    st.download_button(
        label="📥 Download High Risk Items",
        data=csv,
        file_name='high_stockout_risk_items.csv',
        mime='text/csv'
    )
with col2:
    st.markdown("### 📦 Overstock Items")
    overstock_items = filtered_df[filtered_df['Overstock_Risk'] == 'Overstock'][
        ['Store ID', 'Product ID', 'Category', 'Inventory Level', 'Units Sold', 'Stock_Coverage_Days']
    ].sort_values('Stock_Coverage_Days', ascending=False).head(20)

    st.dataframe(
        overstock_items.style.background_gradient(subset=['Stock_Coverage_Days'], cmap='Oranges'),
        width="stretch", height=400
    )
    csv = overstock_items.to_csv(index=False)
    st.download_button(
        label="📥 Download Overstock Items",
        data=csv,
        file_name='overstock_items.csv',
        mime='text/csv'
    )

st.markdown("---") 
st.markdown(
    Components.page_header("📈 Sales Analytics"),
    unsafe_allow_html=True
)
col1, col2, col3, col4 = st.columns(4)  
  
total_revenue = filtered_df['Revenue'].sum()  
avg_daily_sales = filtered_df.groupby('Date')['Units Sold'].sum().mean()  
best_category = filtered_df.groupby('Category')['Revenue'].sum().idxmax()  
growth_rate = ((filtered_df.groupby('Date')['Revenue'].sum().iloc[-30:].mean() /  
filtered_df.groupby('Date')['Revenue'].sum().iloc[:30].mean() - 1) * 100)  

with col1:
    st.markdown(
        Components.metric_card(
            title="Total Revenue",
            value=f"${total_revenue:,.0f}",
            delta="💰",
            card_type="success"
        ), unsafe_allow_html=True
    )
with col2:
    st.markdown(
        Components.metric_card(
            title="Avg Daily Sales",
            value=f"{avg_daily_sales:.0f} units",
            delta="📊",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col3:
    st.markdown(
        Components.metric_card(
            title="Top Category",
            value=f"{best_category}",
            delta="🏆",
            card_type="success"
        ), unsafe_allow_html=True
    )
with col4:
    st.markdown(
        Components.metric_card(
            title="Growth Rate",
            value=f"{growth_rate:.1f}%",
            delta=f"{growth_rate:.1f}%",
            card_type="error"
        ), unsafe_allow_html=True
    )

st.markdown("---") 
st.markdown(
    Components.section_header("Sales & Revenue Trends", "📈"),
    unsafe_allow_html=True
)
trend_option = st.radio(
    "Select Aggregation",
    ['Daily', 'Weekly', 'Monthly'],
    horizontal=True
)

if trend_option == 'Daily':
    trend_data = filtered_df.groupby('Date').agg({
        'Units Sold': 'sum',
        'Revenue': 'sum'
    }).reset_index()
    x_col = 'Date'
elif trend_option == 'Weekly':
    filtered_df['Weekly'] = filtered_df['Date'].dt.to_period('W').astype(str)
    trend_data = filtered_df.groupby('Week').agg({
        'Units Sold': 'sum',
        'Revenue': 'sum'
    }).reset_index()
    x_col = 'Week'
else:
    filtered_df['Year_Month'] = filtered_df['Date'].dt.to_period('M').astype(str)
    trend_data = filtered_df.groupby('Year_Month').agg({
        'Units Sold': 'sum',
        'Revenue': 'sum'
    }).reset_index()
    x_col = 'Year_Month'

    fig_trend = make_subplots(specs=[[{'secondary_y': True}]])

    fig_trend.add_trace(
        go.Scatter(
            x=trend_data[x_col],
            y=trend_data['Units Sold'],
            name='Units Sold',
            line=dict(color='#1f77b4', width=2),
            mode='lines+markers'
        ), secondary_y=False
    )
    fig_trend.update_xaxes(title_text='Period')
    fig_trend.update_yaxes(title_text='Units Sold', secondary_y=False)
    fig_trend.update_yaxes(title_text='Revenue ($)', secondary_y=True)
    fig_trend.update_layout(height=400, hovermode='x unified')

    st.plotly_chart(fig_trend, width="stretch")

st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    st.markdown("### 📦 Sales by Category")
    category_sales = filtered_df.groupby('Category').agg({
        'Units Sold': 'sum',
        'Revenue': 'sum'
    }).reset_index().sort_values('Revenue', ascending=False)

    fig_cat = px.bar(
        category_sales,
        x='Category',
        y=['Units Sold', 'Revenue'],
        barmode='group',
        color_discrete_sequence=['#1f77b4', '#2ca02c'],
        labels={'value': 'Amount', 'variable': 'Metric'}
    )
    fig_cat.update_layout(height=400)
    st.plotly_chart(fig_cat, width="stretch")

with col2:
    st.markdown("### 🗺️ Sales by Region")
    regional_sales = filtered_df.groupby('Region').agg({
        'Units Sold': 'sum',
        'Revenue': 'sum'
    }).reset_index()

    fig_reg = px.scatter(
        regional_sales,
        x='Units Sold',
        y='Revenue',
        size='Revenue',
        color='Region',
        text='Region',
        size_max=60,
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_reg.update_traces(textposition='top center')
    fig_reg.update_layout(height=400)
    st.plotly_chart(fig_reg, width="stretch")


st.markdown("---") 
st.markdown(
    Components.section_header("Seasonality & Day of Week Analysis", "🍂"),
    unsafe_allow_html=True
)
with st.container():
    seasonal_data = filtered_df.groupby('Seasonality').agg({  
    'Units Sold': 'sum',  
    'Revenue': 'sum'  
    }).reset_index()  
  
    # Order seasons  
    season_order = ['Spring', 'Summer', 'Autumn', 'Winter']  
    seasonal_data['Seasonality'] = pd.Categorical(  
    seasonal_data['Seasonality'],  
    categories=season_order,  
    ordered=True  
    )  
    seasonal_data = seasonal_data.sort_values('Seasonality') 
    fig_seas = go.Figure()  
    fig_seas.add_trace(go.Bar(  
    x=seasonal_data['Seasonality'],  
    y=seasonal_data['Units Sold'],  
    name='Units Sold',  
    marker_color='#8dd3c7'  
    ))  
    fig_seas.add_trace(go.Bar(  
    x=seasonal_data['Seasonality'],  
    y=seasonal_data['Revenue'] / 100, # Scale for visibility  
    name='Revenue (÷100)',  
    marker_color='#fb8072'  
    ))  
    fig_seas.update_layout(barmode='group', height=400, title="Seasonal Performance")  
    st.plotly_chart(fig_seas, width="stretch") 

with st.container():
    dow_data = filtered_df.groupby('DayOfWeek').agg({  
    'Units Sold': 'mean',  
    'Revenue': 'mean'  
    }).reset_index()  
  
    dow_labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']  
    dow_data['Day'] = dow_data['DayOfWeek'].map(dict(enumerate(dow_labels))) 

    fig_day = px.line_polar(
        dow_data,
        r='Units Sold',
        theta='Day',
        line_close=True,
        color_discrete_sequence=['#1f77b4']
    )
    fig_day.update_traces(fill='toself')
    fig_day.update_layout(height=400, title="Day of the Week Analysis")
    st.plotly_chart(fig_day, width="stretch")

with st.container():
    top_n = st.slider("Select number of products to display", 5, 20, 10)  
  
    top_products = filtered_df.groupby('Product ID').agg({  
    'Units Sold': 'sum',  
    'Revenue': 'sum',  
    'Category': 'first'  
    }).reset_index().sort_values('Revenue', ascending=False).head(top_n)

    fig_prod = px.bar(
        top_products,
        x='Product ID',
        y='Revenue',
        color='Category',
        hover_data=['Units Sold'],
        labels={'Revenue': 'Total Revenue ($)'}
    )
    fig_prod.update_layout(height=400, title="Top Performing Products")
    st.plotly_chart(fig_prod, width="stretch")

st.markdown("---") 
st.markdown(
    Components.page_header("💰 Pricing Strategy"),
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)  
  
avg_discount = filtered_df['Discount'].mean()  
avg_price = filtered_df['Price'].mean()  
price_premium = (filtered_df['Price'] - filtered_df['Competitor Pricing']).mean()  
discount_revenue = filtered_df[filtered_df['Discount'] > 0]['Revenue'].sum()

with col1:
    st.markdown(
        Components.metric_card(
            title="Avg Discount",
            value=f"{avg_discount:.1f}%",
            delta="🏷️",
            card_type="success"
        ), unsafe_allow_html=True
    )
with col2:
    st.markdown(
        Components.metric_card(
            title="Avg Price",
            value=f"${avg_price:.2f}",
            delta="💵",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col3:
    st.markdown(
        Components.metric_card(
            title="Price vs Competitor",
            value=f"${price_premium:.2f}",
            delta="📊",
            card_type="info"
        ), unsafe_allow_html=True
    )
with col4:
    st.markdown(
        Components.metric_card(
            title="Discount Revenue",
            value=f"${discount_revenue:,.0f}",
            delta="💰",
            card_type="success"
        ), unsafe_allow_html=True
    )

with st.container():
    discount_bins = [0, 5, 10, 15, 20, 30]
    filtered_df['Discount_Range'] = pd.cut(filtered_df['Discount'], bins=discount_bins)
    discount_analysis = filtered_df.groupby('Discount_Range').agg({
        'Units Sold':'mean',
        'Revenue': 'mean'
    }).reset_index()
    discount_analysis = discount_analysis.dropna()
    discount_analysis['Discount_Range'] = discount_analysis['Discount_Range'].astype(str)

    fig_disc = px.line(discount_analysis, x='Discount_Range', y=['Units Sold', 'Revenue'],  
    title='Discount Range Impact on Sales & Revenue',  
    markers=True,  
    labels={'value': 'Average Value', 'variable': 'Metric'})  
    st.plotly_chart(fig_disc, width="stretch") 

with st.container():
    filtered_df['Price_Position'] = np.where(  
    filtered_df['Price'] < filtered_df['Competitor Pricing'], 'Below Competitor',  
    np.where(filtered_df['Price'] > filtered_df['Competitor Pricing'], 'Above Competitor', 'Equal')  
    )
    pricing_perf = filtered_df.groupby('Price_Position').agg({  
        'Units Sold': 'mean',  
        'Revenue': 'mean'  
    }).reset_index()
    fig_price = px.bar(pricing_perf, x='Price_Position', y='Units Sold',  
    title='Sales Performance vs Competitor Pricing',  
    text='Units Sold', color='Units Sold')  
    fig_price.update_traces(texttemplate='%{text:.1f}', textposition='outside')  
    st.plotly_chart(fig_price, width="stretch")  

with st.container():
    price_corr = filtered_df.groupby('Category').apply(  
    lambda x: x[['Price', 'Units Sold']].corr().iloc[0, 1]  
    ).reset_index()  
    price_corr.columns = ['Category', 'Price_Elasticity']

    fig_corr = px.bar(price_corr, x='Category', y='Price_Elasticity',  
    title='Price Elasticity by Category (Correlation)',  
    text='Price_Elasticity',  
    color='Price_Elasticity',  
    color_continuous_scale='RdYlGn')  
    fig_corr.update_traces(texttemplate='%{text:.3f}', textposition='outside')  
    fig_corr.add_hline(y=0, line_dash="dash", line_color="gray")  
    st.plotly_chart(fig_corr, width="stretch")  
