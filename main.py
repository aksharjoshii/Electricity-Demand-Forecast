# Import Required liibraries 
import streamlit as st 
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objects as go
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from statsmodels.tsa.seasonal import  seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.graphics.tsaplots import month_plot, quarter_plot
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import pmdarima as pm



# Define a custom colors  using a hex value
h1_color = '#008b8b'
h2_color = '#44d7a8'
h3_color = '#e2062c'
h5_col = '#0892d0'

# title 
st.markdown(f"""
    <h1 style='color: {h1_color}; text-decoration: underline; font-weight: bold;'>Electricity Demand Forecast</h1>
""", unsafe_allow_html=True)

st.write('The data of monthly electricity consumption available starting from January 1973 to December 2019')
# read electricty consumption data 
data = pd.read_csv('data/Electricity Consumption.csv')
# rename columns for consistency 
data.columns = ['date', 'electricity_consumption']
# change to date time format
data['date'] = pd.to_datetime(data['date'])
# display dataframe with expander 
with st.expander("See full Time-Series data"):
    df = data.copy()
    df['date'] = df['date'].dt.date
    st.write(df)
    
with st.form("selection-form"): 
    col1, col2 = st.columns(2)    
    
    with col1:
        
        st.write("Choose a starting period")
        start_quarter = st.selectbox("Quarter",  options=['Q1', 'Q2', 'Q3', 'Q4'], index=2, key='start_q')
        start_year = st.slider("Year", min_value=1973, max_value=2019, value=1973, step=1, key='start_y')
        
    with col2:
        
        st.write("Choose a Ending period")
        end_quarter = st.selectbox("Quarter",  options=['Q1', 'Q2', 'Q3', 'Q4'], index=0, key='end_q')
        end_year = st.slider("Year", min_value=1973, max_value=2019, value=2019, step=1, key='end_y')
           
    submit_button = st.form_submit_button('Analyse', type='primary')

# Start from the beginning of the quarter for start_date
start_date = pd.to_datetime(f"{start_year}-{(int(start_quarter[1])-1)*3 + 1:02d}-01")

# Finish at the end of the quarter for end_date
end_date = pd.to_datetime(f"{end_year}-{(int(end_quarter[1])-1)*3 + 3:02d}-01") + pd.offsets.MonthEnd(0)

# Filter data based on user input
filtered_data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]

# display analysis dashboard
def analysis_dashboard(start_date, end_date, target):
    if start_date > end_date:
        st.error("Start date should come before end date.")
    else:
        # Convert timestamps to strings for display
        start_date_str = start_date.strftime('%Y-%m')
        end_date_str = end_date.strftime('%Y-%m')     
        color_hex_code = '#ff0038'
        st.markdown(f"""
            <h3 style='color: {h3_color};'>Electricity Consumption Trend from {start_date_str} to {end_date_str}</h4>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1,3])

        with col1:
            initial = filtered_data.loc[filtered_data['date'] == filtered_data['date'].min(), target].item()
            final = filtered_data.loc[filtered_data['date'] == filtered_data['date'].max(), target].item()
            perc_diff = round((final - initial) / initial * 100, 2)
            delta = f"{perc_diff} %"
            st.metric(start_date_str, value=initial)
            st.metric(end_date_str, value=final, delta=delta)

        with col2:
            fig = px.line(filtered_data, 
                          x='date',
                          y='electricity_consumption',
                          height=600, width=1000)
            # Customize x-axis and y-axis labels
            fig.update_xaxes(title_text='Date')
            fig.update_yaxes(title_text='Electricity Consumption (Trillion Watts)')      
            st.plotly_chart(fig)

# Assuming start_date, end_date, and filtered_data are defined
analysis_dashboard(start_date, end_date, target='electricity_consumption')

######## Time series analysis###############


# Use Markdown with inline CSS for styling
st.markdown(f"""
    <h2 style='color: {h2_color}; text-decoration: underline; font-weight: bold'>Time Series Analysis</h2>
""", unsafe_allow_html=True)


#1. yearly, quarterly, monthly trend in Electricity consumptions
section_color = '#f2003c'
st.markdown(f"""
    <h3 style='color: {h3_color};'>Electricity Consumption Trends</h3>
""", unsafe_allow_html=True)

def display_trends(df):
    df_ts = data.copy()
    df_ts['year'] = df_ts['date'].dt.year
    df_ts.set_index('date', inplace=True)

    tab1, tab2, tab3 = st.tabs(['Yearly Trend', 'Quarterly Trends', 'Monthly Trends'])
    
    with tab1:
        st.markdown(f"""
                <h5 style='color: {h5_col};'>Yearly Trends of Electricity consumption(in trillion Watts)</h5>
            """, unsafe_allow_html=True)
        yearly_mean = df_ts.groupby(['year'])['electricity_consumption'].mean()
        # Plot the yearly trend using Seaborn and Matplotlib
        # Create a Plotly figure
        fig = px.line(x=yearly_mean.index, 
                      y=yearly_mean.values, 
                      labels={'x': 'Year', 'y': 'Average Electricity Consumption (TW)'},
                      height=700, width=1200)
        # Display in  streamlit
        st.plotly_chart(fig)
    
    with tab2:
        st.markdown(f"""
                <h5 style='color: {h5_col};'>Quarterly Electricity Consumption Trends</h5>
            """, unsafe_allow_html=True)
        # quarterly electricity trends 
        plt.rcParams['figure.figsize'] = (20,10)
        fig = quarter_plot(df_ts['electricity_consumption'].resample(rule='Q').mean())
        st.pyplot(fig)
    
    with tab3:
        st.markdown(f"""
                <h5 style='color: {h5_col};'>Monthly Electricity Consumption Trends</h5>
            """, unsafe_allow_html=True)
        # monthly electricity trends 
        plt.rcParams['figure.figsize'] = (20,10)
        fig = month_plot(df_ts['electricity_consumption'])
        st.pyplot(fig)

display_trends(df=data)



# 2. seasonal decomposition 
st.markdown(f"""
    <h3 style='color: {h3_color};'>Seasonal Decomposition</h3>
""", unsafe_allow_html=True)
decomp_type = st.selectbox('Select type of decomposition', options=["Additive", 'Multiplicative'], key='decomp_type')

def plot_decomposition(df, model):
    
    df_ts = df.copy()
    df_ts.set_index('date', inplace=True)
    model_lower = model.lower() # Ensure lowercase for compatibility with seasonal_decompose
    # decompose the the time series 
    decomposition = seasonal_decompose(df_ts['electricity_consumption'], model=model_lower, period=12)
    # Create subplots
    fig = sp.make_subplots(rows=3, cols=1, subplot_titles=("Trend Component", "Seasonality Component", "Residuals"))
    # Plot trend
    trace_trend = go.Scatter(x=df_ts.index, y=decomposition.trend, mode='lines', name='Trend')
    fig.add_trace(trace_trend, row=1, col=1)
    # Plot seasonality
    trace_seasonal = go.Scatter(x=df_ts.index, y=decomposition.seasonal, mode='lines', name='Seasonality')
    fig.add_trace(trace_seasonal, row=2, col=1)
    # Plot residuals
    trace_residuals = go.Scatter(x=df_ts.index, y=decomposition.resid, mode='markers', name='Residuals')
    fig.add_trace(trace_residuals, row=3, col=1)
    # Update layout
    fig.update_layout(title_text= f"{model} Seasonal Decomposition", 
                      title_font=dict(size=18, color='#0892d0'), 
                      showlegend=False,
                      height =700, width =1200)
    # Show the Plotly figure in Streamlit
    st.plotly_chart(fig)

plot_decomposition(df=data, model=decomp_type)


# forecasting using  ETS Model  / SARIMA Model

color_hex_code = '#44d7a8'
st.markdown(f"""
    <h2 style='color: {h2_color}; text-decoration: underline; font-weight: bold'>Electricity Demand Forecasting</h2>
""", unsafe_allow_html=True)

forecasting_model = st.selectbox("Select Time-Series Forecasting Model", options=['ETS Model', 'SARIMA'], key='model')

# prepare data for model training and validation 

# drop redundant columns 
#df_ts = data.drop('year', axis=1)
# set  frequqncy of index as 'Month Start'
df_ts= data.copy()

df_ts['date'] = df_ts['date'].dt.strftime('%Y-%m-%d')
df_ts.set_index('date', inplace=True)
df_ts.index.freq = 'MS'
df_ts.dropna(inplace=True)

# last 24 months as set for validation
Ntest = 24
train_size = len(df_ts) - Ntest
train_data, test_data = df_ts.iloc[:train_size], df_ts.iloc[train_size:]
    
# calculate error metrics
def error_metrics(test, pred):
   rmse = round(np.sqrt(mean_squared_error(test, pred)), 3)
   rmspe = round(np.sqrt(np.mean(((test - pred) / test)**2)) * 100, 3)
   mape = round(np.mean(np.abs((test - pred) / test)) * 100, 3)
   return {'RMSE': rmse, 'RMSPE': rmspe, 'MAPE': mape}


# function for ETS Model 
def ets_forcast_model(df=df_ts, train=train_data, test=test_data):

    # Instantiate ETS model with multiplicative trend and seasonality
    ets_model = ExponentialSmoothing(train_data,
                                    trend='multiplicative', seasonal='multiplicative', seasonal_periods=12, 
                                    initialization_method='legacy-heuristic'
                                    )
    # fit the ETS model to the training data
    ets_fit = ets_model.fit()
    # Generate forecast for the test set
    ets_predictions = ets_fit.forecast(steps=len(test_data))

    # test and predicted values
    test_values = test_data.values.ravel()
    ets_preds = ets_predictions.values.ravel()
    # calculate error 
    ets_errors = error_metrics(test_values, ets_preds)
    
    # forecasting  demand for next 24 months using ETS model 
    #  define forecast period
    forecast_periods = 24
    # Instantiate ETS model with multiplicative trend and seasonality
    ets_model = ExponentialSmoothing(df_ts,
                                    trend='multiplicative', seasonal='multiplicative', seasonal_periods=12, 
                                    initialization_method='legacy-heuristic'
                                    )
    # fit the ETS model to the training data                                 
    ets_fit = ets_model.fit()
    # Generate forecast for  forcast period 
    ets_forecast = ets_fit.forecast(steps=forecast_periods)
    
    # create a dataframe to store and display the forecast 
    df_forcast = pd.DataFrame(ets_forecast, columns=['electricity_demand_forecast']) 
    #combined dataframe for visualisation 
    df_ets = df_forcast.copy()
    df_ets.columns = ['electricity_consumption']
    df_combined = pd.concat([df_ts, df_ets])
        
    # create tabs to display results
    tab1, tab2 = st.tabs(['Forecast Table and Error Metrics', 'Forecast Visualization'])
    

    with tab1:
        col1, col2 = st.columns([1,3])
        
        with col1:
            
            st.markdown(f"""
                <h5 style='color: {h5_col};'>Error Metrics</h5>
            """, unsafe_allow_html=True)
            # display error metrics 
            st.metric('RMSE', value=ets_errors['RMSE'])
            st.metric('RMSPE', value=ets_errors['RMSPE'])
            st.metric('MAPE', value=ets_errors['MAPE'])
        
        with col2:
            
            st.markdown(f"""
                <h5 style='color: {h5_col};'>Forecasted Values in Trillion Watts</h5>
            """, unsafe_allow_html=True)
            df_forcast.index = df_forcast.index.date
            st.dataframe(df_forcast)
    
    with tab2:
        
        st.markdown(f"""
            <h5 style='color: {h5_col};'>ETS Model: Electricity Demand Forecast</h5>
        """, unsafe_allow_html=True)
        # Plot the actual vs. predicted values using Plotly
        fig = go.Figure()
        # Add actual time-series data
        fig.add_trace(go.Scatter(x=df_combined.index, y=df_combined.electricity_consumption, mode='lines', name='Time-Series Data', line=dict(color='blue', width=0.8)))
        # Add forecasted data
        fig.add_trace(go.Scatter(x=df_ets.index, y=df_ets.electricity_consumption, mode='lines', name='Forecast', line=dict(color='green', width=2.5, dash='dash')))
        # Update layout
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Electricity Consumption (Trillion Watts)',
            legend=dict(x=0, y=1, traceorder='normal', font=dict(family='sans-serif', size=12), bgcolor='rgba(255, 255, 255, 0.5)'),
            showlegend=True,
            height =700, width =1200
        )
        # Show the Plotly figure in Streamlit
        st.plotly_chart(fig)

# read saved best  hyperparameters for model 
best_params = pd.read_csv('data/best_params_0123.csv')
def extract_best_params(best_params=best_params):
    # transpose dataframe for convinience
    best_params = best_params.T
    # name the column
    best_params.columns =['values']
        # extract individual best parameters 
    p = best_params.loc['p']['values']
    d = best_params.loc['d']['values']
    q = best_params.loc['q']['values']
    P = best_params.loc['P']['values']
    D = best_params.loc['D']['values']
    Q = best_params.loc['Q']['values']
    
    return p, d, q, P, D, Q

# p, d, q, P, D, Q = extract_best_params()

@st.cache_resource(show_spinner='Loading Model....')
def sarima_tuned_model():
    # extract best parameters 
    p, d, q, P, D, Q = extract_best_params()
    # Instantiate a SARIMA model
    sarima = pm.ARIMA(
        order=(p, d, q), 
        seasonal_order=(P, D, Q, 12), 
        suppress_warnings=True,  
        force_stationarity=False  
    )
    return sarima

@st.cache_data(show_spinner='Evaluating the Model....')
def evaluation(train=train_data, test= test_data):
    
    Ntest = len(test)
    model = sarima_tuned_model().fit(train)
    test_preds = model.predict(n_periods=Ntest)
    sarima_errors= error_metrics(test.values.ravel(), test_preds.values)
    
    return sarima_errors

@st.cache_data(show_spinner='Forecasting Values...')
def sarima_forecast(timeseries, forecast_periods=24):
    
    # fit the model on entire time series data
    sarima_model = sarima_tuned_model().fit(timeseries)
    # Generate point forecasts and prediction intervals
    forecast , conf_int = sarima_model.predict(n_periods=forecast_periods, return_conf_int=True)
    # create dataframe of forecast values 
    forecast = pd.DataFrame(forecast)
    # rename the column of the forecast DataFrame to 'electricity_demand_forecast'
    forecast.columns = ['electricity_demand_forecast']
    # confidence interval for forcasting 
    forecast['lower_lim'] = pd.Series(conf_int[:, 0], index=forecast.index)
    forecast['upper_lim'] = pd.Series(conf_int[:, 1], index=forecast.index)
    
    return forecast
    
def plot_forcast(forecast_data, timeseries=df_ts):
    
    # Create a figure
    fig = go.Figure()
    # Plot the actual time series data
    fig.add_trace(go.Scatter(x=timeseries.index, y=timeseries['electricity_consumption'], mode='lines', name='Actual', line=dict(color='blue', width=0.8)))
    # Plot the forecasted values
    fig.add_trace(go.Scatter(x=forecast_data.index, y=forecast_data['electricity_demand_forecast'], mode='lines', name='Forecast', line=dict(color='green', dash='dash', width=2)))
    # Plot the confidence interval
    fig.add_trace(go.Scatter(x=forecast_data.index, y=forecast_data['lower_lim'], fill='tonexty', mode='lines', line=dict(color='orange', width=0), fillcolor='rgba(255, 165, 0, 0.2)', name='Confidence Interval'))
    fig.add_trace(go.Scatter(x=forecast_data.index, y=forecast_data['upper_lim'], fill='tonexty', mode='lines', line=dict(color='orange', width=0), fillcolor='rgba(255, 165, 0, 0.2)', showlegend=False))

    # Customize the layout
    fig.update_layout(
        xaxis=dict(title='Date', showgrid=True),
        yaxis=dict(title='Electricity Consumption(Trillion Watts)', showgrid=True),
        legend=dict(x=0, y=1, traceorder='normal', orientation='h'),
        margin=dict(l=0, r=0, t=40, b=0),
        height =700, width =1200
    )
    
    # Show the Plotly figure 
    st.plotly_chart(fig)
    
# sarima model 
def sarima_forecast_model(data=df_ts, train=train_data, test=test_data): 
    # create tabs to display results
    tab1, tab2 = st.tabs(['Error Metrics and Forecast Table', 'Forecast Visualization'])

    with tab1:
        col1, col2 = st.columns([1,3])       
        with col1:
            
            st.markdown(f"""
                <h5 style='color: {h5_col};'>Evaluation Metrics</h5>
            """, unsafe_allow_html=True)
            # display error metrics 
            sarima_errors = evaluation()
            st.metric('RMSE', value=sarima_errors['RMSE'])
            st.metric('RMSPE', value=sarima_errors['RMSPE'])
            st.metric('MAPE', value=sarima_errors['MAPE'])
        
        with col2:
            st.markdown(f"""
                <h5 style='color: {h5_col};'>Forecasted Values in Trillion Watts</h5>
            """, unsafe_allow_html=True)
            forecast_df = sarima_forecast(data)
            forecast_df.index  = forecast_df.index.date
            st.dataframe(forecast_df)
    
    with tab2:
        
        st.markdown(f"""
            <h5 style='color: {h5_col};'>SARIMA : Electricity Demand Forecast</h5>
        """, unsafe_allow_html=True)
        
        plot_forcast(forecast_data=forecast_df)


# disply the forecast results 
if forecasting_model =='SARIMA':
    sarima_forecast_model()
else :
    ets_forcast_model()
    
    

    
    