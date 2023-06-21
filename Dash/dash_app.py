import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
# from util import db
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.ticker as mtick
from dash import Dash, dcc, html, Input, Output
import dash_auth
import logging
import pymongo
logging.basicConfig(filename='error.log', level=logging.DEBUG)


img_src = 'assets/wordcloud.png'
# Keep this out of source code repository - save in a file or a database
VALID_USERNAME_PASSWORD_PAIRS = {
    'hello': 'world'
}

database_name = "TimeSeries_prod"

# Create a connection string using the format mongodb://username:password@hostname:port/database_name
connection_string = f"mongodb+srv://Chashivmad:GS5iR2Heom7qmz9q@timeseries.zx4n7pp.mongodb.net/?retryWrites=true&w=majority"

# Connect to MongoDB using the connection string
client = pymongo.MongoClient(connection_string)

# Access a specific database and collection
db = client[database_name]
FRED_INDICATORS = ['GDP', 'GDPC1', 'GDPPOT', 'NYGDPMKTPCDWLD',         # 1. Growth
                   'CPIAUCSL', 'CPILFESL', 'GDPDEF',                   # 2. Prices and Inflation
                   'M1SL', 'WM1NS', 'WM2NS', 'M1V', 'M2V', 'WALCL',    # 3. Money Supply
                   'UNRATE', 'NROU', 'CIVPART', 'EMRATIO',             # 4. Employment
                   'UNEMPLOY', 'PAYEMS', 'MANEMP', 'ICSA', 'IC4WSA',   # 4. Employment
                   'CDSP', 'MDSP', 'FODSP', 'DSPIC96', 'PCE', 'PCEDG', # 5. Income and Expenditure
                   'PSAVERT', 'DSPI', 'RSXFS',                         # 5. Income and Expenditure
                   'GFDEBTN', 'GFDEGDQ188S', 'VDE.US','VHT.US'                          # 6. Gov-t debt
                   ]

# ETF indexes
ETF_INDICATORS = ['VDE.US','VHT.US']

app = dash.Dash(__name__)
auth = dash_auth.BasicAuth(
    app,
    VALID_USERNAME_PASSWORD_PAIRS
)
server = app.server

# Define app layout with a navigation bar and empty page content
app.layout = html.Div([
    
    dcc.Location(id='url', refresh=False),
    html.Div([
        html.H1('Welcome to the app'),
        html.H3('You are successfully authorized'),
        dcc.Link('Monitoring', href='/monitoring', className='nav-link'),
        dcc.Link('Prediction', href='/prediction', className='nav-link')
    ], className='nav'),
    html.Div([
        html.Div(id='page-content', className='content')
    ], className='hello')
], style={ 'margin': 'auto', 'textAlign': 'center'})





    

# Define the callbacks to update the page content based on the URL
@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/monitoring':
        return html.Div(
            children=[html.H1('Analysis of Relationship between Macroeconomic Indicators & ETFs'),
            html.H2('Monitoring page 📈'),
            dcc.Dropdown(FRED_INDICATORS, 'GDP', id='demo-dropdown'),
             html.Div(id='dd-output-container'),
            dcc.Graph(id='my-graph')
            ]
        )

    
    elif pathname == '/prediction':
        return html.Div(
            children=[html.H1('Analysis of Relationship between Macroeconomic Indicators & ETFs'),
            html.H2('Prediction page 🤖'),
            dcc.Dropdown(ETF_INDICATORS, 'VHT.US', id='demo-dropdown_ETF'),
            html.Div(id='dd-output-container'),
            dcc.Graph(id='my-graph_ETF')]
        )
    else:
        return html.Div(
            children=[
            html.H1('Analysis of Relationship between Macroeconomic Indicators & ETFs'),
            html.Img(src=img_src)
            ]
        )

@app.callback(Output('my-graph', 'figure'),
              Input('demo-dropdown', 'value'))
def update_graph(selected_option):
    collection = db[selected_option]
    data = pd.DataFrame(list(collection.find()))
    print(data)
    # filtered_data = data[data['<column-containing-options>'] == selected_option]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['DATE'],
                         y=data[selected_option],
                         mode='lines'))

    fig.update_layout(xaxis_title='Date',
                  yaxis_title='Price')

    return fig

@app.callback(Output('my-graph_ETF', 'figure'),
              Input('demo-dropdown_ETF', 'value'))
def update_graph(selected_option):
#     collection = db[selected_option]
#     data = pd.DataFrame(list(collection.find()))
    # print(selected_option)
    # filtered_data = data[data['<column-containing-options>'] == selected_option]
    if selected_option == 'VDE.US':
        
        collection = db["VDE.US_PRED"]
        df_vde = pd.DataFrame(list(collection.find({},{"_id":0})))
        pd.to_datetime(df_vde['index'])
        df_vde = df_vde.set_index('index')
        print(df_vde)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_vde.index, y=df_vde['Actual'], name='Actual'))
        fig.add_trace(go.Scatter(x=df_vde.index, y=df_vde['Predicted'], name='Predicted'))


    else:
        
        collection = db["VHT.US_PRED"]
        df_vht = pd.DataFrame(list(collection.find({},{"_id":0})))
        pd.to_datetime(df_vht['index'])
        df_vht = df_vht.set_index('index')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_vht.index, y=df_vht['Actual'], name='Actual'))
        fig.add_trace(go.Scatter(x=df_vht.index, y=df_vht['Predicted'], name='Predicted'))
    
    fig.update_layout(xaxis_title='Date',
                    yaxis_title='value')

    return fig
# , style={'textAlign': 'center'}
if __name__ == '__main__':
    try:
        app.run_server()
    except Exception as e:
        print(e)
        logging.exception(e)




