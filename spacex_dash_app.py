# Import required libraries
import pandas as pd
import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import plotly.express as px

# Read the SpaceX launch data into pandas dataframe
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "spacex_launch_dash.csv")

# Download CSV if not present
if not os.path.exists(csv_path):
    import urllib.request
    url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/spacex_launch_dash.csv"
    print("Downloading spacex_launch_dash.csv ...")
    urllib.request.urlretrieve(url, csv_path)

spacex_df = pd.read_csv(csv_path)
max_payload = spacex_df['Payload Mass (kg)'].max()
min_payload = spacex_df['Payload Mass (kg)'].min()

# Create a dash application
app = dash.Dash(__name__)

# Build the launch site options list for the dropdown
site_options = [{'label': 'All Sites', 'value': 'ALL'}] + [
    {'label': site, 'value': site}
    for site in sorted(spacex_df['Launch Site'].unique())
]

# Create an app layout
app.layout = html.Div(children=[
    html.H1(
        'SpaceX Launch Records Dashboard',
        style={'textAlign': 'center', 'color': '#503D36', 'font-size': 40}
    ),

    # TASK 1: Add a dropdown list to enable Launch Site selection
    dcc.Dropdown(
        id='site-dropdown',
        options=site_options,
        value='ALL',
        placeholder='Select a Launch Site here',
        searchable=True
    ),
    html.Br(),

    # TASK 2: Pie chart for total successful launches count, or success vs failed for a selected site
    html.Div(dcc.Graph(id='success-pie-chart')),
    html.Br(),

    html.P("Payload range (Kg):"),

    # TASK 3: Add a slider to select payload range
    dcc.RangeSlider(
        id='payload-slider',
        min=0,
        max=10000,
        step=1000,
        marks={i: str(i) for i in range(0, 10001, 1000)},
        value=[min_payload, max_payload]
    ),
    html.Br(),

    # TASK 4: Scatter chart for correlation between payload and launch success
    html.Div(dcc.Graph(id='success-payload-scatter-chart')),
])


# TASK 2: Add a callback function for `site-dropdown` as input, `success-pie-chart` as output
@app.callback(
    Output(component_id='success-pie-chart', component_property='figure'),
    Input(component_id='site-dropdown', component_property='value')
)
def get_pie_chart(entered_site):
    if entered_site == 'ALL':
        # Show total successful launches by site
        fig = px.pie(
            spacex_df[spacex_df['class'] == 1],
            names='Launch Site',
            title='Total Successful Launches by Site'
        )
    else:
        # Show success vs failure counts for the selected site
        filtered_df = spacex_df[spacex_df['Launch Site'] == entered_site]
        fig = px.pie(
            filtered_df,
            names='class',
            title='Total Launch Outcomes for site {}'.format(entered_site),
            color='class',
            color_discrete_map={0: 'red', 1: 'green'},
            labels={'class': 'Outcome'},
        )
        fig.update_traces(
            texttemplate='%{label}<br>%{percent}',
        )
    return fig


# TASK 4: Add a callback function for `site-dropdown` and `payload-slider` as inputs,
#         `success-payload-scatter-chart` as output
@app.callback(
    Output(component_id='success-payload-scatter-chart', component_property='figure'),
    [
        Input(component_id='site-dropdown', component_property='value'),
        Input(component_id='payload-slider', component_property='value')
    ]
)
def get_scatter_chart(entered_site, payload_range):
    low, high = payload_range
    mask = (
        (spacex_df['Payload Mass (kg)'] >= low) &
        (spacex_df['Payload Mass (kg)'] <= high)
    )
    filtered_df = spacex_df[mask]

    if entered_site != 'ALL':
        filtered_df = filtered_df[filtered_df['Launch Site'] == entered_site]
        title = 'Correlation Between Payload and Success for {}'.format(entered_site)
    else:
        title = 'Correlation Between Payload and Success for All Sites'

    fig = px.scatter(
        filtered_df,
        x='Payload Mass (kg)',
        y='class',
        color='Booster Version Category',
        title=title,
        labels={'class': 'Launch Outcome (1=Success, 0=Failure)'},
    )
    return fig


# Run the app
if __name__ == '__main__':
    app.run(debug=True)
