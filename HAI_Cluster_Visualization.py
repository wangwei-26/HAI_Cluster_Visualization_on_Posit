from dash import Dash, html, dash_table, dcc, callback, Output, Input, State
import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
import glob
import base64

# Months to monitor
month_list = ['2023-11','2023-12','2024-01', '2024-02', '2024-03', '2024-04', '2024-05', '2024-06', '2024-07', '2024-08',
              '2024-09','2024-10','2024-11','2024-12','2025-01','2025-02']###########################################################
month_list2 = ['11.2023','12.2023','1.2024','2.2024','3.2024','4.2024','5.2024','6.2024','7.2024','8.2024','9.2024',
               '10.2024','11.2024','12.2024','1.2025','2.2025','3.2025']####################################################################

# Dictionary of state abbreviations
state_abbreviations = {
        "Alaska": "AK",
        "Alabama": "AL",
        "Arkansas": "AR",
        "Arizona": "AZ",
        "California": "CA",
        "Colorado": "CO",
        "Connecticut": "CT",
        "District of Columbia": "DC",
        "Delaware": "DE",
        "Florida": "FL",
        "Georgia": "GA",
        "Guam": "GU",
        "Hawaii": "HI",
        "Houston": "TX", ###
        "Iowa": "IA",
        "Idaho": "ID",
        "Illinois": "IL",
        "Indiana": "IN",
        "Kansas": "KS",
        "Kentucky": "KY",
        "Louisiana": "LA",
        "LA County": "CA", ###
        "Massachusetts": "MA",
        "Maryland": "MD",
        "Maine": "ME",
        "Michigan": "MI",
        "Minnesota": "MN",
        "Missouri": "MO",
        "Mississippi": "MS",
        "Montana": "MT",
        "North Carolina": "NC",
        "North Dakota": "ND",
        "Nebraska": "NE",
        "New Hampshire": "NH",
        "New Jersey": "NJ",
        "New Mexico": "NM",
        "Nevada": "NV",
        "New York": "NY",
        "New York City": "NY",  ###
        "Ohio": "OH",
        "Oklahoma": "OK",
        "Oregon": "OR",
        "Pennsylvania": "PA",
        "Philadelphia": "PA",
        "Puerto Rico": "PR",
        "Rhode Island": "RI",
        "South Carolina": "SC",
        "South Dakota": "SD",
        "Tennessee": "TN",
        "Texas": "TX",
        "Utah": "UT",
        "Virginia": "VA",
        "Vermont": "VT",
        "Washington": "WA",
        "Wisconsin": "WI",
        "West Virginia": "WV",
        "Wyoming": "WY"
    }

state_coords = {
    "Wisconsin": {"lat": 44.500000, "lon": -89.500000},
    "West Virginia": {"lat": 39.000000, "lon": -80.500000},
    "Vermont": {"lat": 44.000000, "lon": -72.699997},
    "Texas": {"lat": 31.000000, "lon": -100.000000},
    "South Dakota": {"lat": 44.500000, "lon": -100.000000},
    "Rhode Island": {"lat": 41.742325, "lon": -71.742332},
    "Oregon": {"lat": 44.000000, "lon": -120.500000},
    "New York": {"lat": 43.000000, "lon": -75.000000},
    "New Hampshire": {"lat": 44.000000, "lon": -71.500000},
    "Nebraska": {"lat": 41.500000, "lon": -100.000000},
    "Kansas": {"lat": 38.500000, "lon": -98.000000},
    "Mississippi": {"lat": 33.000000, "lon": -90.000000},
    "Illinois": {"lat": 40.000000, "lon": -89.000000},
    "Delaware": {"lat": 39.000000, "lon": -75.500000},
    "Connecticut": {"lat": 41.599998, "lon": -72.699997},
    "Arkansas": {"lat": 34.799999, "lon": -92.199997},
    "Indiana": {"lat": 40.273502, "lon": -86.126976},
    "Missouri": {"lat": 38.573936, "lon": -92.603760},
    "Florida": {"lat": 27.994402, "lon": -81.760254},
    "Nevada": {"lat": 39.876019, "lon": -117.224121},
    "Maine": {"lat": 45.367584, "lon": -68.972168},
    "Michigan": {"lat": 44.182205, "lon": -84.506836},
    "Georgia": {"lat": 33.247875, "lon": -83.441162},
    "Hawaii": {"lat": 19.741755, "lon": -155.844437},
    "Alaska": {"lat": 66.160507, "lon": -153.369141},
    "Tennessee": {"lat": 35.860119, "lon": -86.660156},
    "Virginia": {"lat": 37.926868, "lon": -78.024902},
    "New Jersey": {"lat": 39.833851, "lon": -74.871826},
    "Kentucky": {"lat": 37.839333, "lon": -84.270020},
    "North Dakota": {"lat": 47.650589, "lon": -100.437012},
    "Minnesota": {"lat": 46.392410, "lon": -94.636230},
    "Oklahoma": {"lat": 36.084621, "lon": -96.921387},
    "Montana": {"lat": 46.965260, "lon": -109.533691},
    "Washington": {"lat": 47.751076, "lon": -120.740135},
    "Utah": {"lat": 39.419220, "lon": -111.950684},
    "Colorado": {"lat": 39.113014, "lon": -105.358887},
    "Ohio": {"lat": 40.367474, "lon": -82.996216},
    "Alabama": {"lat": 32.318230, "lon": -86.902298},
    "Iowa": {"lat": 42.032974, "lon": -93.581543},
    "New Mexico": {"lat": 34.307144, "lon": -106.018066},
    "South Carolina": {"lat": 33.836082, "lon": -81.163727},
    "Pennsylvania": {"lat": 41.203323, "lon": -77.194527},
    "Arizona": {"lat": 34.048927, "lon": -111.093735},
    "Maryland": {"lat": 39.045753, "lon": -76.641273},
    "Massachusetts": {"lat": 42.407211, "lon": -71.382439},
    "California": {"lat": 36.778259, "lon": -119.417931},
    "Idaho": {"lat": 44.068203, "lon": -114.742043},
    "Wyoming": {"lat": 43.075970, "lon": -107.290283},
    "North Carolina": {"lat": 35.782169, "lon": -80.793457},
    "Louisiana": {"lat": 30.391830, "lon": -92.329102},
    "Houston": {"lat": 29.7601, "lon": 95.3701},
    "Santa Clara County": {"lat": 37.2939, "lon": 121.7195},
    "District of Columbia": {"lat": 38.9072, "lon": 77.0369},
    "Orange County": {"lat": 33.7175, "lon": 117.8311},
    "CDC/DHQP/CEMB": {"lat": 33.4758, "lon": 84.1942},
    "LA County": {"lat": 34.3872, "lon": 118.1123}
}

# Load Excel file and get sheet names
excel_file = 'NY.Subclusters_Feb2025.selected.xlsx'
sheet_names = pd.ExcelFile(excel_file).sheet_names  # Get all sheet names

# Initialize Dash app
app = Dash(__name__)
server = app.server

# Define a truncate function for long strings
def truncate_strings(df, max_length=15):
    # Limit string length in each cell
    return df.map(lambda x: x[:max_length] + '...' if isinstance(x, str) and len(x) > max_length else x)

# Define app layout
slider_marks = {i: month for i, month in enumerate(month_list)}
app.layout = html.Div([
    html.H1("Subcluster visualization"),
    html.H3("Please select the organism:"),
    dcc.RadioItems(
        id="sheet-radio",
        options=[{'label': sheet, 'value': sheet} for sheet in sheet_names],
        value=None,  # No sheet selected initially
        labelStyle={'display': 'block'}
    ),
    html.H3("Please select the subcluster type:"),
    html.Div(id="Type-dropdown-container"),
    html.H3("Subclusters within the selected type:"),
    html.Div(id="output-container_1"),
    html.Div(id="output-container_2"),
    dcc.Store(id="stored-table-data"),
    html.H3("Uploaded curve"),
    dcc.Slider(
        id="timeline-slider",
        min=0,
        max=len(month_list) - 1,
        step=1,
        value=0,
        marks=slider_marks,
        tooltip={"placement": "bottom", "always_visible": True}
    ),
    dcc.Graph(id="bar-chart"),
    html.H3("SNP distance"),
    html.Div(id="SNP-distance"),
    html.H3("Phylogenetic Tree"),
    html.Img(
        id="tree-image",
        style={
            'max-width': '100%',
            "max-height": "100%",
            'height': 'auto',
            "width": "auto",
            "display": "block",
            "margin": "auto"
        }
    )
])


# Callback to create the dropdown options based on selected sheet
type_order = ['new', 'pre-existing']
@app.callback(
    Output("Type-dropdown-container", "children"),
    Input("sheet-radio", "value")
)
def update_Type_dropdown(selected_sheet):
    #if not selected_sheet:
        #return html.Div("Please select the organism.") # No dropdown if no sheet is selected

    # Load the selected sheet and get unique values in 'Type' column
    df = pd.read_excel(excel_file, sheet_name=selected_sheet)
    if 'Type' in df.columns:
        unique_categories = df['Type'].fillna('others').unique()
        unique_categories = [str(cat) for cat in unique_categories]
        sorted_categories = sorted(
            unique_categories,
            key=lambda x: type_order.index(x) if x in type_order else len(type_order)
        )
        dropdown_options = [{'label': cat, 'value': cat} for cat in sorted_categories]
        return dcc.Dropdown(
            id="Type-dropdown",
            options=dropdown_options,
            #placeholder="Select a type",
            multi=False
        )
    return ""

# Callback to load and filter data based on selected sheet and Type
@app.callback(
    Output("output-container_1", "children"),
    [Input("sheet-radio", "value"),
     Input("Type-dropdown", "value")]
)
def display_sheet(selected_sheet, selected_Type):
    global original_df # Use the global variable to keep track of the original DataFrame

    if not selected_sheet:
        return html.Div("Select a subcluster.")

    # Load data for the selected sheet and truncate long strings
    original_df = pd.read_excel(excel_file, sheet_name=selected_sheet)
    #df_truncated = truncate_strings(filtered_original_df)

    # Filter data based on the selected Type
    filtered_original_df = original_df[original_df['# of new isolates after removing those from the same patient'].astype(int) > 1]
    df_type = filtered_original_df[filtered_original_df['Type'] == selected_Type]
    df_type = df_type.sort_values(by=['# of new isolates after removing those from the same patient'], ascending=True)

    # Create a DataTable with pagination and row click functionality
    return dash_table.DataTable(
        id="data-table",
        data=df_type.to_dict('records'),
        columns=[{"name": col, "id": col} for col in df_type.columns],
        page_size=5,
        style_table={'overflowX': 'auto'},
    )

# Callback to display data from another sheet when a row is clicked
@app.callback(
    [Output("output-container_2", "children"),
     Output("stored-table-data", "data")],
    Input("data-table", "active_cell"),
    State("data-table", "data"),
    State("data-table", "page_current")
)
def display_related_data(active_cell, table_data, page_current):
    if not active_cell:
        return html.Div("Click on a row to display related data.")

    # Identify the clicked row
    page_current = page_current or 0
    index_in_page = active_cell['row']
    row_index = page_current * 5 + index_in_page

    # Retrieve isolate ids
    table_data_df = pd.DataFrame(table_data)
    isolate_ids = table_data_df.loc[row_index, "new isolates after removing those from the same patient"].split()  # Get original IDs
    if not isolate_ids:
        return "No related isolates available."

    # Load full dataset as a DataFrame and filter based on isolate IDs
    df_all = pd.read_csv('Linelist.2025-03-25.tsv', delimiter='\t')
    df_filtered = df_all[df_all['Isolate'].isin(isolate_ids)]
    #df_filtered_truncated = truncate_strings(df_filtered)

    data_table = dash_table.DataTable(
        id = 'linelist_table',
        data=df_filtered.to_dict('records'),
        columns=[{"name": col, "id": col} for col in df_filtered.columns],
        page_size=5,
        style_table={'overflowX': 'auto'}
    )

    title = html.H3("Line list of isolates in the selected subcluster:")
    return [title, data_table], df_filtered.to_dict('records')

@app.callback(
    Output("bar-chart", "figure"),
    Input("stored-table-data", "data")
)
def update_bar_chart(linelist):
    df = pd.DataFrame(linelist)
    number_list = []
    isolate_list = []
    hovertext_list = []
    for i in range(len(month_list)):
        month = month_list2[i]
        target_rows = df[df['Added_month'].astype(str) == month]
        if not target_rows.empty:
            number_list.append(len(target_rows))
            targeted_isolates = target_rows['HAI_WGS_ID'].tolist()
            isolate_list.append(targeted_isolates)
            hovertext_list.append(','.join(targeted_isolates))
        else:
            number_list.append(0)
            isolate_list.append(None)
            hovertext_list.append('')
    bar_df = pd.DataFrame({
            "month": month_list,
            "number": number_list,
            "isolates": isolate_list,
            "hovertext": hovertext_list
    })
    # Create the bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=bar_df['month'],
            y=bar_df['number'],
            marker=dict(color="blue"),
            hovertext=bar_df['hovertext'],
            hoverinfo="text"
        )
    ])

    # Customize layout
    fig.update_layout(
        plot_bgcolor="white",  # Set the plot area background to white
        paper_bgcolor="white",  # Set the entire background to white
        xaxis=dict(showgrid=False),  # Hide x-axis gridlines
        yaxis=dict(showgrid=False, dtick=1),
        height=100,
        margin=dict(t=0, b=0, l=0, r=0),
    )

    return fig


@app.callback(
    Output("SNP-distance", "children"),
    [Input("stored-table-data", "data"),
     Input("sheet-radio", "value"),
     Input("data-table", "active_cell")],
    [State("data-table", "data")]
)
def update_SNP_distance(linelist, organism, active_cell, table_data):
    if not linelist:
        return html.Div("No data available.")  # Return a message if linelist is empty

    # Linelist data to a DataFrame
    df_linelist = pd.DataFrame(linelist)

    # Check if a specific cell is selected and handle accordingly
    if active_cell:
        row_index = active_cell['row']
        selected_row = table_data[row_index]
        subcluster_name = selected_row['subcluster_id']
        previous_isolate_number = 0

    # Construct SNP path dynamically based on selected organism and subcluster name
    SNP_path = f'tree/{organism}/{subcluster_name}-{month_list2[-1]}_SNP_matrix.txt'

    try:
        SNP_file = glob.glob(SNP_path)[0]
        SNP_df = pd.read_csv(SNP_file, delimiter='\t')
    except IndexError:  # In case the file is not found
        return html.Div("SNP file not found.")
    except Exception as e:  # Handle any other errors gracefully
        return html.Div(f"An error occurred: {str(e)}")

    # If SNP_df is empty, provide an alternative message
    if SNP_df.empty:
        return html.Div("SNP matrix is empty.")

    # Create and return a DataTable with the SNP data
    return html.Div(
        dash_table.DataTable(
            id="SNP-distance-table",  # Change ID if needed
            data=SNP_df.to_dict('records'),
            columns=[{"name": col, "id": col} for col in SNP_df.columns],
            style_table={
                'width': '100%',  # Let the table use the full width of the parent container
                'maxWidth': '100%',  # Ensure the table doesn't exceed 100% of the container width
                'overflowX': 'auto',  # Allow horizontal scrolling if the content overflows
                'overflowY': 'auto',  # Allow vertical scrolling if the content is long
                'height': 'auto',  # Let the height adjust dynamically
            },
            style_cell={
                'textAlign': 'center',  # Align text to the center
                'minWidth': '100px',  # Set minimum cell width
                'maxWidth': '300px',  # Set maximum cell width
                'whiteSpace': 'normal',  # Allow word wrapping if needed
            },
        )
    )

@app.callback(
    [Output("tree-image", "src")],
    [Input("stored-table-data", "data"),
     Input("sheet-radio", "value"),
     Input("data-table", "active_cell")],
     [State("data-table", "data")]
)
def update_SNP_distance(linelist, organism, active_cell, table_data):
    if not linelist:
        return [""]

    # Linelist data to a dataframe
    df_linelist = pd.DataFrame(linelist)

    # Find out if there is isolates prior to 2024
    if active_cell:
        row_index = active_cell['row']
        selected_row = table_data[row_index]
        subcluster_name = selected_row['subcluster_id']

    ## Load the .pgn phylogenetic tree
    tree_path = f'tree/{organism}/{subcluster_name}-{month_list2[-1]}_tree.png'
    try:
        tree_file = glob.glob(tree_path)[0]
        encoded_image = base64.b64encode(open(tree_file, 'rb').read()).decode()
    except IndexError:  # In case the file is not found
        return html.Div("SNP file not found.")
    except Exception as e:  # Handle any other errors gracefully
        return html.Div(f"An error occurred: {str(e)}")

    return [f"data:image/png;base64,{encoded_image}"]


if __name__ == "__main__":
    app.run_server(debug=True)