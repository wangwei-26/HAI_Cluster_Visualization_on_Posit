from dash import Dash, html, dash_table, dcc, callback, Output, Input, State
import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
import glob
import base64
import os

# Months to monitor
month_list = ['2023-11','2023-12','2024-01', '2024-02', '2024-03', '2024-04', '2024-05', '2024-06', '2024-07', '2024-08',
              '2024-09','2024-10','2024-11','2024-12','2025-01','2025-02','2025-03']###########################################################
month_list2 = ['11.2023','12.2023','1.2024','2.2024','3.2024','4.2024','5.2024','6.2024','7.2024','8.2024','9.2024',
               '10.2024','11.2024','12.2024','1.2025','2.2025','3.2025','4.2025']####################################################################

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
        "Houston": "HOU", ###
        "Iowa": "IA",
        "Idaho": "ID",
        "Illinois": "IL",
        "Indiana": "IN",
        "Kansas": "KS",
        "Kentucky": "KY",
        "Louisiana": "LA",
        "LA County": "LAC", ###
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
        "New York City": "NYC",  ###
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

# Load Excel file and get sheet names
excel_file = 'docs/March.2025/Subclusters_Mar2025.IDupdated.xlsx' ###############################################################
sheet_names = [sheet for sheet in pd.ExcelFile(excel_file).sheet_names if sheet != 'Data Dictionary']  # Get all sheet names

# Load sequence data
organism_df_dict = {}
sequence_excel_file = pd.ExcelFile('docs/March.2025/Sequences_Mar2025.xlsx', engine='openpyxl')###################################################
for sheet_name in sequence_excel_file.sheet_names:
    df = pd.read_excel(sequence_excel_file, sheet_name)
    organism = sheet_name.split('_')[0]
    if organism != 'Data Dictionary':
        organism_df_dict[organism] = df

# Dictionary of added_month
linelist_df = pd.read_csv('docs/March.2025/Linelist.2025-04-07.tsv', delimiter='\t')
HAI_added_dict = linelist_df.set_index('HAI_WGS_ID')['Added_month'].to_dict()

# Initialize Dash app
app = Dash(__name__)
server = app.server

# Define a truncate function for long strings
# def truncate_strings(df, max_length=15):
#     # Limit string length in each cell
#     return df.map(lambda x: x[:max_length] + '...' if isinstance(x, str) and len(x) > max_length else x)

# Define app layout
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
    dcc.RadioItems(
        id="sheet-radio2",
        options=[
            {'label': 'New', 'value': 'new'},
            {'label': 'Pre-existing', 'value': 'pre-existing'}
        ],
        value=None,  # No option selected initially
        labelStyle={'display': 'block'}
    ),
    html.H3("Subclusters within the selected type:"),
    html.Div(id="output-container_1"),
    html.Div(id="output-container_2"),
    dcc.Store(id="stored-table-data"),
    html.H3("Uploaded curve"),
    dcc.Graph(id="bar-chart"),
    html.H3("SNP distance"),
    html.Div(id="SNP-distance"),
    html.H3("Phylogenetic tree"),
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

# Callback to load and filter data based on selected sheet and Type
@app.callback(
    Output("output-container_1", "children"),
    [Input("sheet-radio", "value"),
     Input("sheet-radio2", "value")]
)
def display_sheet(selected_sheet, selected_Type):
    global original_df

    if not selected_sheet:
        return html.Div("Select a subcluster.")

    # Load data for the selected sheet and truncate long strings
    original_df = pd.read_excel(excel_file, sheet_name=selected_sheet)
    #df_truncated = truncate_strings(filtered_original_df)

    # Filter data based on the selected Type
    filtered_original_df = original_df[original_df['Sequence Count'].astype(int) > 1]
    df_type = filtered_original_df[filtered_original_df['Subcluster Type'] == selected_Type]

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
    Input("sheet-radio", "value"),
    State("data-table", "data"),
    State("data-table", "page_current")
)
def display_related_data(active_cell, organism, table_data, page_current):
    if not active_cell:
        return html.Div("Click on a row to display related data.")

    # Identify the clicked row
    page_current = page_current or 0
    index_in_page = active_cell['row']
    row_index = page_current * 5 + index_in_page

    # Retrieve isolate ids
    table_data_df = pd.DataFrame(table_data)
    isolate_ids = table_data_df.loc[row_index, "Sequences"].split(' ')  # Get original IDs
    if not isolate_ids:
        return "No related isolates available."

    # Load full dataset as a DataFrame and filter based on isolate IDs
    sequence_df = organism_df_dict[organism]
    df_filtered = sequence_df[sequence_df['HAI WGS ID'].isin(isolate_ids)]
    #df_filtered_truncated = truncate_strings(df_filtered)

    # Added month
    df_filtered['Added Month.Year'] = df_filtered['HAI WGS ID'].map(HAI_added_dict)

    data_table = dash_table.DataTable(
        id = 'linelist_table',
        data=df_filtered.to_dict('records'),
        columns=[{"name": col, "id": col} for col in df_filtered.columns],
        page_size=5,
        style_table={'overflowX': 'auto'}
    )

    title = html.H3("Line list of sequences in the selected subcluster:")
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
        target_rows = df[df['Added Month.Year'].astype(str) == month]
        if not target_rows.empty:
            number_list.append(len(target_rows))
            targeted_isolates = target_rows['HAI WGS ID'].tolist()
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
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(
            showgrid=False,
            showline=True,
            linecolor='black'
        ),
        yaxis=dict(
            title='Upload count',
            showgrid=False,
            dtick=1,
            showline=True,
            linecolor='black'
        ),
        height=200,
        margin=dict(t=20, b=40, l=60, r=20),
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
        subcluster_name = selected_row['Subcluster ID (internal)']

    # Construct SNP path dynamically based on selected organism and subcluster name
    SNP_path = f'docs/March.2025/trees/{organism}/{subcluster_name}-{month_list2[-1]}_SNP_matrix.txt'

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
        subcluster_name = selected_row['Subcluster ID (internal)']

    ## Load the .pgn phylogenetic tree
    tree_path = f'docs/March.2025/trees/{organism}/{subcluster_name}-{month_list2[-1]}_tree.png'
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
