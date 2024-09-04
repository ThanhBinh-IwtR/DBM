import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

# Loading tap data
df = pd.read_csv("D:\FPTMaterial\ky 7\DBM\emissions_high_granularity.csv")

# gộp các hàng có cùng year và tính tổng production_value và total_emissions_MtCO2e trong cùng một năm
new_df = df.groupby(['year']).sum().reset_index()

# gộp các hàng cùng year và parent_entity và tính tổng production_value và total_emissions_MtCO2e của parent_type trong cùng một năm
new_df_2 = df.groupby(['year', 'parent_entity']).sum().reset_index()

# Create a Dash app
app = dash.Dash(__name__)

# tạo ra line chart của tập new_df của production_value qua thời gian
def line_chart(new_df):
    fig = px.line(new_df, x='year', y='production_value', title='Total Production Value Over Time')
    return fig

# tạo ra line chart của tập new_df của 'total_emissions_MtCO2e' qua thời gian
def line_chart2(new_df):
    fig = px.line(new_df, x='year', y='total_emissions_MtCO2e', title='Total Emissions Over Time')
    return fig

# tạo ra biểu đồ line chart của production_value của tất cả parent_entity qua thời gian trong tập data
def line_chart6(new_df_2):
    fig = px.line(new_df_2, x='year', y='production_value', title='Total Production Value Over Time by Parent Entity', color='parent_entity')
    return fig
    
# tạo biểu đồ bar chart thể hiện số lượng commodity trong data với các cột các màu khác nhau
def bar_chart1(df):
    # Sử dụng bảng màu Spectral có độ tương phản cao
    fig = px.bar(df, x='commodity', title='Number of Commodities', 
                 color='commodity', 
                 color_discrete_sequence=px.colors.diverging.Spectral)
    fig.update_traces(marker=dict(line=dict(width=0)))
    return fig

# tạo biểu đồ bar chart thể hiện số lượng parent_type trong data
def bar_chart2(df):
    # Tạo biểu đồ cột với màu sắc đậm hơn
    custom_colors = ['#FF5733', '#33FF57', '#3357FF']  # Màu sắc đậm hơn cho cái cột
    fig = px.bar(df, x='parent_type', title='Number of Parent Types', 
                 color='parent_type')  # Đặt độ mờ bằng 1 để có màu đậm nhất
    fig.update_traces(marker=dict(line=dict(width=0)))
    return fig

# tạo biểu đồ bar chart thể hiện top 10 parent_entity có production_value lớn nhất trong data
def bar_chart5(df):
    # Sử dụng bảng màu Viridis có độ tương phản cao
    fig = px.bar(df.nlargest(10, 'production_value'), x='year', y='production_value', 
                 color='parent_entity', title='Top 10 Largest Production Values ', 
                 color_discrete_sequence=px.colors.sequential.Viridis)
    fig.update_traces(marker=dict(line=dict(width=0)))

    return fig

# tạo biểu đồ bar chart thể hiện top 5 parent_entity có production_value lớn nhất trong data 
def bar_chart3(df):
    # Chỉ chọn các bản ghi có giá trị sản xuất lớn nhất
    df_top = df.groupby(['year', 'parent_entity'])['production_value'].sum().reset_index()
    df_top = df_top.sort_values('production_value', ascending=False).groupby('year').head(5)
    
    # Sử dụng bảng màu Viridis có độ tương phản cao
    fig = px.bar(df_top, x='year', y='production_value', 
                 color='parent_entity', 
                 title='Top 5 Largest Production Values by Years', 
                 color_discrete_sequence=px.colors.sequential.Viridis)
    fig.update_traces(marker=dict(line=dict(width=0)))

    return fig

# tạo biểu đồ bar chart thể hiện top 5 parent_entity có total_emissions_MtCO2e lớn nhất trong data
def bar_chart4(df):
    # Sử dụng bảng màu Inferno có độ tương phản cao
    df_top = df.groupby(['year', 'parent_entity'])['total_emissions_MtCO2e'].sum().reset_index()
    df_top = df_top.sort_values('total_emissions_MtCO2e', ascending=False).groupby('year').head(5)
    
    # Sử dụng bảng màu Viridis có độ tương phản cao
    fig = px.bar(df_top, x='year', y='total_emissions_MtCO2e', 
                 color='parent_entity', 
                 title='Top 5 Largest total_emissions_MtCO2e by Years', 
                 color_discrete_sequence=px.colors.sequential.Viridis)
    fig.update_traces(marker=dict(line=dict(width=0)))

    return fig
    
# tao biểu đồ line chart của production_value của parent_entity qua thời gian
def line_chart3(new_df_2, parent_entity):
    new_df_2 = new_df_2[new_df_2['parent_entity'] == parent_entity]
    fig = px.line(new_df_2, x='year', y='production_value', title=f'Production Value for {parent_entity}', color='parent_entity')
    return fig

# tạo biểu đồ line chart của total_emissions_MtCO2e của parent_entity qua thời gian
def line_chart4(new_df_2, parent_entity):
    new_df_2 = new_df_2[new_df_2['parent_entity'] == parent_entity]
    fig = px.line(new_df_2, x='year', y='total_emissions_MtCO2e', title=f'Total Emissions for {parent_entity}')
    return fig

# tạo biểu đồ line chart của emissions_per_value của parent_entity qua thời gian
def line_chart5(new_df_2, parent_entity):
    new_df_2 = new_df_2[new_df_2['parent_entity'] == parent_entity]
    new_df_2['emissions_per_value'] = new_df_2['total_emissions_MtCO2e'] / new_df_2['production_value']
    fig = px.line(new_df_2, x='year', y='emissions_per_value', title=f'Emissions per Value for {parent_entity}')
    return fig


# tạo biểu đồ scatter plot của total_emissions_MtCO2e của parent_entity qua thời gian
def scatter_plot(new_df_2):
    fig = px.scatter(new_df_2, x='year', y='total_emissions_MtCO2e', title='Total Emissions Over Time by Parent Type', color='parent_entity')
    return fig

# tạo biểu đồ scatter plot của production value theo thời gian
def scatter_plot2(new_df_2):
    fig = px.scatter(new_df_2, x='year', y='production_value', title='Total Production Value Over Time by Parent Type', color='parent_entity')
    return fig

# Heatmap cho biết mức độ tương quan giữa các cột trong data
# chuyển hóa dữ liệu thành dạng số để tính toán mà không ảnh hưởng đến dữ liệu gốc

def correlation_heatmap(df):
    data = df.copy()
    cat_columns = data.select_dtypes(['object']).columns
    for col in cat_columns:
        data[col] = data[col].astype('category').cat.codes
    corr = data.corr()
    fig = px.imshow(corr, title='Correlation Heatmap')
    return fig


# layout
app.layout = html.Div([
    html.H1('Dashboard of Gas Emissions'),
    html.P('This dashboard shows the total production value and emissions over time, as well as the number of commodities and parent types in the data.'),
    dcc.Graph(id='line-chart', figure=line_chart(new_df)),
    dcc.Graph(id='line-chart2', figure=line_chart2(new_df)),
    dcc.Graph(id='line-chart6', figure=line_chart6(new_df_2)),
    dcc.Graph(id='bar-chart1', figure=bar_chart1(df)),
    dcc.Graph(id='bar-chart2', figure=bar_chart2(df)),
    dcc.Graph(id='bar-chart5', figure=bar_chart5(df)),
    dcc.Graph(id='bar-chart3', figure=bar_chart3(df)),
    dcc.Graph(id='bar-chart4', figure=bar_chart4(df)),
    dcc.Graph(id='correlation-heatmap', figure=correlation_heatmap(df)),
    dcc.Dropdown(
        id='parent-entity-dropdown',
        options=[{'label': x, 'value': x} for x in df['parent_entity'].unique()],
        value='World',
        clearable=False
    ),
    dcc.Graph(id='line-chart3'),
    dcc.Graph(id='line-chart4'),
    dcc.Graph(id='line-chart5'),
    dcc.Graph(id='scatter-plot'),
    dcc.Graph(id='scatter-plot2', figure=scatter_plot2(new_df_2))
])

# callback
@app.callback(
    Output('line-chart3', 'figure'),
    Output('line-chart4', 'figure'),
    Output('line-chart5', 'figure'),
    Output('scatter-plot', 'figure'),
    Input('parent-entity-dropdown', 'value')
)
def update_charts(parent_entity):
    fig1 = line_chart3(new_df_2, parent_entity)
    fig2 = line_chart4(new_df_2, parent_entity)
    fig3 = line_chart5(new_df_2, parent_entity)
    fig4 = scatter_plot(new_df_2)
    return fig1, fig2, fig3, fig4

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)







