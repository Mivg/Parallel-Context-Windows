import sys
import pandas as pd
import itertools
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def wrap_html(fig, output_path, nrows, ncols):
    fig_height_per_subplot = 1000  # or whatever height you want per subplot
    total_fig_height = fig_height_per_subplot * nrows
    fig.update_layout(height=total_fig_height)


    # # Force axes to be displayed on all subplots
    # for dataset_idx in range(1, nrows + 1):
    #     for model_idx in range(1, ncols + 1):
    #         axis_suffix = f"{dataset_idx}{model_idx}" if dataset_idx > 1 or model_idx > 1 else ""
    #         fig.update_layout({
    #             f"xaxis{axis_suffix}.showticklabels": True,
    #             f"yaxis{axis_suffix}.showticklabels": True
    #         })

    # Reduce padding between subplots
    fig.update_layout(grid={'rows': nrows, 'columns': ncols, 'roworder': 'top to bottom'},
                      margin=dict(t=50, b=50, l=50, r=50),
                      title_x=0.5)

    # Wrap the figure in a div with custom CSS for scrolling
    wrapped_html = f"""
        <html>
            <head>
                <style>
                    #scrollDiv {{
                        overflow-y: auto;
                        height: 80vh;  # Adjust if needed
                    }}
                </style>
            </head>
            <body>
                <div id="scrollDiv">
                    {fig.to_html(full_html=False)}
                </div>
            </body>
        </html>
        """

    with open(output_path, "w") as f:
        f.write(wrapped_html)

def main(csv_path, std=True):
    # Reload the CSV and apply the recent modifications to recreate the figure
    df = pd.read_csv(csv_path)  # make sure you run in root as workdir
    df['windows'] = (df['n_shots'] / df['nspw']).astype(int)

    unique_datasets = df['dataset'].unique()
    unique_models = df['model'].unique()

    marker_list = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'triangle-down', 'star']
    color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    unique_nspw = df['nspw'].unique()
    unique_windows = df['windows'].unique()
    nspw_marker_mapping = dict(zip(unique_nspw, itertools.cycle(marker_list)))
    windows_color_mapping = dict(zip(unique_windows, itertools.cycle(color_list)))

    fig = make_subplots(rows=len(unique_datasets), cols=len(unique_models), 
                        subplot_titles=[f"{dataset} - {model}" for dataset, model in itertools.product(unique_datasets, unique_models)],
                        shared_xaxes=True, shared_yaxes=True)

    added_legends = set()
    for dataset_idx, dataset in enumerate(unique_datasets, start=1):
        for model_idx, model in enumerate(unique_models, start=1):
            subset_df = df[(df['dataset'] == dataset) & (df['model'] == model)]
            
            for windows in unique_windows:
                windows_df = subset_df[subset_df['windows'] == windows]
                grouped_all = windows_df.groupby('n_shots').agg({'accuracy': ['mean', 'std']}).reset_index()
                
                show_legend_line = f"line_windows: {windows}" not in added_legends
                fig.add_trace(go.Scatter(x=grouped_all['n_shots'], y=grouped_all['accuracy']['mean'], 
                                        mode='lines',
                                        line=dict(color=windows_color_mapping[windows], width=2),
                                        legendgroup=f"windows: {windows}",
                                        showlegend=show_legend_line,
                                        name=f"windows: {windows}" if show_legend_line else ""),
                            row=dataset_idx, col=model_idx)
                if show_legend_line:
                    added_legends.add(f"line_windows: {windows}")
                
                for nspw in windows_df['nspw'].unique():
                    trace_df = windows_df[windows_df['nspw'] == nspw]
                    if not trace_df.empty:
                        grouped = trace_df.groupby('n_shots').agg({'accuracy': ['mean', 'std']}).reset_index()
                        x_values = grouped['n_shots']
                        y_values = grouped['accuracy']['mean']
                        y_error = grouped['accuracy']['std']
                        error_y = dict(type='data', array=y_error, visible=True) if std else None
                        
                        show_legend = f"nspw: {nspw}, windows: {windows}" not in added_legends
                        text_values = [f"nspw: {nspw}, windows: {windows}, accuracy: {y:.2f}" for y in y_values]
                        fig.add_trace(go.Scatter(x=x_values, y=y_values,
                                                 name=f"nspw: {nspw}, windows: {windows}" if show_legend else "",
                                                 mode='markers',
                                                 marker=dict(symbol=nspw_marker_mapping[nspw], size=10),
                                                 line=dict(color=windows_color_mapping[windows]),
                                                 error_y=error_y,
                                                 legendgroup=f"windows: {windows}",
                                                 showlegend=show_legend,
                                                 hoverinfo="x+y+text",
                                                 text=text_values),
                                      row=dataset_idx, col=model_idx)
                        if show_legend:
                            added_legends.add(f"nspw: {nspw}, windows: {windows}")

    for i in fig['layout']['annotations']:
        i['font'] = dict(size=12)

    fig.update_layout(legend_orientation="h", legend=dict(x=0, y=-0.15, xanchor='left'), 
                    xaxis_title="n_shots", yaxis_title="Accuracy")

    # Save the plot as another HTML file
    another_updated_html_file_path = csv_path.replace('.csv', '.html') if std else csv_path.replace('.csv', '_no_std.html')
    wrap_html(fig, another_updated_html_file_path, len(unique_datasets), len(unique_models))
    # fig.write_html(another_updated_html_file_path)


    print(f'Updated HTML file saved at {another_updated_html_file_path}')


if __name__ == '__main__':
    main(sys.argv[1])
    main(sys.argv[1], False)