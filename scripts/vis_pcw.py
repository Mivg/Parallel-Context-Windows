import sys
import pandas as pd
import itertools
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def main(csv_path):
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
                        
                        show_legend = f"nspw: {nspw}, windows: {windows}" not in added_legends
                        fig.add_trace(go.Scatter(x=x_values, y=y_values, 
                                                name=f"nspw: {nspw}, windows: {windows}" if show_legend else "",
                                                mode='markers',
                                                marker=dict(symbol=nspw_marker_mapping[nspw], size=10),
                                                line=dict(color=windows_color_mapping[windows]),
                                                error_y=dict(type='data', array=y_error, visible=True),
                                                legendgroup=f"windows: {windows}", 
                                                showlegend=show_legend),
                                    row=dataset_idx, col=model_idx)
                        if show_legend:
                            added_legends.add(f"nspw: {nspw}, windows: {windows}")

    for i in fig['layout']['annotations']:
        i['font'] = dict(size=12)

    fig.update_layout(legend_orientation="h", legend=dict(x=0, y=-0.15, xanchor='left'), 
                    xaxis_title="n_shots", yaxis_title="Accuracy")

    # Save the plot as another HTML file
    another_updated_html_file_path = csv_path.replace('.csv', '.html')
    fig.write_html(another_updated_html_file_path)

    print(f'Updated HTML file saved at {another_updated_html_file_path}')


if __name__ == '__main__':
    main(sys.argv[1])