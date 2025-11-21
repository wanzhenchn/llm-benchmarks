import fire

from rich.console import Console
from rich.style import Style
from rich.table import Table
import pandas as pd


def display_performance_table(csv_file: str):

    styles = {
        "header": "bold cyan",
        "border": Style(color="rgb(238,77,45)"),
        "good": Style(color="bright_green"),
        "warn": Style(color="bright_yellow", bold=True),
        "bad": Style(color="bright_red"),
        "normal": Style(color="white"),
    }

    width = 215
    console = Console(width=width)  # Set fixed width
    table = Table(
        title='Performance Test Summary',
        show_header=True,
        header_style=styles['header'],
#        border_style=styles['border'],
        width=width,  # Set total table width
    )

    # Add columns (set fixed column widths)
    table.add_column('Request Num.', justify='center')
    table.add_column('Max Tokens', justify='center')
    table.add_column('Batch Size', justify='center')
    table.add_column('ISL avg', justify='center')
    table.add_column('OSL avg', justify='center')
    table.add_column('Elapse Time (s)', justify='center')
    table.add_column('TTFT avg (ms)', justify='center')
    table.add_column('TTFT p99 (ms)', justify='center')
    table.add_column('TPOT avg (ms)', justify='center')
    table.add_column('TPOT p99 (ms)', justify='center')
    table.add_column('Lat. p90 (ms)', justify='center')
    table.add_column('Lat. p95 (ms)', justify='center')
    table.add_column('Lat. p99 (ms)', justify='center')
    table.add_column('Lat. avg (ms)', justify='center', style=styles['bad'])
    table.add_column('TPS (toks/s)', justify='center', style=styles['good'])
    table.add_column('RPS (req/s)', justify='center')
    table.add_column('Decode TPS (tps/usr)', justify='center')
    table.add_column('Prefix Cache Hit Rate', justify='center')
    table.add_column('External Prefix Cache Hit Rate', justify='center')
    table.add_column('GPU UTIL', justify='center')
    table.add_column('TENSOR Active', justify='center')
    table.add_column('SM Active', justify='center')
#    table.add_column('Request Rate', justify='center')
#    table.add_column('Burstiness', justify='center')

    try:
        df = pd.read_csv(csv_file, index_col=False)

        for row in df.iloc():
            try:
                table.add_row(
                    str(int(row.iloc[0])), # Successful Request
                    str(int(row.iloc[1])), # Request_Gen_Token_Len
                    str(int(row.iloc[2])), # Batch Size
                    str(row.iloc[3]), # Avg_Input_Token_Len
                    str(row.iloc[4]), # Avg_Gen_Token_Len
                    str(row.iloc[5]), # Elapse_Time
                    str(int(row.iloc[6])), # Time_to_First_Token_AVG
                    str(int(row.iloc[7])), # Time_to_First_Token_P99
                    str(int(row.iloc[8])), # Time_per_Output_Token_AVG
                    str(int(row.iloc[9])), # Time_per_Output_Token_P99
                    str(int(row.iloc[10])), # Latency_P90
                    str(int(row.iloc[11])), # Latency_P95
                    str(int(row.iloc[12])), # Latency_P99
                    str(int(row.iloc[13])), # Latency_AVG
                    str(int(row.iloc[14])), # Token Throughput
                    str(row.iloc[15]), # Service Throughput
                    str(int(row.iloc[16])), # Decode Token Throughput
                    str(row.iloc[17]), # Prefix Cache Hit Rate
                    str(row.iloc[18]), # External Prefix Cache Hit Rate
                    str(row.iloc[19]), # GPU UTIL
                    str(row.iloc[21]), # Tensor Active
                    str(row.iloc[22]), # SM Active
#                    str(row.iloc[25]), # Request Rate
#                    str(row.iloc[26]), # Burstiness
                )
            except ValueError as e:
                console.print(
                    f'Warning: Error processing row data: {str(e)}', style='bold red'
                )
                continue

#        console.print('\n')
        console.print(table)

    except Exception as e:
        print(f"failed to read the csv file: {e}")
        return


if __name__ == "__main__":
    fire.Fire(display_performance_table)
