from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def print_prompt_completions_sample(prompt: list[str], completion: list[str], step: int, **kwargs) -> None:
    """
    Adapted from trl.trainer.utils.print_prompt_completions_sample but add kwargs to the table.
    """
    console = Console()
    table = Table(show_header=True, header_style="bold white", expand=True)

    # Add columns
    table.add_column("Prompt", style="bright_yellow")
    table.add_column("Completion", style="bright_green")
    for key in kwargs.keys():
        table.add_column(key.capitalize(), style="bold cyan", justify="right")

    # Prepare data for zipping
    data_to_zip = [prompt, completion] + list(kwargs.values())

    for row_data in zip(*data_to_zip):
        prompt, completion = row_data[:2]
        kwarg_values = row_data[2:]

        # Prepare row values for the table
        row_values = [
            Text(prompt),
            Text(completion),
        ]
        
        # Format and add kwarg values
        for val in kwarg_values:
            if isinstance(val, float):
                row_values.append(f"{val:.2f}")
            else:
                row_values.append(str(val))
        
        table.add_row(*row_values)
        table.add_section()  # Adds a separator between rows

    panel = Panel(table, expand=False, title=f"Step {step}", border_style="bold white")
    console.print(panel)