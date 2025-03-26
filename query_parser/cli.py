import click

from query_parser import (
    transform_subqueries_to_ctes,
    transform_ctes_to_subqueries,
    MultipleQueriesError
)


@click.command()
@click.option(
    '--input', '-i',
    'input_file',
    type=click.File('r'),
    default='-',
    help="Input SQL file. Use '-' to read from standard input."
)
@click.option(
    '--output', '-o',
    'output_file',
    type=click.File('w'),
    default='-',
    help="Output file. Use '-' to write to standard output."
)
@click.option(
    '--mode',
    type=click.Choice(['sub2cte', 'cte2sub'], case_sensitive=False),
    default='sub2cte',
    help="Transformation mode: 'sub2cte' converts subqueries to CTEs; 'cte2sub' converts CTEs to inline subqueries."
)
def cli(input_file, output_file, mode) -> None:
    """
    Transform Vertica queries.

    Reads a query from the input file (or standard input) and transforms it according to the mode specified.

    Modes:
      - sub2cte: Convert transformable subqueries into CTEs.
      - cte2sub: Convert CTEs back into inline subqueries.

    Writes the transformed query to the output file (or standard output).

    Raises an error if multiple queries are detected.
    """
    query: str = input_file.read()
    try:
        if mode.lower() == 'cte2sub':
            transformed_query: str = transform_ctes_to_subqueries(query)
        else:
            transformed_query: str = transform_subqueries_to_ctes(query)
    except MultipleQueriesError as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()
    output_file.write(transformed_query)


if __name__ == '__main__':
    cli()
