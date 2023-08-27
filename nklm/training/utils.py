import click


def validate_proportion(ctx, param, value):
    if value is None:
        return
    elif 0 <= value <= 1:
        return value
    raise click.BadParameter(
        f'{value} is not between 0 and 1 (inclusive).'
    )
