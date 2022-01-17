import click


@click.command()
@click.argument("drusen_threshold", type=int, default=2)
def drusen(drusen_threshold):
    print("drusen")
