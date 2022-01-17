import click
from octseg.scripts.layers import layers
from octseg.scripts.drusen import drusen


@click.group()
def main():
    pass


@main.group()
def predict():
    pass


@main.command()
def quantify():
    print("quantify")


predict.add_command(layers)
predict.add_command(drusen)
