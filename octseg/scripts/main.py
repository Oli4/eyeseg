import click
from octseg.scripts.layers import layers
from octseg.scripts.drusen import drusen


@click.group()
def main():
    pass


@main.group()
def predict():
    pass


predict.add_command(drusen)
predict.add_command(layers)
