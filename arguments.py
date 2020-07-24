import argparse


class ArgumentParser(argparse.ArgumentParser):

    def __init__(self):
        super().__init__()

        # Base arguments
        self.add_argument("--batch_size", default=32)


    def parse_args(self):
        args = super().parse_args()

        return args
