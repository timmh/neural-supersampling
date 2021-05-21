import os

# TODO: remove
def parse_filename(filename):
    return os.path.splitext(os.path.basename(filename))[0].split("_")


def noop(arg):
    return arg


class noop_context():
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc_value, traceback):
        return False