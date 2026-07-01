import os


def create_directory(path):
    """
    Create a directory if it does not already exist.
    """
    os.makedirs(path, exist_ok=True)


def print_section(title):
    """
    Print a formatted section title.
    """
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)