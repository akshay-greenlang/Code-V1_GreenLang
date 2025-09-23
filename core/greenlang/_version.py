from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

try:
    # Try to get version from installed package
    __version__ = version("greenlang-cli")
except PackageNotFoundError:
    try:
        # Fallback to greenlang package name
        __version__ = version("greenlang")
    except PackageNotFoundError:
        # Editable installs / dev fallback - read from VERSION file
        version_file = Path(__file__).resolve().parents[3].joinpath("VERSION")
        if version_file.exists():
            __version__ = version_file.read_text().strip()
        else:
            __version__ = "0.2.0"  # Fallback version
