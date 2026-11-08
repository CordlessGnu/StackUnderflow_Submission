"""Launch the local GeoLens app with its project theme/server configuration.

Usage: python run_geolens.py [--server.port 8502]
"""
from pathlib import Path
import os
import sys


def main() -> None:
    app_dir = Path(__file__).resolve().parent / 'Viridien-GeoLens'
    os.chdir(app_dir)
    sys.path.insert(0, str(app_dir))
    sys.argv = ['streamlit', 'run', str(app_dir / 'app.py'), *sys.argv[1:]]
    from streamlit.web.cli import main as streamlit_main
    raise SystemExit(streamlit_main())


if __name__ == '__main__':
    main()
