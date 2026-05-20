import sys
from pathlib import Path

# Ensure the root directory is in sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.backend.flask_app import app

if __name__ == "__main__":
    print("Starting Amazon Electronics Recommendation Dashboard...")
    app.run(host="0.0.0.0", port=8050, debug=False)
