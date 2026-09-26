import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Tests must never reach real services.
os.environ.setdefault("MONGODB_URI", "mongodb://127.0.0.1:1/doqseal-test")
os.environ.setdefault("AMQP_URI", "amqp://guest:guest@127.0.0.1:1")
os.environ["AZURE_OPENAI_ENDPOINT"] = ""
os.environ["AZURE_OPENAI_API_KEY"] = ""
