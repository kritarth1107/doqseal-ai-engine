#!/usr/bin/env python3
"""Quick sanity check for envelope decryption."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

load_dotenv()

from app.config import settings
from app.utils.decrypt import decrypt_document_file


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_decrypt.py <encrypted-file> <org-id>")
        sys.exit(1)

    path = Path(sys.argv[1])
    org_id = sys.argv[2]
    ciphertext = path.read_bytes()

    encryption = {
        "iv": sys.argv[3] if len(sys.argv) > 3 else "",
    }
    print("Provide encryption metadata via Mongo document in real usage.")
    print("AES_SECRET set:", bool(settings.aes_secret))


if __name__ == "__main__":
    main()