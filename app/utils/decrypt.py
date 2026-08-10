"""Envelope decryption — must match backend/utils/envelope-encryption.util.ts."""

from __future__ import annotations

import base64
import hashlib
from typing import Any

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

SCRYPT_SALT = b"doqseal-envelope-v1"
DEK_LENGTH = 32


def derive_org_key(aes_secret: str, organisation_id: str) -> bytes:
    if not aes_secret or len(aes_secret) < 32:
        raise ValueError("AES_SECRET must be at least 32 characters")

    password = f"{aes_secret}:{organisation_id}".encode("utf-8")
    return hashlib.scrypt(password, salt=SCRYPT_SALT, n=16384, r=8, p=1, dklen=DEK_LENGTH)


def _aes_gcm_decrypt(key: bytes, iv: bytes, ciphertext: bytes, auth_tag: bytes) -> bytes:
    aesgcm = AESGCM(key)
    return aesgcm.decrypt(iv, ciphertext + auth_tag, None)


def decrypt_document_file(
    ciphertext: bytes,
    encryption: dict[str, Any],
    organisation_id: str,
    aes_secret: str,
) -> bytes:
    org_key = derive_org_key(aes_secret, organisation_id)

    dek_iv = base64.b64decode(encryption["dekIv"])
    dek_auth_tag = base64.b64decode(encryption["dekAuthTag"])
    encrypted_dek = base64.b64decode(encryption["encryptedDEK"])
    dek = _aes_gcm_decrypt(org_key, dek_iv, encrypted_dek, dek_auth_tag)

    file_iv = base64.b64decode(encryption["iv"])
    file_auth_tag = base64.b64decode(encryption["authTag"])
    return _aes_gcm_decrypt(dek, file_iv, ciphertext, file_auth_tag)