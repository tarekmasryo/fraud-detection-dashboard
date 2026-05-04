from __future__ import annotations

import logging
import socket
from dataclasses import dataclass
from typing import BinaryIO
from urllib.parse import urlparse

from fraud_dashboard.core.config import get_settings

logger = logging.getLogger(__name__)
QUEUE_KEY = "fraud-risk-ops:jobs"


@dataclass(frozen=True)
class RedisEndpoint:
    host: str
    port: int
    db: int
    username: str | None = None
    password: str | None = None


def _endpoint(url: str) -> RedisEndpoint:
    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    port = int(parsed.port or 6379)
    try:
        db = int((parsed.path or "/0").lstrip("/") or "0")
    except ValueError:
        db = 0
    return RedisEndpoint(
        host=host,
        port=port,
        db=db,
        username=parsed.username,
        password=parsed.password,
    )


def _encode_command(*parts: str) -> bytes:
    out = [f"*{len(parts)}\r\n".encode()]
    for part in parts:
        raw = part.encode()
        out.append(f"${len(raw)}\r\n".encode())
        out.append(raw + b"\r\n")
    return b"".join(out)


def _read_line(reader: BinaryIO) -> bytes:
    line = reader.readline()
    if not line:
        raise ConnectionError("Redis connection closed")
    if not line.endswith(b"\r\n"):
        raise RuntimeError(f"Invalid Redis line terminator: {line!r}")
    return line[:-2]


def _read_bulk(reader: BinaryIO, length: int) -> bytes:
    data = reader.read(length + 2)
    if len(data) != length + 2:
        raise ConnectionError("Redis bulk response ended early")
    if not data.endswith(b"\r\n"):
        raise RuntimeError("Invalid Redis bulk response terminator")
    return data[:length]


def _read_response(reader: BinaryIO):  # type: ignore[no-untyped-def]
    line = _read_line(reader)
    prefix, payload = line[:1], line[1:]
    if prefix == b"+":
        return payload.decode()
    if prefix == b":":
        return int(payload)
    if prefix == b"-":
        raise RuntimeError(payload.decode())
    if prefix == b"$":
        length = int(payload)
        if length == -1:
            return None
        return _read_bulk(reader, length).decode()
    if prefix == b"*":
        count = int(payload)
        if count == -1:
            return None
        return [_read_response(reader) for _ in range(count)]
    raise RuntimeError(f"Unsupported Redis response: {line!r}")


def _command(*parts: str, timeout_s: float = 3.0):  # type: ignore[no-untyped-def]
    settings = get_settings()
    ep = _endpoint(settings.redis_url)
    with socket.create_connection((ep.host, ep.port), timeout=timeout_s) as sock:
        sock.settimeout(timeout_s + 1.0)
        with sock.makefile("rb") as reader:
            if ep.password:
                if ep.username:
                    sock.sendall(_encode_command("AUTH", ep.username, ep.password))
                else:
                    sock.sendall(_encode_command("AUTH", ep.password))
                _read_response(reader)
            if ep.db:
                sock.sendall(_encode_command("SELECT", str(ep.db)))
                _read_response(reader)
            sock.sendall(_encode_command(*parts))
            return _read_response(reader)


def enqueue_job(job_id: str) -> bool:
    """Push a job id into Redis when available.

    The SQLite batch_jobs table remains the source of truth. Redis is used as a
    lightweight wake-up queue for the Docker worker. If Redis is unavailable,
    workers can still poll queued jobs from SQLite.
    """
    try:
        _command("LPUSH", QUEUE_KEY, job_id)
        return True
    except Exception as exc:  # pragma: no cover - network availability dependent
        logger.warning("redis_enqueue_failed", extra={"job_id": job_id, "error": str(exc)})
        return False


def dequeue_job(timeout_s: int | None = None) -> str | None:
    settings = get_settings()
    timeout = int(timeout_s if timeout_s is not None else settings.worker_poll_seconds)
    try:
        response = _command("BRPOP", QUEUE_KEY, str(max(timeout, 1)), timeout_s=timeout + 2.0)
    except Exception as exc:  # pragma: no cover - network availability dependent
        logger.warning("redis_dequeue_failed", extra={"error": str(exc)})
        return None
    if not response:
        return None
    if isinstance(response, list) and len(response) == 2:
        return str(response[1])
    return None
