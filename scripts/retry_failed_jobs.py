#!/usr/bin/env python3
"""Re-run failed extraction jobs (e.g. after fixing the worker)."""

from __future__ import annotations

import argparse
import json
import sys

import pika

from app.config import settings
from app.db.mongo import get_db
from app.worker import process_job


def list_failed(limit: int) -> list[dict]:
    return list(
        get_db()
        .extraction_jobs.find({"status": "failed"})
        .sort("updatedAt", -1)
        .limit(limit)
    )


def reset_job(job_id: str) -> None:
    db = get_db()
    job = db.extraction_jobs.find_one({"jobId": job_id})
    if not job:
        raise ValueError(f"Job not found: {job_id}")
    db.extraction_jobs.update_one(
        {"jobId": job_id},
        {"$set": {"status": "queued", "error": None, "completedAt": None}},
    )
    db.documents.update_one(
        {"documentId": job["documentId"]},
        {"$set": {"status": "queued"}},
    )


def enqueue(job_id: str) -> None:
    connection = pika.BlockingConnection(pika.URLParameters(settings.amqp_uri))
    channel = connection.channel()
    channel.queue_declare(queue=settings.extraction_queue, durable=True)
    channel.basic_publish(
        exchange="",
        routing_key=settings.extraction_queue,
        body=json.dumps({"jobId": job_id}),
        properties=pika.BasicProperties(delivery_mode=2),
    )
    connection.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--job-id",
        action="append",
        dest="job_ids",
        help="Specific job ID to retry (repeatable)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Max failed jobs to retry when no --job-id is given",
    )
    parser.add_argument(
        "--enqueue",
        action="store_true",
        help="Publish to RabbitMQ instead of processing inline",
    )
    args = parser.parse_args()

    if args.job_ids:
        jobs = [{"jobId": jid} for jid in args.job_ids]
    else:
        jobs = list_failed(args.limit)

    if not jobs:
        print("No failed jobs found.")
        return 0

    for job in jobs:
        job_id = job["jobId"]
        reset_job(job_id)
        if args.enqueue:
            enqueue(job_id)
            print(f"Queued {job_id}")
        else:
            process_job(job_id)
            print(f"Completed {job_id}")

    return 0


if __name__ == "__main__":
    sys.exit(main())