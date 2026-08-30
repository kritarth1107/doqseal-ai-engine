import json
import logging

import pika

from app.config import settings
from app.db.mongo import (
    load_document,
    load_job,
    load_project,
    mark_job_completed,
    mark_job_failed,
    mark_job_processing,
)
from app.pipeline.runner import run_extraction_pipeline
from app.rag.indexer import index_extraction
from app.webhooks import dispatch_project_webhooks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("doqseal.worker")


def process_job(job_id: str) -> None:
    job = load_job(job_id)
    if not job:
        raise ValueError(f"Job not found: {job_id}")

    if job.get("status") == "completed":
        logger.info("Skipping already completed job %s", job_id)
        return

    document_id = job["documentId"]
    project_id = job.get("projectId") or None
    organisation_id = job["organisationId"]

    document = load_document(document_id)
    if not document:
        raise ValueError(f"Document not found: {document_id}")

    if project_id:
        project = load_project(project_id)
        if not project:
            raise ValueError(f"Project not found: {project_id}")
    else:
        project = {
            "projectId": None,
            "name": "Organisation Drive",
            "extractionHint": "",
            "fields": [],
            "crossFieldRules": [],
        }

    mark_job_processing(job_id, document_id)
    hint = (project.get("extractionHint") or "").strip()
    logger.info(
        "Processing job %s document=%s project=%s mode=%s hint_chars=%d",
        job_id,
        document_id,
        project_id or "_common",
        settings.extraction_mode,
        len(hint),
    )
    if hint:
        logger.info("Extraction context for %s: %s", project_id, hint[:240])

    if project_id:
        try:
            dispatch_project_webhooks(
                project,
                event="document.processing",
                project_id=project_id,
                document_id=document_id,
                job_id=job_id,
                organisation_id=organisation_id,
                document=document,
                status="processing",
            )
        except Exception:
            logger.exception("Webhook dispatch failed for processing %s", job_id)

    try:
        extraction_payload = run_extraction_pipeline(
            document, project, organisation_id
        )
    except Exception as error:
        mark_job_failed(job_id, document_id, str(error))
        if project_id:
            try:
                dispatch_project_webhooks(
                    project,
                    event="document.failed",
                    project_id=project_id,
                    document_id=document_id,
                    job_id=job_id,
                    organisation_id=organisation_id,
                    document=document,
                    error=str(error),
                    status="failed",
                )
            except Exception:
                logger.exception("Webhook dispatch failed for failed %s", job_id)
        raise

    mark_job_completed(
        job_id,
        document_id,
        organisation_id,
        project_id,
        extraction_payload,
    )
    logger.info(
        "Completed job %s strategy=%s status=%s",
        job_id,
        extraction_payload.get("strategy"),
        extraction_payload.get("status"),
    )

    if project_id:
        try:
            dispatch_project_webhooks(
                project,
                event="document.processed",
                project_id=project_id,
                document_id=document_id,
                job_id=job_id,
                organisation_id=organisation_id,
                document=document,
                extraction_payload=extraction_payload,
                status="completed",
            )
        except Exception:
            logger.exception("Webhook dispatch failed for job %s", job_id)

    try:
        indexed = index_extraction(
            organisation_id=organisation_id,
            document_id=document_id,
            job_id=job_id,
            project_id=project_id,
            uploaded_by=document.get("uploadedBy"),
            shared_with_organisation=document.get("sharedWithOrganisation") is not False,
            ocr_full_text=extraction_payload.get("ocrFullText"),
            extraction_data=extraction_payload.get("data"),
        )
        logger.info("RAG indexed %d chunks for job %s", indexed, job_id)
    except Exception:
        logger.exception("RAG indexing failed for job %s", job_id)


def on_message(channel, method, _properties, body):
    try:
        payload = json.loads(body.decode("utf-8"))
        job_id = payload.get("jobId")
        if not job_id:
            raise ValueError("Missing jobId in queue payload")

        process_job(job_id)
        channel.basic_ack(delivery_tag=method.delivery_tag)
    except Exception as error:
        logger.exception("Worker failed: %s", error)
        try:
            payload = json.loads(body.decode("utf-8"))
            job_id = payload.get("jobId")
            if job_id:
                job = load_job(job_id)
                if job and job.get("status") != "failed":
                    mark_job_failed(job_id, job["documentId"], str(error))
                    project_id = job.get("projectId")
                    if project_id:
                        project = load_project(project_id)
                        document = load_document(job["documentId"])
                        dispatch_project_webhooks(
                            project,
                            event="document.failed",
                            project_id=project_id,
                            document_id=job["documentId"],
                            job_id=job_id,
                            organisation_id=job["organisationId"],
                            document=document,
                            error=str(error),
                            status="failed",
                        )
        except Exception:
            logger.exception("Failed to mark job as failed")
        channel.basic_ack(delivery_tag=method.delivery_tag)


def start_worker() -> None:
    if settings.warmup_models:
        try:
            from app.pipeline.ocr import warmup_ocr

            warmup_ocr()
        except Exception:
            logger.exception("OCR warmup failed (continuing)")

        if settings.warmup_vlm and settings.extraction_mode.lower() == "hybrid":
            try:
                from app.pipeline.vlm_extract import warmup_vlm

                warmup_vlm()
            except Exception:
                logger.exception("VLM warmup failed (continuing)")

    connection = pika.BlockingConnection(pika.URLParameters(settings.amqp_uri))
    channel = connection.channel()
    channel.queue_declare(queue=settings.extraction_queue, durable=True)
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(
        queue=settings.extraction_queue,
        on_message_callback=on_message,
    )

    logger.info(
        "DoqSeal extraction worker listening on queue '%s' (mode=%s)",
        settings.extraction_queue,
        settings.extraction_mode,
    )
    channel.start_consuming()


if __name__ == "__main__":
    start_worker()
