from celery import shared_task


@shared_task
def process_images_task(*args, **kwargs):
    """
    Legacy Celery task retained for compatibility. The previous ml_models-based
    processing has been removed, so this now simply returns an empty result.
    """
    return {}
