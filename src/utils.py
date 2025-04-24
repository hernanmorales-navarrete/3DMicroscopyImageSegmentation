import tensorflow as tf
from loguru import logger


def configure_gpu():
    """Configure GPU memory growth to avoid taking all memory.

    This should be called at the beginning of scripts that use GPU.
    """
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices("GPU")
            logger.info(f"Using {len(logical_gpus)} Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            logger.error(e)
    else:
        logger.warning("No GPUs found. Running on CPU only.")
