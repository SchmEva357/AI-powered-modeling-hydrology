import logging
import parsenvy

logger = logging.getLogger(__name__)

# mlflow
try:
    TRACKING_URI = open(".mlflow_uri").read()
except:
    TRACKING_URI = parsenvy.str("MLFLOW_URI")

EXPERIMENT_NAME = "0-template-ds-modeling"