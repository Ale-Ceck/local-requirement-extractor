import json
from pathlib import Path
from typing import Optional

from config.schema import OutputConfig
from src.data_models.requirement import RequirementList
from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)


class ReviewArtifactWriter:
    """Write a machine-readable review artifact with full provenance."""

    def __init__(self, config: OutputConfig):
        self.config = config

    def write(self, requirement_list: RequirementList, artifact_path: Optional[str] = None) -> str:
        if artifact_path is None:
            artifact_path = str(Path(self.config.directory) / self.config.review_artifact_filename)
        write_review_artifact(requirement_list, artifact_path)
        return artifact_path


def write_review_artifact(requirement_list: RequirementList, artifact_path: str) -> None:
    if not isinstance(requirement_list, RequirementList):
        raise ValueError("requirement_list must be a RequirementList instance")

    output_path = Path(artifact_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "requirements": [requirement.model_dump(mode="json") for requirement in requirement_list],
    }

    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Successfully wrote review artifact to %s", output_path)
