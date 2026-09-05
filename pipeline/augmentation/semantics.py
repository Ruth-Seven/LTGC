"""Class semantics kept separate from coarse father-pool categories."""
import re
from .records import ImageReference, Sample
from utils import parse_semantic_label


def semantic_label(name: str, fallback: str, description: str) -> str:
    """Recover the fine-grained sense written in the original Step1 caption."""
    pattern = re.compile(
        re.escape(name) + r"\s*\((?P<sense>[^()]+)\)", re.IGNORECASE)
    match = pattern.search(description or "")
    if not match:
        return fallback
    return f"{name} ({match.group('sense').strip()})"


def sample_semantic_label(sample: Sample) -> str:
    return semantic_label(
        sample.name, sample.class_name, sample.source_prompt)


def reference_semantic_label(reference: ImageReference) -> str:
    name, _, _ = parse_semantic_label(reference.class_name)
    description = reference.descriptions[0] if reference.descriptions else ""
    return semantic_label(name, reference.class_name, description)


def replace_label(description: str, name: str, replacement: str) -> str:
    pattern = re.compile(
        re.escape(name) + r"\s*\([^()]+\)", re.IGNORECASE)
    return pattern.sub(replacement, description, count=1)


def sample_semantic_description(sample: Sample, description: str) -> str:
    return replace_label(
        description, sample.name, sample_semantic_label(sample))


def reference_semantic_description(reference: ImageReference) -> str:
    name, _, _ = parse_semantic_label(reference.class_name)
    description = reference.descriptions[0] if reference.descriptions else ""
    return replace_label(
        description, name, reference_semantic_label(reference))


def clip_text(sample: Sample) -> str:
    return f"A photo of {sample_semantic_label(sample)}"
