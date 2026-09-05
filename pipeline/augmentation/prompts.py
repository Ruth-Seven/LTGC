"""Pure prompt construction, separate from model invocation and persistence."""
import re
from utils import parse_semantic_label
from .semantics import reference_semantic_label, sample_semantic_label

def mapped_description(description, class_name):
    """Use the selected parent taxonomy for LLM input, retaining raw Step1 captions on disk."""
    name, _, display = parse_semantic_label(class_name)
    return re.sub(re.escape(name) + r"\s*\([^()]+\)", lambda _: display,
                  description, count=1, flags=re.IGNORECASE)


def kontext_prompt(row):
    return (f"Use the supplied image as the visual reference for {row['class_name']}. "
            f"The subject must remain a {row['name']} in the parent category {row['parent_category']}. "
            "Create a photorealistic variation matching this description, preserving the defining "
            "class characteristics while applying the described instance details, action and scene: "
            + row["extended_prompt"])


def parent_context_edit_prompt(row):
    context = row["prompt_instance"]
    return (f"Edit image 1, whose subject is {row['class_name']}. "
        f"Image 2 shows {context['class_name']}, another class in the shared parent category {row['parent_category']}. "
        f"Keep the output subject a {row['class_name']} and preserve its defining class features from image 1. "
        "Use image 2 only as additional context for physically compatible actions, composition or setting. "
        "Do not copy its class-specific anatomy, shape or surface features, do not replace the target subject, "
        "and do not make a collage. The target edit below takes priority over conflicting context. "
        f"Target edit: {row['extended_prompt']} "
        f"Additional context description for image 2: {context['descriptions'][0]} "
        "Produce one photorealistic image.")


def short_parent_context_edit_prompt(row):
    return ("Keep the subject from image 1. "
        "Use image 2 only for background and lighting; exclude its subject entirely. "
        f"Target: {row['extended_prompt']} "
        f"Context: {row['context_scene']}.")


def _compose_edit_prompt(row, short, gen_cfg):
    """Compose the flux-klein two-reference edit prompt from config templates."""
    context = row["prompt_instance"]
    if short:
        template = gen_cfg.get("klein_edit_prompt_short")
        if template:
            return template.format(
                target=row["extended_prompt"], context_scene=row["context_scene"]
            )
        return short_parent_context_edit_prompt(row)
    template = gen_cfg.get("klein_edit_prompt_full")
    if template:
        return template.format(
            class_name=row["class_name"],
            context_class=context["class_name"],
            category=row["parent_category"],
            target=row["extended_prompt"],
            context_description=context["descriptions"][0],
        )
    return parent_context_edit_prompt(row)


def generation_prompt(record, config, templates):
    row = record.to_dict()
    row["class_name"] = sample_semantic_label(record.sample)
    row["prompt_instance"]["class_name"] = reference_semantic_label(
        record.sample.father)
    return (_compose_edit_prompt(
        row, config.prompt_style == 'short', templates)
        if config.two_references else kontext_prompt(row))
