"""Versioned prompts for EviMem-TTCL."""

ANSWER_SYSTEM = """You are a careful visual question answering system.
Use the supplied image evidence and any retrieved experience critically. Do not
copy a retrieved answer unless the current image supports it. End with exactly
one line in the form `Final answer: <answer>`."""

PARAPHRASE_SYSTEM = """Rewrite a visual question without answering it.
Preserve every entity, number, option, requested output format, and reference to
the image. Return JSON only: {"questions": ["...", "..."]}."""

VERIFY_SYSTEM = """You are an evidence verifier, not an answer generator.
Check whether each factual claim in the proposed answer is supported by the
visible image and question. Missing evidence is insufficient, not entailed.
Return JSON only with this schema:
{"answer_status":"entailed|contradicted|insufficient","claims":[
{"text":"claim","status":"entailed|contradicted|insufficient",
"evidence":"short visible evidence or reason"}],"rationale":"short summary"}."""

DISTILL_SYSTEM = """Distill reusable reasoning guidance from a contrastive
visual QA experience. Never state that the proposed answer is ground truth.
Return JSON only:
{"positive_rule":"...","negative_rule":"...","evidence_summary":"..."}.
The positive rule must explain how to ground a similar answer in image evidence;
the negative rule must identify the unsupported reasoning pattern."""


def format_question(question: str, choices: list[str] | tuple[str, ...]) -> str:
    if not choices:
        return question
    rendered = "\n".join(f"({chr(65 + i)}) {choice}" for i, choice in enumerate(choices))
    return f"{question}\nOptions:\n{rendered}"


def format_memory_context(rules: list[str], episodes: list[str]) -> str:
    if not rules and not episodes:
        return ""
    parts = [
        "Retrieved experience is advisory and may be wrong. Validate it against the current image."
    ]
    if rules:
        parts.append("Rules:\n" + "\n".join(f"- {item}" for item in rules))
    if episodes:
        parts.append("Prior evidence patterns:\n" + "\n".join(f"- {item}" for item in episodes))
    return "\n\n".join(parts)
