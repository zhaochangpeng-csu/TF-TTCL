"""Versioned prompts for EviMem-TTCL."""

ANSWER_SYSTEM = """You are a careful visual question answering system.
Use the supplied image evidence and any retrieved experience critically. Do not
copy a retrieved answer unless the current image supports it. End with exactly
one line in the form `Final answer: <answer>`."""

PARAPHRASE_SYSTEM = """Rewrite only the supplied question clause without
answering it. The surrounding context and answer options are intentionally not
included and will be restored verbatim, so do not invent or summarize either.
Preserve every entity, number, negation, comparison, requested output format,
and reference to the image. Return JSON only:
{"questions": ["...", "..."]}."""

VERIFY_SYSTEM = """You are an evidence verifier, not an answer generator.
Check whether each factual claim in the proposed answer is supported by the
visible image and question. Missing evidence is insufficient, not entailed.
Return JSON only with this schema:
{"answer_status":"entailed|contradicted|insufficient","claims":[
{"text":"claim","status":"entailed|contradicted|insufficient",
"evidence":"short visible evidence or reason"}],"rationale":"short summary"}."""

CONTRASTIVE_VERIFY_SYSTEM = """You are a contrastive multimodal evidence
adjudicator, not an answer generator. Compare exactly RESPONSE_X and RESPONSE_Y
against the supplied image, question, context, and options. Treat an absent
visual fact as insufficient rather than inventing it. Ignore how frequently an
answer was proposed and do not favor response position. Choose a response
only when the available evidence entails it and contradicts or leaves the
other candidate insufficient; otherwise choose undecidable. Return JSON only:
{"winner":"response_x|response_y|undecidable",
"response_x_status":"entailed|contradicted|insufficient",
"response_y_status":"entailed|contradicted|insufficient",
"decisive_evidence":["short image or question evidence"],
"counterevidence":["short contradiction or missing-evidence statement"],
"rationale":"short comparison"}. Include every key. If exactly one candidate
is entailed and the other is contradicted or insufficient, winner MUST name the
entailed candidate. decisive_evidence and counterevidence MUST be JSON arrays,
even when empty."""

DISTILL_SYSTEM = """Distill reusable reasoning guidance from a contrastive
visual QA experience. Never state that the proposed answer is ground truth.
Return JSON only:
{"positive_rule":"...","negative_rule":"...","evidence_summary":"..."}.
The rules must transfer even if every noun, adjective, category, color, number,
and named entity in the example changes. Replace example-specific criteria with
generic variables such as "the requested categories". For example, do not write
"subtract large rubber spheres"; write "enumerate the requested removal sets,
take their union to avoid double counting, then subtract it from the total".
The positive rule must explain how to ground a similar answer in image evidence.
Emit an empty negative_rule when no explicit
contradicted candidate is supplied; otherwise identify only its unsupported
reasoning pattern. The evidence_summary may describe this image specifically."""


def format_question(
    question: str,
    choices: list[str] | tuple[str, ...],
    metadata: dict | None = None,
) -> str:
    metadata = metadata or {}
    parts = [question]
    if choices:
        rendered = "\n".join(f"({chr(65 + i)}) {choice}" for i, choice in enumerate(choices))
        parts.append(f"Options:\n{rendered}")
    if str(metadata.get("question_type", "")).lower() == "multi_choice":
        parts.append("Put only the correct option letter after `Final answer:`.")
    elif str(metadata.get("answer_type", "")).lower() == "integer":
        parts.append("Put only the integer value, without units, after `Final answer:`.")
    elif str(metadata.get("answer_type", "")).lower() == "float":
        precision = int(float(metadata.get("precision", 2)))
        parts.append(
            f"Put only the numeric value rounded to {precision} decimal places, without units, after `Final answer:`."
        )
    elif str(metadata.get("answer_type", "")).lower() == "list":
        parts.append("Put only a Python-style list after `Final answer:`.")
    return "\n".join(parts)


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
