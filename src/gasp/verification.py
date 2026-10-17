import re
from decimal import Decimal, InvalidOperation


def extract_answer(text):
    answers = re.findall(r"<answer>(.*?)</answer>", text, re.S)
    if answers:
        return answers[-1].strip()
    # Balanced braces, including nested LaTeX fractions.
    boxes = list(re.finditer(r"\\boxed\s*", text))
    if boxes:
        start = boxes[-1].end()
        # MATH contains valid TeX single-token arguments: $\\boxed 2$.
        # Do not turn a malformed/multi-token answer into a last-number guess.
        if start < len(text) and text[start] != "{":
            single = re.match(r"([0-9])(?=\s*(?:\$|\\\)|\\\]|$))", text[start:])
            return single.group(1) if single else None
        start, depth = start + 1, 1
        for i in range(start, len(text)):
            depth += (text[i] == "{") - (text[i] == "}")
            if depth == 0:
                return text[start:i].strip()
    if "####" in text:
        return text.rsplit("####", 1)[-1].strip()
    # Never treat an arbitrary last number in unfinished reasoning as the final answer.
    if re.fullmatch(r"\s*[-+]?\d[\d,]*(?:\.\d+)?\s*", text):
        return text.strip()
    return None


def correct(text, gold, verifier="numeric"):
    candidate = extract_answer(text)
    if candidate is None:
        return 0
    gold = extract_answer(str(gold)) or str(gold).strip()
    if verifier == "math_verify":
        from math_verify import parse, verify
        # Plain numeric answers with unit words are prose, not LaTeX products
        # of letters (e.g. "3 bolts" must not become 3*b*o*l*t*s). Delegate
        # their unit handling to Math-Verify's plain-expression parser. Keep
        # symbolic expressions in math mode rather than extracting a coefficient.
        units = re.fullmatch(r"[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)\s+"
                             r"[A-Za-z]{2,}(?:[ -][A-Za-z]{2,})*\.?", candidate)
        prediction = parse(candidate if units else f"${candidate}$")
        return int(verify(parse(f"${gold}$"), prediction))
    if verifier == "choice":
        return int(candidate.strip().upper() == gold.strip().upper())
    try:
        return int(Decimal(candidate.replace(",", "")) == Decimal(gold.replace(",", "")))
    except InvalidOperation:
        return int(candidate.strip() == gold.strip())


def polluter_reward(agent_correctness):
    if agent_correctness not in (0, 1):
        raise ValueError("Polluter needs a binary correctness reward")
    return 1 - agent_correctness


def reasoning(text):
    if "<think>" in text:
        text = text.split("<think>", 1)[1]
    for marker in ("</think>", "<answer>", "\\boxed{", "####"):
        text = text.split(marker, 1)[0]
    return text.strip()


def visible_completion(raw, prompt=""):
    """Exclude scratchpad text, including a <think> opened by the chat template."""
    if "</think>" in raw:
        visible = raw.rsplit("</think>", 1)[-1]
    elif prompt.rfind("<think>") > prompt.rfind("</think>"):
        return ""
    else:
        visible = raw
    return "" if "<think>" in visible else visible


def pollution(raw, prompt=""):
    """Extract the edited window; prompt constraints are not reward filters.

    Keep boundary whitespace and all visible content. A missing tag is recorded
    for inspection, but its trajectory still receives the binary outcome reward.
    """
    visible = visible_completion(raw, prompt)
    match = re.fullmatch(r"\s*<polluted>(.*?)</polluted>\s*", visible, re.S)
    return (match.group(1) if match else visible), bool(match)


def repair_target(raw, tokenizer, prompt=""):
    """Clone the visible self-generated repair without content/length filters."""
    visible = visible_completion(raw, prompt)
    if not visible.strip():
        return []
    return tokenizer.encode(visible, add_special_tokens=False)
