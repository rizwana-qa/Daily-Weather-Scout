"""
Development-time generator and checker for STATIC_TOKEN_TABLE.

weather_log.py counts static request text (chat template, system prompt,
instructions, response schema and every policy block) from a table of
token counts keyed by the SHA-256 of the exact text. This tool produces
that table with the o200k_harmony tokenizer, the gpt-oss tokenizer.

It needs tiktoken (pip install tiktoken). tiktoken is NOT a runtime
dependency of weather_log.py and the workflow never installs it.

Usage, from the repository root (-B keeps bytecode out of the repository):
  python -B tools/token_table.py --print   print the block to paste between
                                           the BEGIN/END STATIC_TOKEN_TABLE
                                           markers in weather_log.py
  python -B tools/token_table.py --check   exit 1 if the table in
                                           weather_log.py is stale or wrong,
                                           or if the estimator pins differ
                                           from the request configuration

Maintenance. Whenever pinned static text changes (the system prompt, the
user preamble or any instruction, the response schema, a heading or section
in guidelines.md, the chat template strings) or the pinned model or
reasoning effort changes:
  1. Run --print and paste the block between the markers.
  2. Run --check. The table is not valid until it prints OK and exits 0.
  3. Run the suite with WEATHER_TOKENIZER_TESTS=1 and tiktoken installed.
A skipped tokenizer-dependent test (the suite prints SKIP without them) is
not a successful tokenizer validation; only --check, or those groups
running to PASS, validates the table.

weather_log.py refuses to send a request whose static text is missing from
the table, so a stale table fails closed. It cannot tell a wrong count from
a right one, and a pin mismatch falls back to the legacy estimate: --check
is the control for both.
"""
import contextlib
import datetime
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tiktoken  # noqa: E402

import weather_log  # noqa: E402

ENCODING = "o200k_harmony"
# o200k_harmony is the gpt-oss tokenizer: counts made with it describe only
# these models, so a pin naming any other model must fail the check.
ENCODING_MODEL_PREFIXES = ("openai/gpt-oss-",)
META_KEYS = {"encoding", "tokenizer", "entries"}
encoding = tiktoken.get_encoding(ENCODING)


def count(text):
    return len(encoding.encode(text, allowed_special="all"))


def entries():
    with contextlib.redirect_stdout(io.StringIO()):
        policy = weather_log.load_guidelines()

    rows = []

    for label, text in weather_log.static_token_texts(policy):
        rows.append((label, weather_log.sha256_text(text), count(text)))

    return rows


def table_block(rows):
    lines = [
        "STATIC_TOKEN_TABLE_META = {",
        f'    "encoding": "{ENCODING}",',
        f'    "tokenizer": "tiktoken {tiktoken.__version__}",',
        f'    "entries": {len(rows)},',
        "}",
        "",
        "STATIC_TOKEN_TABLE = {",
    ]

    for label, digest, tokens in rows:
        lines.append(f"    # {label}")
        lines.append(f'    "{digest}": {tokens},')

    lines.append("}")

    return "\n".join(lines)


def pin_problems():
    """Compare the estimator pins with the request configuration in use."""
    problems = []
    pins = weather_log.ESTIMATOR_PINS
    configured = {
        "model": weather_log.MODEL,
        "reasoning_effort": weather_log.REASONING_EFFORT,
        "response_format_sha256": weather_log.sha256_text(
            weather_log.json_text(weather_log.RESPONSE_FORMAT)
        ),
    }

    if set(pins) != set(configured):
        problems.append(
            f"estimator pins {sorted(pins)} differ from the pins this "
            f"checker compares {sorted(configured)}"
        )

    for name, value in configured.items():
        if pins.get(name) != value:
            problems.append(
                f"estimator pin {name} differs: pinned {pins.get(name)!r}, "
                f"configured {value!r}"
            )

    # weather_log.py's own decision between the component and the legacy
    # estimate must agree with the comparison above.
    try:
        pins_match = weather_log.estimator_pins_match()
    except Exception as exc:
        problems.append(f"weather_log.estimator_pins_match() raised: {exc!r}")
    else:
        if not pins_match:
            problems.append(
                "weather_log.estimator_pins_match() is False: requests fall "
                "back to the legacy estimate"
            )

    if not str(pins.get("model")).startswith(ENCODING_MODEL_PREFIXES):
        problems.append(
            f"pinned model {pins.get('model')!r} is not counted by the "
            f"{ENCODING} tokenizer"
        )

    if f"Reasoning: {pins.get('reasoning_effort')}\n" not in (
        weather_log.HARMONY_SYSTEM_TEXT
    ):
        problems.append(
            "the chat template's Reasoning line differs from the pinned "
            "reasoning effort"
        )

    return problems


def check(rows):
    problems = []
    expected = {digest: tokens for _, digest, tokens in rows}
    actual = weather_log.STATIC_TOKEN_TABLE
    meta = weather_log.STATIC_TOKEN_TABLE_META
    labels = {digest: label for label, digest, _ in rows}

    if len(expected) != len(rows):
        problems.append("two static texts share one SHA-256")

    for digest, tokens in expected.items():
        if digest not in actual:
            problems.append(f"missing: {labels[digest]}")
        elif actual[digest] != tokens:
            problems.append(
                f"wrong count for {labels[digest]}: table {actual[digest]}, "
                f"tokenizer {tokens}"
            )

    for digest in actual:
        if digest not in expected:
            problems.append(f"stale entry {digest[:12]}")

    # The table must also pass the check weather_log.py runs before every
    # request (shape, types, positive counts, entry count, encoding).
    try:
        weather_log.verify_static_token_table()
    except weather_log.RequestBudgetError as exc:
        problems.append(f"weather_log.py rejects the table: {exc}")

    if set(meta) != META_KEYS:
        problems.append(f"table metadata keys differ: {sorted(meta)}")

    if meta.get("encoding") != ENCODING:
        problems.append("table encoding differs")

    if meta.get("entries") != len(actual):
        problems.append("table entry count differs from the table")

    problems.extend(pin_problems())

    # The chat template date must not change the count.
    template = weather_log.HARMONY_SYSTEM_TEXT
    day = datetime.date(2026, 1, 1)
    worst = count(template)

    while day <= datetime.date(2035, 12, 31):
        dated = template.replace("2026-11-09", day.isoformat())
        worst = max(worst, count(dated))
        day += datetime.timedelta(days=1)

    if worst != count(template):
        problems.append(
            f"the date in the chat template changes its token count ({worst})"
        )

    if meta.get("tokenizer") != f"tiktoken {tiktoken.__version__}":
        print(
            "note: table generated with", meta.get("tokenizer"),
            "- checking with tiktoken", tiktoken.__version__,
        )

    return problems


def main(argv):
    mode = argv[1] if len(argv) > 1 else ""
    rows = entries()

    if mode == "--print":
        print(table_block(rows))
        return 0

    if mode == "--check":
        problems = check(rows)

        for problem in problems:
            print("PROBLEM:", problem)

        pins = weather_log.ESTIMATOR_PINS

        print(
            "static token table and estimator pins:",
            "OK" if not problems else f"{len(problems)} problem(s)",
            f"({len(rows)} entries, encoding {ENCODING}, model "
            f"{pins.get('model')}, reasoning effort "
            f"{pins.get('reasoning_effort')}, response format sha256 "
            f"{str(pins.get('response_format_sha256'))[:12]})",
        )
        return 1 if problems else 0

    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
