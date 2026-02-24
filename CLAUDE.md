# Joy Formatter

Assignment-alignment formatter for Python. Designed to run **after** a primary formatter like ruff or black.

## What it does

Aligns adjacent assignment expressions so `=` signs line up vertically. Also aligns type hints in function signatures. Does not touch code inside docstrings.

## Architecture

Single-file, regex-based formatter (`joy.py`). No dependencies beyond the standard library.

- **Regex patterns** (top of file): detect alignable lines (assignments, typed assignments, function args, type hints)
- `_is_alignable_line()`: classifies whether a line should be aligned; uses conservative heuristics (skips `==`, decorators, returns, multiline openers)
- `_handle_assignment_group()`: aligns a contiguous block of same-indentation assignments by padding variable names. **Mutates** the lines list in place.
- `align_assignment_expressions()`: main pipeline — splits code into lines, tracks docstring context, groups consecutive alignable lines, delegates to `_handle_assignment_group`
- `main()`: CLI entry point. With args: reads/writes files in place (pre-commit mode). Without args: stdin/stdout (VSCode custom formatter mode).

## Invocation modes

| Mode | Command |
|------|---------|
| CLI (files) | `joy file1.py file2.py` |
| stdin/stdout | `python joy.py < input.py` |
| Pre-commit | hook id `align-assignments` |
| VSCode | Custom Local Formatters extension (see below) |

## VSCode integration (chained formatting)

To run ruff/black first, then joy, use `format_chain.sh` as the custom formatter command:

```json
"customLocalFormatters.formatters": [
    {
        "command": "/home/extasia/Desktop/joy_formatter/format_chain.sh",
        "languages": ["python"]
    }
],
"[python]": {
    "editor.defaultFormatter": "jkillian.custom-local-formatters"
}
```

The script pipes stdin through `ruff format`, then through `joy.py`. Edit the script to swap ruff for black if preferred.

## Running tests

```bash
./rt           # shorthand for pytest
./rt -v        # verbose
./rt -k test_base  # specific test
```

## Code style conventions

- Functional style; no classes
- Regex patterns compiled at module level
- Type hints use modern syntax: `list[str]`, `tuple[int, str]`
- Walrus operator (`:=`) used for inline assignments
- f-strings for all string formatting
- Tests use plain `assert` (no unittest classes)
- `# HACK` and `### TODO ###` comments mark known technical debt
- Formatting is idempotent — running joy twice produces the same output
- Zero external dependencies
