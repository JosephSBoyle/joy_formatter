import re

normal_assignment_expr = re.compile(r"^(?!\s*#|\")[a-zA-Z_\D+][a-zA-Z0-9_.\D+]+\s*=\s*[a-zA-Z0-9\"[({_.\D+]+")
typed_assignment_expr  = re.compile(r"^(?!\s*#)[a-zA-Z_\D+][a-zA-Z0-9_.\D+]+\s*:\s*[a-zA-Z_]+\s*=\s*[a-zA-Z0-9\"[({_.\D+]+")

function_arg_assignment_expr  = re.compile(r"\s*[a-zA-Z_][a-zA-Z0-9_]+\s*=\s*[a-zA-Z0-9\"[({_]+")
typed_function_arg_definition = re.compile(r"\s*[a-zA-Z_][a-zA-Z0-9_]+\s*:\s*[a-zA-Z.]+\s*=\s*[a-zA-Z0-9\"[({_]+")

# Note function args assignments can't be typed, except for in function definitions... TODO

multiline_comment_start_expr = re.compile(r"^(?!^\"\"\".*\"\"\"$)\s{0,}\"\"\"|'''")
"""Regex for if a line is the start of a multi-line comment like this one.

Matches both single and double quotes. NOTE: does not match single-line triple-quoted comments,
such as single-line docstrings.
"""

assignment_expressions = (normal_assignment_expr, typed_assignment_expr, function_arg_assignment_expr, typed_function_arg_definition)

def _is_alignable_line(line: str) -> bool:
    """Return `True` if the line contains an alignable expression.
    
    Examples of valid lines are variable assignments as well as default function arguments, provided they are on their own line.
        `x = y`
        `x: int = y`
    
    """
    return any(expr.match(line) for expr in assignment_expressions) \
        and not line.endswith(":") \
            and not line.strip().startswith(("return ", "assert "))
            # note the space after e.g 'return' so we only match the keyword, not e.g variables named return.*!

def align_assignment_expressions(code: str) -> list[str]:
    lines                        = code.split("\n")
    group: list[tuple[int, str]] = []
    inside_multiline_comment     = False  # Are we inside a triple-quotes comment?

    for i, line in enumerate(lines):
        if multiline_comment_start_expr.match(line):
            inside_multiline_comment = not inside_multiline_comment # Flip the value
        elif not inside_multiline_comment and _is_alignable_line(line):
            # Line contains one of the valid assignments and we're not inside a multiline comment.
            group.append((i, line))
        elif group:
            pre_equals_chars = max(
                (line.find("=") + (line[line.find("=") - 1] != " ") if "=" in line else -1)
                for _, line in group
            )

            max_typed_variable_length = max(line.split("=")[0].find(":") for _, line in group)
            max_type_hint_length      = 0

            for _, line in group:
                # Check if there's a colon before an equals symbol
                # (i.e type hint in a variable assignment or default function arg.)
                if (colon_idx := line.find(":")) != -1 and (equals_idx := line.find("=", colon_idx)) != -1:
                    # colon_idx +1 because the first char of the type hint is the one _after_ the colon
                    hint = line[colon_idx + 1 : equals_idx].strip()
                    max_type_hint_length = max(max_type_hint_length, len(hint))

            pre_equals_chars = max(pre_equals_chars,  # The +3 is the number of spaces in `var_name : type_hint =`
                                   (max_typed_variable_length + max_type_hint_length + 4))

            for line_index, line in group:
                var_name, value = line.split('=', 1)
                if ":" in var_name:
                    var_name, type = var_name.split(":")
                else:
                    type = False
                
                type_hint = f"{': '+ type.strip() if type else ''}"
                padded_typed_var_name = f"{var_name:<{max_typed_variable_length + 1}}{type_hint}"
                lines[line_index]     = f"{padded_typed_var_name:<{pre_equals_chars}}= {value.strip()}"
                
                group = []  # Clear the grouped lines.

    # Create a new string from all the joined strings.
    return "\n".join(lines)

if __name__ == "__main__":
    filename = "t.py"
    code = open(filename, "r").read()
    print(align_assignment_expressions(code))