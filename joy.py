import re

normal_assignment_expr = re.compile(r"^(?!\s*#|\")[a-zA-Z_\D+][a-zA-Z0-9_.\D+]+\s*(?<![-\+\%\/])=\s*[a-zA-Z0-9\"[({_.\D+]+")
typed_assignment_expr  = re.compile(r"^(?!\s*#)[a-zA-Z_\D+][a-zA-Z0-9_.\D+]+\s*:\s*[a-zA-Z_]+\s*=\s*[a-zA-Z0-9\"[({_.\D+]+")

function_arg_assignment_expr  = re.compile(r"\s*[a-zA-Z_][a-zA-Z0-9_]+\s*=\s*[a-zA-Z0-9\"[({_]+")
typed_function_arg_definition = re.compile(r"\s*[a-zA-Z_][a-zA-Z0-9_]+\s*:\s*[a-zA-Z.]+\s*=\s*[a-zA-Z0-9\"[({_]+")

# Note function args assignments can't be typed, except for in function definitions... TODO

docstring_start_expr = re.compile(r"^(?!\s{0,}'{3}.*'''$)\s{0,}[rf]*'{3}|^(?!\s{0,}\"{3}.*\"{3}$)\s{0,}[rf]*\"{3}")
"""Regex for if a line is the start of a multi-line comment like this one.

Matches both single and double quotes. NOTE: does not match single-line triple-quoted comments,
such as single-line docstrings.

Prefixed strings like: r'''foobar''' and f'''baz''' are also allowed, with either double or single quotes.
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
            and not line.strip().startswith(("return ", "assert ", "'", '"'))
            # note 1: the space after e.g 'return' so we only match the keyword, not e.g variables named return.*!
            # note 2: str().strip only strips leading and trailing whitespace; not the whitespace we're checking for.

def align_assignment_expressions(code: str) -> list[str]:
    lines                         = code.split("\n")
    group : list[tuple[int, str]] = []
    inside_multiline_comment      = False  # Are we inside a triple-quotes comment?

    for i, line in enumerate(lines):
        if docstring_start_expr.match(line):
            inside_multiline_comment = not inside_multiline_comment # Flip the value
        elif not inside_multiline_comment and _is_alignable_line(line):
            # Line contains one of the valid assignments and we're not inside a multiline comment.
            group.append((i, line))
        elif group:
            pre_equals_chars          = 0
            max_typed_variable_length = 0
            max_type_hint_length      = 0

            for _, line in group:
                pre_equals   = line.split("=")[0]
                equals_index = len(pre_equals)

                # Compute necessary lengths for handling typed variables
                if (semicolon_index := pre_equals.find(":")) != -1:
                    var_name                  = pre_equals.split(":")[0]
                    var_name_length           = len(var_name.rstrip())
                    max_typed_variable_length = max(max_typed_variable_length, var_name_length)

                    type_hint            = pre_equals[semicolon_index+1:].strip()
                    max_type_hint_length = max(max_type_hint_length, len(type_hint))

                if line[equals_index - 1] != " ":
                    # Add 1 for the space character we're going to add.
                    equals_index += 1
                pre_equals_chars = max(pre_equals_chars, equals_index)

            # the `4` here represents the number of chars other than the type hint and the variable name before
            # the equals character: ``<var_name>_:_<type_hint>_=`` 3 empty spaces (underscores) plus the semicolon.
            pre_equals_chars = max(pre_equals_chars, (max_typed_variable_length + max_type_hint_length + 4))

            for line_index, line in group:
                var_name, value = line.split('=', 1)
                type_hint       = ""
                if ":" in var_name:
                    var_name, type_ = var_name.split(":")
                    type_hint      = ': '+ type_.strip()

                padded_typed_var_name = f"{var_name:<{max_typed_variable_length + 1}}{type_hint}"
                lines[line_index]     = f"{padded_typed_var_name:<{pre_equals_chars}}= {value.strip()}"
                
                group = []  # Empty the list of grouped lines.

    # Create a new string from all the joined strings.
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    file = sys.argv[1] if len(sys.argv) >= 2 else __file__

    # Format the file and save it.
    code           = open(file, "r").read()
    formatted_code = align_assignment_expressions(code)

    with open(file, "w") as f:
        f.write(formatted_code)
