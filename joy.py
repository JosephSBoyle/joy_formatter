import re

normal_assignment_expr = re.compile(r"^(?!\s*#|\")[a-zA-Z_\D+][a-zA-Z0-9_.\D+]+\s*(?<![-\+\%\/])=\s*[a-zA-Z0-9\"[({_.\D+]+")
typed_assignment_expr  = re.compile(r"^(?!\s*#)[a-zA-Z_\D+][a-zA-Z0-9_.\D+]+\s*:\s*[a-zA-Z_]+\s*=\s*[a-zA-Z0-9\"[({_.\D+]+")

function_arg_assignment_expr  = re.compile(r"\s*[a-zA-Z_][a-zA-Z0-9_]+\s*=\s*[a-zA-Z0-9\"[({_]+")
typed_function_arg_definition = re.compile(r"\s*[a-zA-Z_][a-zA-Z0-9_]+\s*:\s*[a-zA-Z.]+\s*=\s*[a-zA-Z0-9\"[({_]+")

type_hint_expression = re.compile(r"^(?!\s*#)[a-zA-Z_\D+][a-zA-Z0-9_.\D+]+\s*:\s*[a-zA-Z_]+\s*")  # e.g foo: bar

# Note function args assignments can't be typed, except for in function definitions... TODO

docstring_start_expr = re.compile(r"^(?!\s{0,}'{3}.*'''$)\s{0,}[rf]*'{3}|^(?!\s{0,}\"{3}.*\"{3}$)\s{0,}[rf]*\"{3}")
"""Regex for if a line is the start of a multi-line comment like this one.

Matches both single and double quotes. NOTE: does not match single-line triple-quoted comments,
such as single-line docstrings.

Prefixed strings like: r'''foobar''' and f'''baz''' are also allowed, with either double or single quotes.
"""

alignable_expressions = (
    normal_assignment_expr,
    typed_assignment_expr,
    function_arg_assignment_expr,
    typed_function_arg_definition,
    type_hint_expression,
)

# e.g `foo(bar=baz)`
call_with_named_argument = re.compile(r"\s*[a-zA-Z_][a-zA-Z0-9_\s.\,\(\)]+\([^\(]*=.*\)")


def _is_alignable_line(line: str) -> bool:
    """Return `True` if the line contains an alignable expression.
    
    Examples of valid lines are variable assignments as well as default function arguments, provided they are on their own line.
        `x = y`
        `x: int = y`
    """
    ### TODO REFACTOR ME ###
    if not any(expr.match(line) for expr in alignable_expressions):
        # The line isn't an assignment so we don't care about it!
        return False

    if line.endswith((":", "(", "{")):
        # ':' means we're on the conditional line of an 'if' block.
        # '(' means we're at the start of a multiline function call or object instantiation!
        # '{' means we're at the start of a multiline dictionary instantiation.
        return False
    
    if line.strip().startswith(("assert ", "'", '"')):
        # line.strip only strips leading and trailing whitespace; not the whitespace we're checking for.
        return False
    
    if call_with_named_argument.match(line):
        return False

    return True

def _handle_assignment_group(lines: list[str], group: list[tuple[int, str]]): # TODO type hinting
    """Align assignments for a group of assignments on adjacent lines.

    Warning - mutates `lines`!
    """
    pre_equals_chars          = 0
    max_typed_variable_length = 0
    max_type_hint_length      = 0

    for _, line in group:
        if "=" not in line:
            # Edge case: we're in a non-assignment type hint expression.
            # e.g ``foo: bar`` as opposed to ``foo: bar = ...``
            pre_equals = line
        else:
            pre_equals = line.split("=")[0]

        equals_index = len(pre_equals)

        # Compute necessary lengths for handling typed variables
        if (semicolon_index := pre_equals.find(":")) != -1:
            var_name                  = pre_equals[:semicolon_index]
            var_name_length           = len(var_name.rstrip())
            max_typed_variable_length = max(max_typed_variable_length, var_name_length)

            type_hint = pre_equals[semicolon_index+1:].strip()

            # HACK - resolve with better regex match for 'untyped assignment'
            if "]" in type_hint:
                type_hint = ""

            max_type_hint_length = max(max_type_hint_length, len(type_hint))

        if line[equals_index - 1] != " ":
            # Add 1 for the space character we're going to add.
            equals_index += 1
        pre_equals_chars = max(pre_equals_chars, equals_index)

    # the `4` here represents the number of chars other than the type hint and the variable name before
    # the equals character: ``<var_name>_:_<type_hint>_=`` 3 empty spaces (underscores) plus the semicolon.
    if max_type_hint_length:
        max_type_hint_length += 4

    type_hint_pre_equals_chars = max_typed_variable_length + max_type_hint_length
    pre_equals_chars           = max(pre_equals_chars, type_hint_pre_equals_chars)

    for line_index, line in group:
        if "=" in line:
            var_name, value = line.split('=', 1)
            type_hint       = ""
            
            # HACK - remove RHS expression with better regex match for 'untyped assignment'
            if ":" in var_name and ("[" not in var_name):
                var_name, type_ = var_name.split(":")
                type_hint       = ': '+ type_.strip()

            padded_typed_var_name = f"{var_name:<{max_typed_variable_length + 1}}{type_hint}"
            lines[line_index]     = f"{padded_typed_var_name:<{pre_equals_chars}}= {value.strip()}"
        else:
            # Edge case: we're in a non-assignment type hint expression.
            var_name, type_hint = line.split(":")
            lines[line_index]   = f"{var_name:<{max_typed_variable_length + 1}}: {type_hint.strip()}"
        
def align_assignment_expressions(code: str) -> str:
    lines                         = code.split("\n")
    group : list[tuple[int, str]] = []
    inside_multiline_comment      = False  # Are we inside a triple-quotes comment?

    for i, line in enumerate(lines):
        if docstring_start_expr.match(line):
            # We're entering a multiline comment or leaving one.
            # Either way, we flip the value of this variable.
            inside_multiline_comment = not inside_multiline_comment
        elif not inside_multiline_comment and _is_alignable_line(line):
            # Line contains one of the valid assignments and we're not inside a multiline comment.

            # Check that the indentation of the line is the same as the rest of the group.
            # If it's not, handle that group and we'll start a new one
            if group and \
                (len(group[-1][1]) - len(group[-1][1].lstrip())) != (len(line) - len(line.lstrip())):
                
                _handle_assignment_group(lines, group)
                group = []

            group.append((i, line))
            
        elif group:
            _handle_assignment_group(lines, group)
            group = []

    if group:
        # Edge case.
        # Handle the last assignment group - only occurs if the group is at the very
        # end of the file, and the file has no additional e.g empty lines at the end.
        #
        # PEP-8 specifies that this shouldn't happen though;)
        _handle_assignment_group(lines, group)
        group = []

    # Create a new string from all the joined strings.
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    lines = sys.stdin.read()
    # Printing to stdout is how the vscode extension gets your formatted code back.
    print(align_assignment_expressions(lines), end="")
