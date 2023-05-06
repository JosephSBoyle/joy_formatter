import re

normal_assignment_expr = re.compile(r"^(?!\s*#)[a-zA-Z_\D+][a-zA-Z0-9_.\D+]+\s*=\s*[a-zA-Z0-9\"[({_.\D+]+")
typed_assignment_expr  = re.compile(r"^(?!\s*#)[a-zA-Z_\D+][a-zA-Z0-9_.\D+]+\s*:\s*[a-zA-Z_]+\s*=\s*[a-zA-Z0-9\"[({_.\D+]+")

function_arg_assignment_expr  = re.compile(r"\s*[a-zA-Z_][a-zA-Z0-9_]+\s*=\s*[a-zA-Z0-9\"[({_]+")
typed_function_arg_definition = re.compile(r"\s*[a-zA-Z_][a-zA-Z0-9_]+\s*:\s*[a-zA-Z.]+\s*=\s*[a-zA-Z0-9\"[({_]+")

# Note function args assignments can't be typed, except for in function definitions... TODO

regexes = (normal_assignment_expr, typed_assignment_expr, function_arg_assignment_expr, typed_function_arg_definition)

def align_assignment_expressions(code: str) -> list[str]:
    lines                        = code.split("\n")
    group: list[tuple[int, str]] = []

    for i, line in enumerate(lines):
        if any(regex.match(line) for regex in regexes):
            # Line contains one of the valid assignments.
            if not line.endswith(":"):
                group.append((i, line))
        elif group:
            pre_equals_chars = max(
                (line.find("=") + (line[line.find("=") - 1] != " ") if "=" in line else -1)
                for _, line in group
            )

            # pre_equals_chars          = max(line.find("=") for _, line in group)
            max_typed_variable_length = max(line.find(":") for _, line in group)

            max_type_hint_length = max((len(line.split("=", 1)[0].split(":", 1)[1].strip()) for _, line in group if ":" in line),
                                        default=-2)
            pre_equals_chars     = max(pre_equals_chars,  # The +3 is the number of spaces in `var_name : type_hint =`
                                       (max_typed_variable_length + max_type_hint_length + 4))
            
            for i, line in group:
                var_name, value = line.split('=', 1)
                if ":" in var_name:
                    var_name, type = var_name.split(":")
                else:
                    type = False
                
                type_hint = f"{': '+ type.strip() if type else ''}"
                padded_typed_var_name = f"{var_name:<{max_typed_variable_length + 1}}{type_hint}"
                lines[i] = f"{padded_typed_var_name:<{pre_equals_chars}}= {value.strip()}"
                
                group = []  # Clear the grouped lines.

    # Create a new string from all the joined strings.
    return "\n".join(lines)

if __name__ == "__main__":
    filename = "t.py"
    code = open(filename, "r").read()
    print(align_assignment_expressions(code))