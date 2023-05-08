from pytest import mark

from joy import align_assignment_expressions


def test_base():
    code = \
"""
foo = "123"
bar_bar = 111
"""

    assert align_assignment_expressions(code) == \
"""
foo     = "123"
bar_bar = 111
"""

def test_valid_right_side_expressions():
    code = \
"""
foo = [1, ]
barbar = {}
baz_baz = _MyPrivateType()
"""
    assert align_assignment_expressions(code) == \
"""
foo     = [1, ]
barbar  = {}
baz_baz = _MyPrivateType()
"""

def test_type_hint():
    code = \
"""
li: list = [1, 2, 3]
tup: tuple = (3, 2, 1)
"""
    assert align_assignment_expressions(code) == \
"""
li  : list  = [1, 2, 3]
tup : tuple = (3, 2, 1)
"""

def test_type_hint_with_untyped():
    code = \
"""
li: list = [1, 2, 3]
tup: tuple = (3, 2, 1)
untyped = 7
"""
    assert align_assignment_expressions(code) == \
"""
li  : list  = [1, 2, 3]
tup : tuple = (3, 2, 1)
untyped     = 7
"""

def test_custom_type_hint():
    code = \
"""
my_long_variable_name: list = [1, 2, 3]
tup: tuple = (3, 2, 1)
untyped_var = 7
"""
    assert align_assignment_expressions(code) == \
"""
my_long_variable_name : list  = [1, 2, 3]
tup                   : tuple = (3, 2, 1)
untyped_var                   = 7
"""

def test_custom_type_hint():
    code = \
"""
my_long_variable_name: tuple  = (1, 2, 3)
tup: CustomTuple = (3, 2, 1)
untyped_var = 7
"""
    assert align_assignment_expressions(code) == \
"""
my_long_variable_name : tuple       = (1, 2, 3)
tup                   : CustomTuple = (3, 2, 1)
untyped_var                         = 7
"""

def test_multiline_function_args():
    code = \
"""
my_func(
    my_long_variable_name={1, 2, 3},
    var_name=_MyCustomType(),
    untyped_var= 7,
)
"""
    assert align_assignment_expressions(code) == \
"""
my_func(
    my_long_variable_name = {1, 2, 3},
    var_name              = _MyCustomType(),
    untyped_var           = 7,
)
"""

def test_method_definition_with_default_values():
    code = \
"""
    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
"""
    assert align_assignment_expressions(code) == \
"""
    def forward(
        self,
        input_ids : torch.Tensor = None,
        attention_mask           = None,
        token_type_ids           = None,
        position_ids             = None,
        head_mask                = None,
        inputs_embeds            = None,
        labels                   = None,
        output_attentions        = None,
        output_hidden_states     = None,
        return_dict              = None,
    ):
"""

def test_attribute_assignment():
    code = \
"""
x.foo = [1, ]
yy.bar = {}
zzz.baz_baz = _MyPrivateType()
"""
    assert align_assignment_expressions(code) == \
"""
x.foo       = [1, ]
yy.bar      = {}
zzz.baz_baz = _MyPrivateType()
"""

def test_typed_attribute_assignment():
    code = \
"""
x.foo: list = [1, ]
yy.bar: dict = {}
zzz.baz_baz: _MyPrivateType = _MyPrivateType()
"""
    assert align_assignment_expressions(code) == \
"""
x.foo       : list           = [1, ]
yy.bar      : dict           = {}
zzz.baz_baz : _MyPrivateType = _MyPrivateType()
"""

def test_attribute_and_variable_assignment_on_adjacent_lines():
    code = \
"""
siu_siu = (_MyPrivateType(), tuple())
x.foo = [1, ]
"""
    assert align_assignment_expressions(code) == \
"""
siu_siu = (_MyPrivateType(), tuple())
x.foo   = [1, ]
"""

def test_weird_characters():
    code = \
"""
x.ñ = ñ
ŷ3 = self._conditioning(ŷ2)
loss: ñ = torch.binary_cross_entropy_with_logits(ŷ3, labels)
"""
    assert align_assignment_expressions(code) == \
"""
x.ñ      = ñ
ŷ3       = self._conditioning(ŷ2)
loss : ñ = torch.binary_cross_entropy_with_logits(ŷ3, labels)
"""

def test_commented_adjacent_line():
    static_code = \
"""
# commented_variable_assignment = 1
variable = loss.sum()
"""
    assert align_assignment_expressions(static_code) == static_code

def test_indented_commented_adjacent_line():
    static_code = \
"""
    # commented_variable_assignment = 1
    variable = loss.sum()
"""
    assert align_assignment_expressions(static_code) == static_code

def test_commented_and_typed_adjacent_line():
    static_code = \
"""
# commented_variable_assignment: int = 1
variable = loss.sum()
"""
    assert align_assignment_expressions(static_code) == static_code

def test_commented_typed_function_arg_definition():
    static_code = \
"""
def f(
    # commented_variable_assignment: int = 1,
    variable = loss.sum()
)
"""
    assert align_assignment_expressions(static_code) == static_code

def test_conditional_line():
    static_code = \
"""
if not torch.all((input_ids1 >= 0) & (input_ids1 <= self._num_embeddings)):
    pass
"""
    assert align_assignment_expressions(static_code) == static_code

def test_conditional_line_2():
    static_code = \
"""
if True:
    x: int = y
"""
    assert align_assignment_expressions(static_code) == \
"""
if True:
    x : int = y
"""

def test_ignore_anything_in_triple_quoted_comment_double_quotes():
    static_code = \
"""
'''
This is a docstring.

    :arg  football:
    :type football: sphere

Something = some thing
Something_ = some other thing
'''
"""
    assert align_assignment_expressions(static_code) == static_code

def test_ignore_anything_in_triple_quoted_comment_single_quotes():
    static_code = \
'''
"""
This is a docstring.

    :arg  football:
    :type football: sphere

Something = some thing
Something_ = some other thing
"""
'''
    assert align_assignment_expressions(static_code) == static_code

def test_docstring_first_line_assignment():
    static_code = \
'''
code = True
"""looks_like_variable = True
"""
'''
    assert align_assignment_expressions(static_code) == static_code

def test_single_line_docstring():
    static_code = \
'''
code = True
"""looks_like_variable = True"""
foo = True
foobar = True
'''
    assert align_assignment_expressions(static_code) == \
'''
code = True
"""looks_like_variable = True"""
foo    = True
foobar = True
'''

def test_single_line_docstring_indented():
    static_code = \
'''
    code = True
    """looks_like_variable = True"""
    foo = True
    foobar = True
'''
    assert align_assignment_expressions(static_code) == \
'''
    code = True
    """looks_like_variable = True"""
    foo    = True
    foobar = True
'''

def test_single_line_docstring_triple_single_quotes():
    static_code = \
"""
code = True
'''looks_like_variable = True'''
foo = True
foobar = True
"""
    assert align_assignment_expressions(static_code) == \
"""
code = True
'''looks_like_variable = True'''
foo    = True
foobar = True
"""

def test_semicolon_in_string():
    static_code = \
"""
foobar = "val with semicolon in:"
"""
    assert align_assignment_expressions(static_code) == static_code

def test_semicolon_in_string_2():
    code = \
"""
type_hint = f"{': '+ type.strip() if type else ''}"
"""
    assert align_assignment_expressions(code) == \
"""
type_hint = f"{': '+ type.strip() if type else ''}"
"""

def test_assignment_on_line_adjacent_to_named_initializer_argument():
    code = \
"""
loss = torch.Tensor([0.])
loss.requires_grad = True
return SequenceClassifierOutput(loss.sum(), logits=torch.zeros_like(input_ids1))
"""
    assert align_assignment_expressions(code) == \
"""
loss               = torch.Tensor([0.])
loss.requires_grad = True
return SequenceClassifierOutput(loss.sum(), logits=torch.zeros_like(input_ids1))
"""

def test_indented_assignment_on_line_adjacent_to_named_initializer_argument():
    code = \
"""
    loss = torch.Tensor([0.])
    loss.requires_grad = True
    return SequenceClassifierOutput(loss.sum(), logits=torch.zeros_like(input_ids1))
"""
    assert align_assignment_expressions(code) == \
"""
    loss               = torch.Tensor([0.])
    loss.requires_grad = True
    return SequenceClassifierOutput(loss.sum(), logits=torch.zeros_like(input_ids1))
"""

def test_assertion_statement_ignored():
    static_code = \
"""
foobar = 1
assert foobar == 1
"""
    assert align_assignment_expressions(static_code) == static_code

def test_non_adjacent_assignment_ignored():
    static_code = \
"""
    input_ids1 = input_ids.squeeze()
    if not torch.all((input_ids1 >= 0) & (input_ids1 <= self._num_embeddings)):
        loss = torch.Tensor([0.])
"""
    assert align_assignment_expressions(static_code) == static_code

@mark.parametrize("string_prefix", ("f", "r"))
def test_triple_quote_string_prefix(string_prefix):
    static_code = \
f'''
    {string_prefix}"""
    labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
    Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
    config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
    If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
    """
'''
    assert align_assignment_expressions(static_code) == static_code

def test_formatting_is_idempotent():
    code = \
"""
    group: list[tuple[int, str]] = []
    lines = code
"""
    formatted_once  = align_assignment_expressions(code)
    formatted_twice = align_assignment_expressions(formatted_once)
    assert formatted_once == formatted_twice