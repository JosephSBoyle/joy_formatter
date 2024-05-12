from pytest import mark

from joy import align_assignment_expressions


def test_base():
    """

    """
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
    """

    """
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
    """

    """
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
    """

    """
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
    """

    """
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
    """

    """
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
    """

    """
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
    """

    """
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
    """

    """
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
    """

    """
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
    """

    """
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
    """

    """
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
    """

    """
    static_code = \
"""
# commented_variable_assignment = 1
variable = loss.sum()
"""
    assert align_assignment_expressions(static_code) == static_code

def test_indented_commented_adjacent_line():
    """

    """
    static_code = \
"""
    # commented_variable_assignment = 1
    variable = loss.sum()
"""
    assert align_assignment_expressions(static_code) == static_code

def test_commented_and_typed_adjacent_line():
    """

    """
    static_code = \
"""
# commented_variable_assignment: int = 1
variable = loss.sum()
"""
    assert align_assignment_expressions(static_code) == static_code

def test_commented_typed_function_arg_definition():
    """

    """
    static_code = \
"""
def f(
    # commented_variable_assignment: int = 1,
    variable = loss.sum()
)
"""
    assert align_assignment_expressions(static_code) == static_code

def test_conditional_line():
    """

    """
    static_code = \
"""
if not torch.all((input_ids1 >= 0) & (input_ids1 <= self._num_embeddings)):
    pass
"""
    assert align_assignment_expressions(static_code) == static_code

def test_conditional_line_2():
    """

    """
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
    """

    """
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
    """

    """
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
    """

    """
    static_code = \
'''
code = True
"""looks_like_variable = True
"""
'''
    assert align_assignment_expressions(static_code) == static_code

def test_single_line_docstring():
    """

    """
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
    """

    """
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
    """

    """
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
    """

    """
    static_code = \
"""
foobar = "val with semicolon in:"
"""
    assert align_assignment_expressions(static_code) == static_code

def test_semicolon_in_string_2():
    """

    """
    code = \
"""
type_hint = f"{': '+ type.strip() if type else ''}"
"""
    assert align_assignment_expressions(code) == \
"""
type_hint = f"{': '+ type.strip() if type else ''}"
"""

def test_assignment_on_line_adjacent_to_named_initializer_argument():
    """

    """
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
    """

    """
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
    """

    """
    static_code = \
"""
foobar = 1
assert foobar == 1
"""
    assert align_assignment_expressions(static_code) == static_code

def test_non_adjacent_assignment_ignored():
    """

    """
    static_code = \
"""
    input_ids1 = input_ids.squeeze()
    if not torch.all((input_ids1 >= 0) & (input_ids1 <= self._num_embeddings)):
        loss = torch.Tensor([0.])
"""
    assert align_assignment_expressions(static_code) == static_code

@mark.parametrize("string_prefix", ("f", "r"))
def test_triple_quote_string_prefix(string_prefix):
    """

    Args:
        string_prefix ():

    """
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
    """

    """
    code = \
"""
    group: list[tuple[int, str]] = []
    lines = code
"""
    formatted_once  = align_assignment_expressions(code)
    formatted_twice = align_assignment_expressions(formatted_once)
    assert formatted_once == formatted_twice

def test_handle_non_assignment_equals_character_operations():
    """

    """
    code = \
"""
x = 1
bazbar = 2
foo %= bazbar
bar += bazbar
bat -= bazbar
baz /= bazbar
"""
    assert align_assignment_expressions(code) == \
"""
x      = 1
bazbar = 2
foo %= bazbar
bar += bazbar
bat -= bazbar
baz /= bazbar
"""

def test_align_multiline_function_declaration_type_hints():
    """

    """
    code = \
"""
def f(
    attention: torch.Tensor,
    labels: np.ndarray,
    logger: logging.Logger,    
) -> tuple[torch.Tensor, bool]:
"""
    assert align_assignment_expressions(code) == \
"""
def f(
    attention : torch.Tensor,
    labels    : np.ndarray,
    logger    : logging.Logger,
) -> tuple[torch.Tensor, bool]:
"""

def test_indexed_assignment():
    """

    """
    static_code = \
"""
attention_mask[:, 0] = 1
"""
    assert align_assignment_expressions(static_code) == static_code

def test_type_hinted_indexed_assignment():
    """

    """
    static_code = \
"""
attention_mask[:, 0]: int = 1
"""
    assert align_assignment_expressions(static_code) == static_code

def test_single_line_function_assignment():
    """

    """
    static_code = \
"""
progress_bar.set_postfix(loss = epoch_loss / completed_steps)
"""
    assert align_assignment_expressions(static_code) == static_code

def handle_python_3_11_function_arg_type_hints():
    """

    """
    code = \
"""
    def foo(
        bar: int | None = None,
        bazbaz=None
    ):
        return bar
"""
    assert align_assignment_expressions(code) == \
"""
 def foo(
        bar : int | None = None,
        bazbaz           = None
    ):
        return bar
"""

def test_keyword_argument_not_aligned():
    """

    """
    static_code = \
"""
    print(align_assignment_expressions(lines), end="")
"""
    assert align_assignment_expressions(static_code) == static_code

def test_short_variable_declaration():
    """

    """
    static_code = \
"""
K = 10
"""
    assert align_assignment_expressions(static_code) == static_code

def test_multiline_class_instantiation():
    """

    """
    code = \
"""
x = dict(
    foo=bar,
    bazbaz=xxx
)
"""
    assert align_assignment_expressions(code) == \
"""
x = dict(
    foo    = bar,
    bazbaz = xxx
)
"""

def test_multiline_tuple_instantiation():
    """

    """
    code = \
"""
x = (
    foo=bar,
    bazbaz=xxx
)
"""
    assert align_assignment_expressions(code) == \
"""
x = (
    foo    = bar,
    bazbaz = xxx
)
"""

def test_multiline_dict_instantiation():
    """

    """
    code = \
"""
x = {
    foo    : bar,
    bazbaz : xxx
}
"""
    assert align_assignment_expressions(code) == \
"""
x = {
    foo    : bar,
    bazbaz : xxx
}
"""

def test_different_indentations_not_aligned():
    """

    """
    code = \
"""
for i in range(self.Y):
    code = dicts['ind2c'][i]
    weights[i] = code_embs[code]
self.U.weight.data = torch.Tensor(weights).clone()
self.final.weight.data = torch.Tensor(weights).clone()
"""
    assert align_assignment_expressions(code) == \
"""
for i in range(self.Y):
    code       = dicts['ind2c'][i]
    weights[i] = code_embs[code]
self.U.weight.data     = torch.Tensor(weights).clone()
self.final.weight.data = torch.Tensor(weights).clone()
"""

def test_different_indentations_not_aligned_2():
    """

    """
    static_code = \
"""
if self.model_mode == "laat":
    hidden_output = outputs[0].view(batch_size, num_chunks*chunk_size, -1)
elif self.model_mode == "laat-split":
    hidden_output = outputs[0].view(batch_size*num_chunks, chunk_size, -1)
weights = torch.tanh(self.first_linear(hidden_output))
"""
    assert align_assignment_expressions(static_code) == static_code


### Real Examples ###

def test_align_indented_lines():
    """

    """
    code = \
"""
            self.first_linear = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            self.second_linear = nn.Linear(config.hidden_size, config.num_labels, bias=False)
            self.third_linear = nn.Linear(config.hidden_size, config.num_labels)
"""
    assert align_assignment_expressions(code) == \
"""
            self.first_linear  = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            self.second_linear = nn.Linear(config.hidden_size, config.num_labels, bias=False)
            self.third_linear  = nn.Linear(config.hidden_size, config.num_labels)
"""

def test_format_on_last_line():
    """

    """
    code =\
"""
split = "train"
n_docs = 50
language = "gpt-english" # "spanish"
model_name = MODEL_NAME"""
    assert align_assignment_expressions(code) == \
"""
split      = "train"
n_docs     = 50
language   = "gpt-english" # "spanish"
model_name = MODEL_NAME"""

def test_multi_line_dataframe_code():
    # To avoid breaking things, simply do nothing when a line contains double equals.
    """

    """
    static_code =\
"""
x = pd.load_df("...")
x[y == z] = 0
x = x[z == y]
"""
    assert align_assignment_expressions(static_code) == static_code

def test_semicolon_not_type_hint_expression():
    """

    """
    static_code =\
"""
logging.info("content count: gifs: %s, imgs: %s", len(gifs), len(imgs))
logging.info('content count: gifs: %s, imgs: %s', len(gifs), len(imgs))
"""
    assert align_assignment_expressions(static_code) == static_code

def test_arguments_to_decorator_expression():
    """

    """
    static_code =\
"""
@tree.command(name="gif", description="Get a random gif...")
"""
    assert align_assignment_expressions(static_code) == static_code

def test_format_code_in_nested_triple_quote_docstring():
    """

    """
    code =\
"""
class A:
    '''
    \"\"\"
        foobar = A(...)
        baz = A(..., kwargs=...)
    \"\"\"
    '''
"""
    assert align_assignment_expressions(code) ==\
"""
class A:
    '''
    \"\"\"
        foobar = A(...)
        baz    = A(..., kwargs=...)
    \"\"\"
    '''
"""

def test_dataframe_in_return_statement(): # TODO rename
    """

    """
    static_code =\
"""
    merged_df = merged_df[merged_df["label"].isin(labels)]
    return merged_df[merged_df["subreddit"] == subreddit]
"""
    assert align_assignment_expressions(static_code) == static_code

def test_double_equals_in_non_assignment_statement():
    """

    """
    static_code =\
r"""
    assert (
        num_submitted_user_ids = = 125
    ), f"Expected 125 users, found {num_submitted_user_ids} users in {submission_json_path}."
"""
    assert align_assignment_expressions(static_code) == static_code

def test_weird_alignment():
    """

    """
    static_code =\
"""
    def x() : return None
    _dict = vars().copy()
"""
    assert align_assignment_expressions(static_code) == static_code
