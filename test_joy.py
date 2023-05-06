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
