# joy-format: a python formatter for aligning adjacent assignment expressions
--------------
I like like the aesthetic of assignment statements.

Seeing code formatted in this way brings me _joy_, so I'm writing a formatter in the hope that I can use it in some projects of mine. Ideally, it will extend something like `black`'s formatting, which I by-and-large find to be great.

### Examples:
#### Simple assignment example:
```python
# Before
foobar = baz
foo = foobar

# After
foobar = baz
foo    = foobar
```
#### More realistic example:
```python
# Before
self.first_linear = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
self.second_linear = nn.Linear(config.hidden_size, config.num_labels, bias=False)
self.third_linear = nn.Linear(config.hidden_size, config.num_labels)

# After
self.first_linear  = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
self.second_linear = nn.Linear(config.hidden_size, config.num_labels, bias=False)
self.third_linear  = nn.Linear(config.hidden_size, config.num_labels)
```
#### Method definition with default values:
```python
# Before
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
): ...

# After
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
): ...
```
#### Type hinted function arguments:
```python
# Before
def f(
    attention: torch.Tensor,
    labels: np.ndarray,
    logger: logging.Logger,    
) -> tuple[torch.Tensor, bool]: ...

# After
def f(
    attention : torch.Tensor,
    labels    : np.ndarray,
    logger    : logging.Logger,
) -> tuple[torch.Tensor, bool]: ...
```
## Pre-commit installation

Add the following to your `.pre-commit-config.yaml` file. Note: compatibility with other linters /
formatters is not guaranteed, please create an issue for any conflicts you encounter.

```yaml
repos:
-   repo: https://github.com/JosephSBoyle/joy_formatter.git
    rev: latest
    hooks:
    -   id: align-assignments
```


## VScode Installation
1. Git clone this repo.
2. Copy the full path to `joy.py`
3. Install the 'custom local formatters extension'
4. Open the `settings.json` file by pressing ctrl+shift+p and selecting "Open user settings json".
5. Add these key-value pairs. 

```json
    "python.autoComplete.extraPaths": [],
    "python.analysis.extraPaths": [],
    "customLocalFormatters.formatters": [
        {
          "command": "python <PATH TO YOUR joy.py SCRIPT HERE>",
          "languages": ["python"]
        }
    ],
```

6. Replace `<PATH TO YOUR joy.py SCRIPT HERE>` with the path from step #2!

#### Dev-Notes
The `./rt` command is used as a shorthand to  run the tests, passing any flags to `pytest`.
Sort of how some people use a `test` command in their makefiles and invoke `make test`
to run their tests.

If any code is malformatted by this tool, please open an issue with the input and output that you
expected / received and I'll do my best to take a look.