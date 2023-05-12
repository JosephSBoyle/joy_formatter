# joy-format: a python formatter for aligning adjacent assignment expressions
--------------
### Background
Having tried Golang I grew to appreciate the automatic alignment of statements on adjacent lines.
Personally, I find that it makes the code easier to scan and comprehend.

Seeing code formatted in this way brings me joy, so I'm writing a formatter in the hope that I can
use it in some projects of mine. Ideally, it will extend something like `black`'s formatting,
which I by-and-large find to be great.

## Example:
Before:
```python
class Foo:
    def __init__(self, config):
        self.num_labels = config.num_labels
        self.config = config

        self.longformer = LongformerModel(config, add_pooling_layer=False)
        self.first_linear = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.second_linear = nn.Linear(config.hidden_size, config.num_labels, bias=False)
        self.third_linear = nn.Linear(config.hidden_size, config.num_labels)
```
after applying `joy`:
```python
class Foo:
    def __init__(self, config):
        self.num_labels = config.num_labels
        self.config     = config

        self.longformer    = LongformerModel(config, add_pooling_layer=False)
        self.first_linear  = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.second_linear = nn.Linear(config.hidden_size, config.num_labels, bias=False)
        self.third_linear  = nn.Linear(config.hidden_size, config.num_labels)
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