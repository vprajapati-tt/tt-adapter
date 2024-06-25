# tt-adapter
Model Explorer Adapter built for TT-MLIR Compiled outputs.

## Installation
This repository depends on [nsmithtt/tt-mlir](https://github.com/nsmithtt/tt-mlir), build and activate the `venv` as the python bindings are a dependency of TT-Adapter. Run `pip install .` in the root of this repository to install `tt-adapter` into the `ttmlir_venv` environment.

## Integration into model-explorer
Model-Explorer currently primarily supports loading extensions through the CLI. An example of a run call: 

```sh
model-explorer --extensions=tt_adapter
```

You should be able to see

```sh
Loading extensions...
 - ...
 - Tenstorrent Adapter
 - JSON adapter
```

in the command output to verify that it has been run.
