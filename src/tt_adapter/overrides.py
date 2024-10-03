# Library for implementing overrides to TTIR Modules

from ttmlir import overrides, passes
from ttmlir.dialects import tt
import os
from .ttir import get_ops


class Override:
    def __init__(self, keys):
        self.keys = keys

    def override_present(self, overrides):
        return any(x in overrides for x in self.keys)


class LayoutOverride(Override):
    def __init__(self):
        super().__init__(["Grid Shape", "Memory Space", "Memory Layout"])

    def make_layout_override(self, loc, overrides, module):
        values = dict(
            zip(
                self.keys,
                [overrides[key] if key in overrides else None for key in self.keys],
            )
        )
        ops = get_ops(module)
        for op in ops:
            print(op, op.location)
            try:
                _loc = str(op.location).split('"')[1]
            except:
                _loc = "unknown"
            if _loc == loc:
                # Found the op with which these sizes will be relayed
                if not op.result.type.encoding:
                    print("INVALID")
                    continue  # Op doesn't have a layout associated with it
                layout = tt.ir.LayoutAttr.getLayout(op.result.type)
                for k, v in values.items():
                    print("VALID", op.result.type.encoding)
                    if v is None:
                        if k == "Grid Shape":
                            values[k] = "x".join(map(str, layout.grid_attr.shape))
                        elif k == "Memory Space":
                            values[k] = str(tt.MemorySpace(layout.memory_space_as_int))
                        elif k == "Memory Layout":
                            values[k] = str(tt.TensorMemoryLayout(layout.memory_layout_as_int))
        return f'{loc}={values["Grid Shape"]}:{values["Memory Space"]}:{values["Memory Layout"]}'


def overrides_process_convert_settings(settings, module):
    # First load the SystemDesc if possible
    options = ""
    if "TT_SYSTEM_DESC_PATH" in os.environ:
        options += f'system-desc-path={os.environ["TT_SYSTEM_DESC_PATH"]}'

    # TODO(vprajapati): Parse more settings for pure conversion display here

    passes.ttnn_pipeline_ttir_passes(module, options)


def overrides_process_settings(_settings, module):
    result = {"success": False}
    settings = _settings["changes"]
    print(settings)
    # Run the first pass as a redundancy to allow for the loading of all the modules and such (will need to change)
    overrides_process_convert_settings({}, module)

    overrides = []

    for key in settings:
        if "!!" in key:  # ID From model_explorer, add to override command
            # Parse the override in the format of layout overrides
            override_dict = {}
            for override in settings[key]:
                override_dict[override["key"]] = override["value"]

            layout_override = LayoutOverride()

            if layout_override.override_present(override_dict):
                overrides.append(
                    layout_override.make_layout_override(
                        key.split("!!")[0], override_dict, module
                    )
                )

    override = ""

    if "TT_SYSTEM_DESC_PATH" in os.environ:
        override += f'system-desc-path={os.environ["TT_SYSTEM_DESC_PATH"]}'

    override += " override-output-layout=" + ",".join(overrides)

    print(override)

    passes.ttnn_pipeline_analysis_passes(module, override)
    result["success"] = True
    return result


def save_overriden_module(module, model_path):
    with open(model_path, "w") as f:
        f.write(str(module))
