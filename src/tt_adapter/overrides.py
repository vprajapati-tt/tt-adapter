# Library for implementing overrides to TTIR Modules

from ttmlir import overrides, passes
import os


def overrides_process_convert_settings(settings, module):
    # First load the SystemDesc if possible
    options = ""
    if "TT_SYSTEM_DESC_PATH" in os.environ:
        options += f'system-desc-path={os.environ["TT_SYSTEM_DESC_PATH"]}'

    # TODO(vprajapati): Parse more settings for pure conversion display here

    passes.ttir_first_pipeline(module, options)


def overrides_process_settings(settings, module):
    result = {"success": False}
    print(settings)

    overrides = []

    for key in settings:
        if "!!" in key:  # ID From model_explorer, add to override command
            overrides.append(f'{key.split("!!")[0]}={settings[key][0]["value"]}')

    override = "override-grid-sizes=" + ",".join(overrides)

    if "TT_SYSTEM_DESC_PATH" in os.environ:
        override += f' system-desc-path={os.environ["TT_SYSTEM_DESC_PATH"]}'

    print(override)

    passes.ttir_first_pipeline(module, override)
    result["success"] = True
    return result
