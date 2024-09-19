# main.py

from typing import Dict
from model_explorer import Adapter, AdapterMetadata, ModelExplorerGraphs, graph_builder
from collections import defaultdict
from ttmlir import ir, passes
from ttmlir.dialects import ttkernel, tt, ttir
from .utils import *
from .ttir import *
from .overrides import *
from ttrt.common.api import API
import os
import tempfile
import glob
import subprocess


class TTAdapter(Adapter):

    metadata = AdapterMetadata(
        id="tt_adapter",
        name="Tenstorrent Adapter",
        description="Prototype adapter from TT BUDA to Model Explorer",
        source_repo="https://github.com/vprajapati-tt/tt-explorer",
        fileExts=["ttir", "mlir"],
    )

    # Required.
    def __init__(self):
        super().__init__()

    def initialize(self, model_path: str, settings: Dict):
        # Initialize machine to generate SystemDesc and load up functionality to begin
        API.initialize_apis()

        query = API.Query(args={"save_artifacts": True})
        query()  # Save the SystemDesc to a certain localpath: './ttrt-artifacts/system_desc.ttsys'
        os.environ[
            "TT_SYSTEM_DESC_PATH"
        ] = f"{os.getcwd()}/ttrt-artifacts/system_desc.ttsys"
        return to_adapter_format(
            {"system_desc_path": f"{os.getcwd()}/ttrt-artifacts/system_desc.ttsys"}
        )

    def execute(self, model_path: str, settings: Dict):
        # Initialize the SystemDesc for the PipelineOption
        if "TT_SYSTEM_DESC_PATH" in os.environ:
            ttir_to_ttnn_options = (
                f'system-desc-path={os.environ["TT_SYSTEM_DESC_PATH"]}'
            )
        if "ttir_to_ttnn_options" in settings:
            ttir_to_ttnn_options += "," + ",".join(settings["ttir_to_ttnn_options"])

        # Created scoped context and module
        f = open(model_path, "r")
        with ir.Context() as ctx:
            ttkernel.register_dialect(ctx)
            ttir.register_dialect(ctx)
            tt.register_dialect(ctx)
            module = ir.Module.parse("".join(f.readlines()), ctx)
            # Transform the module into TTNN, apply options if any
            passes.ttir_to_ttnn_backend_pipeline(module, ttir_to_ttnn_options)
            with tempfile.NamedTemporaryFile(mode="w", suffix=".ttnn") as temp_module:
                passes.ttnn_to_flatbuffer_file(module, temp_module.name)
                with tempfile.TemporaryDirectory() as _artifact_dir:
                    artifact_dir = _artifact_dir
                    # Create a subprocess to invoke ttrt perf
                    # Set artifact_dir to permanent path if provided in settings:
                    if "artifact_dir" in settings and is_valid_path(
                        settings["artifact_dir"]
                    ):
                        artifact_dir = settings["artifact_dir"]
                        print(f"Artifacts will be saved to {artifact_dir}")
                    res = subprocess.run(
                        " ".join(
                            [
                                "ttrt",
                                "perf",
                                temp_module.name,
                                "--save-artifacts",
                                f"--artifact-dir={artifact_dir}",
                                f"--log-file={artifact_dir}/ttrt.log",
                            ]
                        ),
                        shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                    )
                    result = {"stdout": res.stdout.decode("utf-8")}
                    csv_result = glob.glob(f"{artifact_dir}/**/*.csv", recursive=True)
                    if csv_result:
                        result["perf_trace"] = open(csv_result[0], "r").read()
                    result["log_file"] = open(f"{artifact_dir}/ttrt.log", "r").read()

        return to_adapter_format(result)

    def override(self, model_path: str, settings: Dict):
        f = open(model_path, "r")
        with ir.Context() as ctx:
            ttir.register_dialect(ctx)
            tt.register_dialect(ctx)
            module = ir.Module.parse(f.read(), ctx)

        # Apply overrides, save them to module
        result = overrides_process_settings(settings, module)

        if result["success"]:
            with open(model_path, "w") as f:
                f.write(str(module))

            graph = ttir_to_graph(module, ctx)
            result["graphs"] = [graph]
        print(result)
        return to_adapter_format(result)

    def convert(self, model_path: str, settings: Dict) -> ModelExplorerGraphs:
        f = open(model_path, "r")
        with ir.Context() as ctx:
            ttkernel.register_dialect(ctx)
            ttir.register_dialect(ctx)
            tt.register_dialect(ctx)
            module = ir.Module.parse("".join(f.readlines()), ctx)
            self.initialize("")  # Initialize to load the System Desc for the first pass
            passes.ttnn_pipeline_ttir_passes(
                module
            )  # Run first pass to put into format for overrides

        # Apply overrides, save new module and send new model_path back.
        overrides_process_convert_settings(settings, module)

        # Convert TTIR to Model Explorer Graphs and Display/Return
        graph = ttir_to_graph(module, ctx)
        return {"graphs": [graph]}
