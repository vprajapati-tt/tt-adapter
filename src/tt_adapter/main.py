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
        os.putenv(
            "TT_SYSTEM_DESC_PATH", f"{os.getcwd()}/ttrt-artifacts/system_desc.ttsys"
        )
        return {"system_desc_path": f"{os.getcwd()}/ttrt-artifacts/system_desc.ttsys"}

    def execute(self, model_path: str, settings: Dict):
        # Execute TTIR Module given model_path

        # Initialize settings and TTRT API
        API.initialize_apis()
        ttrt_args = API.Perf.registered_args

        if "ttrt_args" in settings and isinstance(settings["ttrt_args"], dict):
            for k, v in settings["ttrt_args"].items():
                ttrt_args[k] = v

        # Initialize the SystemDesc for the PipelineOption
        if "TT_SYSTEM_DESC_PATH" in os.environ:
            ttir_to_ttnn_options = f'path={os.environ["TT_SYSTEM_DESC_PATH"]},'
        ttir_to_ttnn_options += settings.get("ttir_to_ttnn_options", "")

        # Created scoped context and module
        f = open(model_path, "r")
        with ir.Context() as ctx:
            ttkernel.register_dialect(ctx)
            ttir.register_dialect(ctx)
            tt.register_dialect(ctx)
            module = ir.Module.parse("".join(f.readlines()), ctx)
            # Transform the module into TTNN, apply options if any
            passes.passes.ttir_to_ttnn_backend_pipeline(module, ttir_to_ttnn_options)
            fb_capsule = passes.passes.ttnn_to_flatbuffer_binary(module)
            ttrt_args["capsule"] = fb_capsule
            if "binary" in ttrt_args:
                del ttrt_args["binary"]
            perf_instance = API.Perf(args=ttrt_args)

        perf_instance()  # Run the performance trace

        return {}

    def convert(self, model_path: str, settings: Dict) -> ModelExplorerGraphs:
        f = open(model_path, "r")
        with ir.Context() as ctx:
            ttkernel.register_dialect(ctx)
            ttir.register_dialect(ctx)
            tt.register_dialect(ctx)
            module = ir.Module.parse("".join(f.readlines()), ctx)

        # Apply overrides, save new module and send new model_path back.
        overrides_process_me_settings(settings)

        # Convert TTIR to Model Explorer Graphs and Display/Return
        graph = ttir_to_graph(module, ctx)
        return {"graphs": [graph]}
