# main.py

from typing import Dict
from model_explorer import Adapter, AdapterMetadata, ModelExplorerGraphs, graph_builder
from collections import defaultdict
from ttmlir import ir
from ttmlir.dialects import ttkernel, tt, ttir
from .utils import *

class TTAdapter(Adapter):

  metadata = AdapterMetadata(id='tt_adapter',
                             name='Tenstorrent Adapter',
                             description='Prototype adapter from TT BUDA to Model Explorer',
                             source_repo='https://github.com/vprajapati-tt/tt-explorer',
                             fileExts=['ttir', 'mlir'])

  # Required.
  def __init__(self):
    super().__init__()

  def convert(self, model_path: str, settings: Dict) -> ModelExplorerGraphs:
    f = open(model_path, 'r')
    
    name_dict = defaultdict(int)
    connections = defaultdict(int)
    value_dict = {}

    graph = graph_builder.Graph(id='ttir-graph')

    def recurse(operation, graph, nsprefix, global_arg_count):

        if hasattr(operation, "arguments"):
            for arg in operation.arguments:
                graph.nodes.append(graph_builder.GraphNode(id="%arg" + str(arg.arg_number), label="%arg" + str(arg.arg_number), namespace=nsprefix))
                value_dict[arg.get_name()] = graph.nodes[-1]
                global_arg_count += 1
        else:
            for opers in operation.operands:
                value_dict["%arg" + str(global_arg_count)] = value_dict[opers.get_name()]
                global_arg_count += 1

        for region in operation.regions:
            for block in region.blocks:
                op_count = 0
                for op in block.operations:

                    ns = get_name(operation.name) if nsprefix == "" else nsprefix + "/" + get_name(operation.name)
                    if len(op.regions) == 0:
                        nid = get_name(op.name) + str(name_dict[get_name(op.name)])
                        name_dict[get_name(op.name)] += 1
                        graph.nodes.append(graph_builder.GraphNode(id=nid, label=get_name(op.name), namespace=ns))
                        graph.nodes[-1].attrs.extend(get_attrs(op))

                        # TODO: In here, you have to propogate the result of the second to last operation in the block to the result of the last because it is a yield
                        # and has no result

                        i = 0
                        for result in op.results:
                            value_dict[result.get_name()] = graph.nodes[-1]
                            if i < len(operation.results):
                                value_dict[operation.results[i].get_name()] = graph.nodes[-1]
                            i += 1

                        for ops in op.operands:

                            if value_dict.get(ops.get_name()):
                                source_node = value_dict[ops.get_name()]
                                graph.nodes[-1].incomingEdges.append(graph_builder.IncomingEdge(
                                    sourceNodeId=source_node.id,
                                    sourceNodeOutputId=str(connections[source_node.id]),
                                    targetNodeInputId=str(len(graph.nodes[-1].incomingEdges))
                                ))

                                connections[source_node.id] += 1
                            else:
                                print("NO SOURCE NODE")
                                print(ops.get_name())
                    else:
                        recurse(op, graph, ns, global_arg_count)
                    op_count += 1

        return graph

    with ir.Context() as ctx:
        ttkernel.register_dialect(ctx)
        ttir.register_dialect(ctx)
        # tt.register_dialect(ctx)

        module = ir.Module.parse(''.join(f.readlines()), ctx)
        graph = recurse(module.body.operations[0], graph, "", 0)

    return {'graphs': [graph]}