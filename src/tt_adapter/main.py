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


    with ir.Context() as ctx:
        ttkernel.register_dialect(ctx)
        ttir.register_dialect(ctx)
        tt.register_dialect(ctx)

        module = ir.Module.parse(''.join(f.readlines()), ctx)
        
        for op in module.body.operations:
            # High level functions, need to list their arguments in the graph
            name_num = name_dict[get_name(op.name)]
            id = get_name(op.name) + str(name_num)
            graph.nodes.append(graph_builder.GraphNode(id=id, label=get_name(op.name)))
            graph.nodes[-1].attrs.extend(get_attrs(op))
            for arg in op.arguments:
                value_dict[arg.get_name()] = graph.nodes[-1]
            
            for region in op.regions:
                for block in region.blocks:
                    for op in block.operations:
                        # Just list out the nodes and assign their ids
                        name_num = name_dict[get_name(op.name)]
                        id = get_name(op.name) + str(name_num)
                        name_dict[get_name(op.name)] += 1
                        graph.nodes.append(graph_builder.GraphNode(id=id, label=get_name(op.name)))
                        graph.nodes[-1].attrs.extend(get_attrs(op))
                        for result in op.results:
                            # Attach the graph node here
                            value_dict[result.get_name()] = graph.nodes[-1]
                        
                        for ops in op.operands:
                            # Guaranteed topological ordering, so we can start to connect previous values to their respective operations
                            source_node: graph_builder.GraphNode = value_dict[ops.get_name()]
                            graph.nodes[-1].incomingEdges.append(graph_builder.IncomingEdge(
                               sourceNodeId=source_node.id,
                               sourceNodeOutputId=str(connections[source_node.id]),
                               targetNodeInputId=str(len(graph.nodes[-1].incomingEdges))
                            ))

                            layout = tt.ir.LayoutAttr.getLayout(ops.type)

                            source_node_attrs = [
                                graph_builder.KeyValue(key='shape', value=str(ops.type.shape)),
                                graph_builder.KeyValue(key='element_type', value=str(ops.type.element_type)),
                                graph_builder.KeyValue(key='rank', value=str(ops.type.rank)),
                                graph_builder.KeyValue(key='strides', value=array_ref_repr(layout.strides)),
                                graph_builder.KeyValue(key='Out of Bounds Value', value=layout.oobval.name),
                                graph_builder.KeyValue(key='Memory Space', value=layout.memory_space.name),
                                graph_builder.KeyValue(key='Grid Shape', value=array_ref_repr(layout.grid_attr.shape))
                            ]

                            source_node.outputsMetadata.append(
                                graph_builder.MetadataItem(
                                   id=str(connections[source_node.id]),
                                   attrs=[graph_builder.KeyValue(key='__tensor_tag', value=id)] + source_node_attrs
                                )
                            )

                            source_node.attrs.extend(source_node_attrs)
                            
                            connections[source_node.id] += 1

    return {'graphs': [graph]}