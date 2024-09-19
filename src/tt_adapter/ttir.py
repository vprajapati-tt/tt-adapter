# Library to manipulate TTIR Modules

from model_explorer import graph_builder
from ttmlir.dialects import tt, ttir, ttkernel
from .utils import *
from collections import defaultdict


def ttir_to_graph(module, ctx):
    name_dict = defaultdict(int)
    connections = defaultdict(int)
    value_dict = {}
    graph = graph_builder.Graph(id="ttir-graph")

    for op in module.body.operations:
        # High level functions, need to list their arguments in the graph
        name = get_loc_str(op.location)
        name_num = name_dict[name]
        id = name + "!!" + str(name_num)
        namespace = name
        graph.nodes.append(graph_builder.GraphNode(id=id, label=get_name(op.name)))
        graph.nodes[-1].attrs.extend(get_attrs(op))
        for arg in op.arguments:
            value_dict[arg.get_name()] = graph.nodes[-1]

        for region in op.regions:
            for block in region.blocks:
                for op in block.operations:
                    # Just list out the nodes and assign their ids
                    name = get_loc_str(op.location)
                    name_num = name_dict[name]
                    id = name + "!!" + str(name_num)
                    name_dict[name] += 1
                    graph.nodes.append(
                        graph_builder.GraphNode(
                            id=id, label=get_name(op.name), namespace=namespace
                        )
                    )
                    graph.nodes[-1].attrs.extend(get_attrs(op))
                    for result in op.results:
                        # Attach the graph node here
                        value_dict[result.get_name()] = graph.nodes[-1]

                    for ops in op.operands:
                        # Guaranteed topological ordering, so we can start to connect previous values to their respective operations
                        source_node: graph_builder.GraphNode = value_dict[
                            ops.get_name()
                        ]
                        graph.nodes[-1].incomingEdges.append(
                            graph_builder.IncomingEdge(
                                sourceNodeId=source_node.id,
                                sourceNodeOutputId=str(connections[source_node.id]),
                                targetNodeInputId=str(
                                    len(graph.nodes[-1].incomingEdges)
                                ),
                            )
                        )

                        if hasattr(ops.type, "encoding") and ops.type.encoding is None:
                            source_node_attrs = [
                                graph_builder.KeyValue(
                                    key="shape", value=str(ops.type.shape)
                                ),
                                graph_builder.KeyValue(
                                    key="element_type",
                                    value=str(ops.type.element_type),
                                ),
                                graph_builder.KeyValue(
                                    key="rank", value=str(ops.type.rank)
                                ),
                            ]
                        else:
                            layout = tt.ir.LayoutAttr.getLayout(ops.type)

                            source_node_attrs = [
                                graph_builder.KeyValue(
                                    key="shape", value=str(ops.type.shape)
                                ),
                                graph_builder.KeyValue(
                                    key="element_type",
                                    value=str(ops.type.element_type),
                                ),
                                graph_builder.KeyValue(
                                    key="rank", value=str(ops.type.rank)
                                ),
                                # graph_builder.KeyValue(
                                #    key="strides",
                                #    value=array_ref_repr(layout.stride),
                                # ),
                                # graph_builder.KeyValue(
                                #    key="Out of Bounds Value",
                                #    value=layout.oobval.name,
                                # ),
                                # graph_builder.KeyValue(
                                #    key="Memory Space",
                                #    value=layout.memory_space.name,
                                # ),
                                make_editable_kv(
                                    graph_builder.KeyValue(
                                        key="Grid Shape",
                                        value=array_ref_repr(layout.grid_attr.shape),
                                    ),
                                    editable={
                                        "input_type": "value_list",
                                        "options": ["1x1", "4x4", "8x8"],
                                    },
                                ),
                            ]

                        source_node.outputsMetadata.append(
                            graph_builder.MetadataItem(
                                id=str(connections[source_node.id]),
                                attrs=[
                                    graph_builder.KeyValue(key="__tensor_tag", value=id)
                                ]
                                + source_node_attrs,
                            )
                        )

                        source_node.attrs.extend(source_node_attrs)

                        connections[source_node.id] += 1
    return graph
