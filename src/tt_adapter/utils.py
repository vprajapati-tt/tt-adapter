from ttmlir import ir
from ttmlir.dialects import ttkernel, tt, ttir
from model_explorer import graph_builder
from collections import defaultdict

def get_attrs(op):
    result = []
    for attr in op.attributes:
        result.append(graph_builder.KeyValue(key=attr.name, value=str(attr.attr)))
    return result

def array_ref_repr(array_ref):
    return str(list(array_ref))

def get_name(name):
    if isinstance(name, str): return name
    else: return name.value