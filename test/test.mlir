#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
#l1_ = #tt.memory_space<l1>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #tt.iterator_type<parallel>
#system = #tt.memory_space<system>
#layout = #tt.layout<8192x128x1, undef, <1x1>, memref<64x128xf32, #system>>
#layout1 = #tt.layout<8192x128x1, undef, <1x1>, memref<64x128xf32, #l1_>>
module attributes {torch.debug_module_name = "_lambda"} {
  func.func @forward(%arg0: tensor<64x128xf32, #layout>, %arg1: tensor<64x128xf32, #layout>) -> tensor<64x128xf32, #layout> {
    %0 = tensor.empty() : tensor<64x128xf32, #layout1>
    %1 = "ttir.layout"(%arg0, %0) : (tensor<64x128xf32, #layout>, tensor<64x128xf32, #layout1>) -> tensor<64x128xf32, #layout1>
    %2 = tensor.empty() : tensor<64x128xf32, #layout1>
    %3 = "ttir.layout"(%arg1, %2) : (tensor<64x128xf32, #layout>, tensor<64x128xf32, #layout1>) -> tensor<64x128xf32, #layout1>
    %4 = tensor.empty() : tensor<64x128xf32, #layout1>
    %5 = "ttir.dispatch"(%1, %3, %4) <{grid = #tt.grid<1x1>, indexing_maps = [#map, #map, #map], iterator_types = [#parallel, #parallel], operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#any_device, #any_device, #any_device]}> ({
    ^bb0(%arg2: memref<64x128xf32, #l1_>, %arg3: memref<64x128xf32, #l1_>, %arg4: memref<64x128xf32, #l1_>):
      %8 = "ttir.kernel"(%arg2, %arg3, %arg4) <{kind = @eltwise, op = @mulitply, operandSegmentSizes = array<i32: 2, 1>}> : (memref<64x128xf32, #l1_>, memref<64x128xf32, #l1_>, memref<64x128xf32, #l1_>) -> memref<64x128xf32, #l1_>
      "ttir.yield"(%8) : (memref<64x128xf32, #l1_>) -> ()
    }) : (tensor<64x128xf32, #layout1>, tensor<64x128xf32, #layout1>, tensor<64x128xf32, #layout1>) -> tensor<64x128xf32, #layout1>
    %6 = tensor.empty() : tensor<64x128xf32, #layout>
    %7 = "ttir.layout"(%5, %6) : (tensor<64x128xf32, #layout1>, tensor<64x128xf32, #layout>) -> tensor<64x128xf32, #layout>
    return %7 : tensor<64x128xf32, #layout>
  }
}