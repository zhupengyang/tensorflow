OpBuilder* CreateArgMinMaxOpBuilder(GraphBuilder* graph_builder, int op_type);
OpBuilder* CreateActivationBuilder(GraphBuilder* graph_builder, int op_type);
OpBuilder* CreateArithmeticBuilder(GraphBuilder* graph_builder, int op_type);
OpBuilder* CreateConcatBuilder(GraphBuilder* graph_builder, int op_type);
OpBuilder* CreateTransposeConv2DBuilder(GraphBuilder* graph_builder,
                                        int op_type);
OpBuilder* CreateReshapeBuilder(GraphBuilder* graph_builder, int op_type);
OpBuilder* CreateReduceBuilder(GraphBuilder* graph_builder, int op_type);
OpBuilder* CreateMirrorPadBuilder(GraphBuilder* graph_builder, int op_type);
OpBuilder* CreatePadBuilder(GraphBuilder* graph_builder, int op_type);
OpBuilder* CreateResizeNearestNeighborBuilder(GraphBuilder* graph_builder,
                                              int op_type);
OpBuilder* CreateL2NormalizationBuilder(GraphBuilder* graph_builder,
                                        int op_type);
OpBuilder* CreateSplitBuilder(GraphBuilder* graph_builder, int op_type);
OpBuilder* CreateResizeBilinearOpBuilder(GraphBuilder* graph_builder,
                                         int op_type);
OpBuilder* CreateNegOpBuilder(GraphBuilder* graph_builder, int op_type);
OpBuilder* CreateTransposeBuilder(GraphBuilder* graph_builder, int op_type);
OpBuilder* CreateSpaceToDepthBuilder(GraphBuilder* graph_builder, int op_type);
OpBuilder* CreateBatchSeqBuilder(GraphBuilder* graph_builder, int op_type,
                                 int max_size_for_batch,
                                 TfLiteIntArray* input_batch_dimensions,
                                 TfLiteIntArray* output_batch_dimensions);
OpBuilder* CreateQuantizeBuilder(GraphBuilder* graph_builder, int op_type);
OpBuilder* CreateHardSwishBuilder(GraphBuilder* graph_builder, int op_type);
OpBuilder* CreateCastBuilder(GraphBuilder* graph_builder, int op_type);
OpBuilder* CreateMinMaxBuilder(GraphBuilder* graph_builder, int op_type);
OpBuilder* CreateSliceOpBuilder(GraphBuilder* graph_builder, int op_type);
OpBuilder* CreatePackBuilder(GraphBuilder* graph_builder, int op_type);
OpBuilder* CreateStridedSliceBuilder(GraphBuilder* graph_builder, int op_type);
OpBuilder* CreateSquaredDifferenceOpBuilder(GraphBuilder* graph_builder,
                                            int op_type);
OpBuilder* CreateRSqrtOpBuilder(GraphBuilder* graph_builder, int op_type);
