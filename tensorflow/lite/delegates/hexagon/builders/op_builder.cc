    case kTfLiteBuiltinArgMax:
      return CreateArgMinMaxOpBuilder(this, OP_ArgMax_8toInt32);
    case kTfLiteBuiltinArgMin:
      return CreateArgMinMaxOpBuilder(this, OP_ArgMin_8);
    case kTfLiteBuiltinSub:
      return CreateArithmeticBuilder(this, OP_QuantizedSub_8p8to8);
    case kTfLiteBuiltinMean:
      return CreateReduceBuilder(this, OP_QuantizedMean_8);
    case kTfLiteBuiltinSum:
      return CreateReduceBuilder(this, OP_QuantizedSum_8to32);
    case kTfLiteBuiltinPad:
      return CreatePadBuilder(this, OP_QuantizedPad_8);
    case kTfLiteBuiltinMirrorPad:
      return CreateMirrorPadBuilder(this, OP_MirrorPad_8);
    case kTfLiteBuiltinMaxPool2d:
      return CreatePool2DBuilder(this, OP_QuantizedMaxPool_8);
    case kTfLiteBuiltinConcatenation:
      return CreateConcatBuilder(this, OP_QuantizedConcat_8);
    case kTfLiteBuiltinTransposeConv:
      return CreateTransposeConv2DBuilder(
          this, OP_QuantizedTransposeConv2d_8x8p32to8);
    case kTfLiteBuiltinDepthwiseConv2d:
      return CreateConv2DBuilder(this, OP_DepthwiseSupernode_8x8p32to8);
    case kTfLiteBuiltinReshape:
      return CreateReshapeBuilder(this, OP_Reshape);
    case kTfLiteBuiltinResizeNearestNeighbor:
      return CreateResizeNearestNeighborBuilder(this,
                                                OP_ResizeNearestNeighbor_8);
    case kTfLiteBuiltinL2Normalization:
      return CreateL2NormalizationBuilder(this, OP_L2Normalize_8);
    case kTfLiteBuiltinRelu6:
      return CreateActivationBuilder(this, OP_QuantizedReluX_8);
    case kTfLiteBuiltinTanh:
      return CreateActivationBuilder(this, OP_QuantizedTanh_8);
    case kTfLiteBuiltinLogistic:
      return CreateActivationBuilder(this, OP_QuantizedSigmoid_8);
    case kTfLiteBuiltinSplit:
      return CreateSplitBuilder(this, OP_QuantizedSplit_8);
    case kTfLiteBuiltinResizeBilinear:
      return CreateResizeBilinearOpBuilder(this, OP_QuantizedResizeBilinear_8);
    case kTfLiteBuiltinNeg:
      return CreateNegOpBuilder(this, OP_QuantizedNeg_8);
    case kTfLiteBuiltinTranspose:
      return CreateTransposeBuilder(this, OP_Transpose_8);
    case kTfLiteBuiltinSpaceToDepth:
      return CreateSpaceToDepthBuilder(this, OP_SpaceToDepth_8);
    case kTfLiteBuiltinDepthToSpace:
      return CreateSpaceToDepthBuilder(this, OP_DepthToSpace_8);
    case kTfLiteBuiltinQuantize:
      return CreateQuantizeBuilder(this, OP_Requantize_8to8);
    case kTfLiteBuiltinHardSwish:
      return CreateHardSwishBuilder(this, OP_QuantizedHardSwish_8);
    case kTfLiteBuiltinMinimum:
      return CreateMinMaxBuilder(this, OP_QuantizedMinimum_8);
    case kTfLiteBuiltinMaximum:
      return CreateMinMaxBuilder(this, OP_QuantizedMaximum_8);
    case kTfLiteBuiltinSlice:
      return CreateSliceOpBuilder(this, OP_QuantizedSlice_8);
    case kTfLiteBuiltinPack:
      return CreatePackBuilder(this, OP_QuantizedPack_8);
    case kTfLiteBuiltinStridedSlice:
      return CreateStridedSliceBuilder(this, OP_QuantizedStridedSlice_8);
    case kTfLiteBuiltinSquaredDifference:
      return CreateSquaredDifferenceOpBuilder(this, OP_QuantizedSub_8p8to8);
    case kTfLiteBuiltinRsqrt:
      return CreateRSqrtOpBuilder(this, OP_QuantizedSqrt_8);

// TODO(b/154604279): Support these casting ops in Hexagon op profiling (which
// seems to key tensors on a single op, which may not be the case now).
TfLiteStatus GraphBuilder::AddCastOp(TfLiteContext* context, int op_type,
                                     int tensor_id,
                                     OpBuilder** cast_op_builder) {
  // Create a new OpBuilder for casting the tensor.
  OpBuilder* cast_builder = CreateCastBuilder(this, op_type);
  builders_.emplace_back(cast_builder);
  cast_builder->SetNodeId(builders_.size());
  // We cast the tensor in-place, so there is only 1 input & output which is the
  // same.
  auto* tensor_data = TfLiteIntArrayCreate(1);
  tensor_data->data[0] = tensor_id;

  TF_LITE_ENSURE_STATUS(
      cast_builder->PopulateSubGraph(tensor_data, tensor_data, context));
  TF_LITE_ENSURE_STATUS(cast_builder->RegisterOutputs(tensor_data, context));

  TfLiteIntArrayFree(tensor_data);
  if (cast_op_builder != nullptr) *cast_op_builder = cast_builder;
  return kTfLiteOk;
}

void GraphBuilder::AddBatchSeqConfig(int max_size_for_batch,
                                     TfLiteIntArray* input_batch_dimensions,
                                     TfLiteIntArray* output_batch_dimensions) {
  OpBuilder* batch_seq_node =
      CreateBatchSeqBuilder(this, OP_BatchSeqConfig, max_size_for_batch,
                            input_batch_dimensions, output_batch_dimensions);
  builders_.emplace_back(batch_seq_node);
  batch_seq_node->SetNodeId(builders_.size());
  batch_seq_node->PopulateSubGraph(nullptr, nullptr, nullptr);
  max_size_for_batch_ = max_size_for_batch;
}
