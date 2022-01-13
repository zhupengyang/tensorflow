    case kTfLiteBuiltinArgMax:
    case kTfLiteBuiltinArgMin:
    case kTfLiteBuiltinConcatenation:
    case kTfLiteBuiltinL2Normalization:
    case kTfLiteBuiltinLogistic:
    case kTfLiteBuiltinMaximum:
    case kTfLiteBuiltinMaxPool2d:
    case kTfLiteBuiltinMean:
    case kTfLiteBuiltinMinimum:
    case kTfLiteBuiltinMirrorPad:
    case kTfLiteBuiltinPack:
    case kTfLiteBuiltinPad:
    case kTfLiteBuiltinQuantize:
    case kTfLiteBuiltinRelu6:
    case kTfLiteBuiltinSlice:
    case kTfLiteBuiltinSpaceToDepth:
    case kTfLiteBuiltinDepthToSpace:
    case kTfLiteBuiltinSplit:
    case kTfLiteBuiltinStridedSlice:
    case kTfLiteBuiltinSub:
    case kTfLiteBuiltinTanh:
    case kTfLiteBuiltinTranspose:
      return registration->version <= 2;
    case kTfLiteBuiltinSquaredDifference:
    case kTfLiteBuiltinRsqrt:
      return registration->version == 2;
    case kTfLiteBuiltinDepthwiseConv2d:
    case kTfLiteBuiltinResizeBilinear:
    case kTfLiteBuiltinResizeNearestNeighbor:
    case kTfLiteBuiltinTransposeConv:
      return registration->version <= 3;
