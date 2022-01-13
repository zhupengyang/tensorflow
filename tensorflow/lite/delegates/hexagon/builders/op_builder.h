  // Adds BatchSeqConfig node to the graph. This is configuration
  // for a dynamic batch size for the graph.
  // A graph can have only one node of this type.
  void AddBatchSeqConfig(int max_size_for_batch,
                         TfLiteIntArray* input_batch_dimensions,
                         TfLiteIntArray* output_batch_dimensions);

  void AddDebugNode() {}

  void print() {
    printf("------------------------------\n");
    std::vector<unsigned char> buf(10000);
    hexagon_nn_->hexagon_nn_snpprint(graph_id_, buf.data(), buf.size());
    printf("%s", buf.data());
    printf("------------------------------\n");
    fflush(stdout);
  }

  // Returns true if the graph supports dynamic batch. False otherwise.
  bool GraphHasDynamicBatch() const { return max_size_for_batch_ != -1; }

  // Returns the maximum value for batch dimension the graph supports.
  // -1 if the graph doesn't support dynamic batch.
  int GetMaxBatchSize() const { return max_size_for_batch_; }

  // Adds a Cast op to convert a tensor from int8 to uint8 (or vice versa).
  // The builder which has the casting operator is filled in 'cast_op_builder'
  // if not nullptr.
  TfLiteStatus AddCastOp(TfLiteContext* context, int op_type, int tensor_id,
                         OpBuilder** cast_op_builder);

  // If the graph being built supports dynamic batch, this represents
  // the maximum value for batch.
  int max_size_for_batch_ = -1;
