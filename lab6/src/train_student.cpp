// Part C: data-parallel training. Fill in TODO(C1, C2).
//
//   ctest --test-dir build/lab6 -L train --output-on-failure
//   mpirun -np 4 build/lab6/train --mode overlap --bucket-mb 4
//
// Read include/coll/train.hpp first: the flat gradient buffer is ordered the
// way backward produces it (last layer first), so ready gradients always form
// a contiguous prefix [0, ready_end) of the buffer.
#include "coll/train.hpp"

#include <stdexcept>

namespace coll {

[[maybe_unused]] static NotImplemented todo(const char* what) {
    return NotImplemented(std::string("TODO ") + what + " is not implemented yet");
}

// ------------------------------------------------------------------- C1 ----
// TODO(C1) gradient bucketing, as in PyTorch DDP's Reducer.
// on_layer_ready(l) is called in the order L-1, L-2, ..., 0.
//   1. Check the order: l must equal expect_ (start at num_layers() - 1, then
//      count down); throw std::logic_error otherwise.
//   2. The layer's gradients [offset(l), offset(l) + size(l)) extend the
//      ready-but-not-submitted range: pending_ += size(l).
//   3. If pending_ >= cap_floats_: comm_.submit(grads + start_, pending_);
//      start_ += pending_; pending_ = 0; ++buckets_.
// finish(): submit what is left (if pending_ > 0, counts as a bucket), wait for
// the comm thread, divide every gradient by P_ (allreduce SUMs; DDP averages),
// and reset start_, pending_, expect_ for the next step.
// With a tiny cap every layer is its own bucket; with a huge cap there is one
// bucket, submitted only after backward is done (= no overlap at all).
Bucketer::Bucketer(MLP& model, size_t bucket_bytes, CommThread& comm, int world_size)
    : m_(model), cap_floats_(std::max<size_t>(1, bucket_bytes / sizeof(float))), comm_(comm),
      P_(world_size), expect_(model.num_layers() - 1) {}

void Bucketer::on_layer_ready(int layer) {
    (void)layer;
    throw todo("C1 Bucketer::on_layer_ready");
}

void Bucketer::finish() {
    throw todo("C1 Bucketer::finish");
}

// ------------------------------------------------------------------- C2 ----
// TODO(C2) ZeRO stage 1 (Rajbhandari et al., SC'20), built on YOUR ring:
//   constructor: [begin_, end_) = chunk_range(num_params, P, rank);
//                m_, v_ = zeros of that length (instead of num_params each!)
//   step():
//     1. reduce_scatter_ring(grads): this rank now holds the SUM of its chunk
//     2. divide its chunk by P (average)
//     3. ++t_; adam_update on params[begin_..end_) with grads[begin_..end_), m_, v_
//     4. allgather_ring(params): every rank gets every updated chunk
// Communication volume equals a plain allreduce (RS + AG), but optimizer
// state per rank drops from 2N to 2N/P floats.
Zero1Adam::Zero1Adam(size_t num_params, const Comm& c, Adam h) : c_(c), h_(h) {
    (void)num_params;
    begin_ = end_ = 0;
}

void Zero1Adam::step(std::vector<float>& params, std::vector<float>& grads) {
    (void)params;
    (void)grads;
    throw todo("C2 Zero1Adam::step");
}

} // namespace coll
