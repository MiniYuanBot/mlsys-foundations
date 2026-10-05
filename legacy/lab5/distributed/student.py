"""Implement this file; the data, optimizer, oracle and launcher are provided."""


def synchronize_gradients(parameters, local_samples, group=None):
    """Replace each local mean-loss gradient with the global sample-mean gradient.

    Contract:
      * Every rank calls this once after backward and before optimizer.step.
      * All ranks have the same ordered list of trainable parameters and positive
        local_samples; gradients are dense float64 CPU tensors in this exercise.
      * Handle world_size=1, non-uniform local batch sizes, and arbitrary rank count.
      * Preserve gradient tensor identity and perform communication under no_grad.

    B1: for equal local_samples, sum gradients and divide by world_size.
    B2: extend to non-uniform shards: reduce sample-weighted gradient sums and
        divide by the globally reduced sample count. Average-of-means is wrong.
    S: after correctness, implement a separate bucketed version and compare traces.

    Use real torch.distributed collectives. Do not call the oracle or read the
    other ranks' files. Do not initialize/destroy process groups in this function.
    """
    raise NotImplementedError("TODO: implement real sample-weighted gradient synchronization")
