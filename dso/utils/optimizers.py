import sys

from ..optimizers import NGDOptimizer, NGDProfiler, NGDGradError


def get_optimizer(model, lr, init_damping, target_damping, damping_decay_steps, weight_decay, freq, momentum,
	batch_size, iters, warmup_epochs, backend, randk, compression_ratio, profiling=False, grad_norm=False,
        grad_error=False, rank_analysis=False, adaptive=None, sngd=False, enable_id=False, enable_is=False,
        prefix=''):

	if profiling or grad_norm or rank_analysis:
		profiler = NGDProfiler(
			model, lr=lr, init_damping=init_damping, target_damping=target_damping, weight_decay=weight_decay,
			freq=freq, momentum=momentum, batch_size=batch_size, iters=iters,
			warmup_epochs=warmup_epochs, backend=backend, compression_ratio=compression_ratio,
			profiling=profiling, grad_norm=grad_norm, rank_analysis=rank_analysis, adaptive=adaptive, sngd=sngd,
                        enable_id=enable_id, enable_is=enable_is, prefix=prefix)

		return profiler

	if grad_error:
		optimizer = NGDGradError(
			model, lr=lr, init_damping=init_damping, target_damping=target_damping, weight_decay=weight_decay,
			freq=freq, momentum=momentum, batch_size=batch_size, iters=iters, 
			warmup_epochs=warmup_epochs, backend=backend, randk=randk, compression_ratio=compression_ratio,
			adaptive=adaptive, enable_id=enable_id, enable_is=enable_is)

		return optimizer

        # for benchmarking
	optimizer = NGDOptimizer(
		model, lr=lr, init_damping=init_damping, target_damping=target_damping, damping_decay_steps=damping_decay_steps, weight_decay=weight_decay,
		freq=freq, momentum=momentum, batch_size=batch_size, iters=iters, 
		warmup_epochs=warmup_epochs, backend=backend, compression_ratio=compression_ratio,
                adaptive=adaptive)

	return optimizer
