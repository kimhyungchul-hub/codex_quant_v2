# Compatibility shim for GlobalBatchEvaluator - PyTorch version
try:
    from engines.mc.entry_evaluation_vmap_pytorch import GlobalBatchEvaluator
except Exception:
    class GlobalBatchEvaluator:
        def __init__(self, *args, **kwargs):
            # Fallback to a disabled state instead of raising in __init__
            self._disabled = True
            
        def evaluate_batch(self, *args, **kwargs):
            raise RuntimeError("GlobalBatchEvaluator (PyTorch) not available")
