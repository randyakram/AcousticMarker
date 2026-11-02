"""
Grad-CAM, factored out so Stage 3 (generating out-of-fold explanations for the whole
development set) and Stage 5 (the same, once, for the confirmation set) use identical code.
"""

import torch


class GradCAM:
    """Hooks a conv layer's activations + gradients, turns the gradient-weighted activation
    map into a heatmap for one input. See MIClassifier.conv2 -- that's the intended target
    layer for this model."""

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x, class_idx):
        """class_idx should almost always be fixed (e.g. always the MI class), not
        "whichever class the model predicted" -- see Stage 3's notebook for why that
        distinction matters for comparing heatmaps across recordings."""
        self.model.zero_grad()
        # cuDNN's fused GRU kernel only keeps what backward() needs when the forward pass runs
        # in train() mode; we want eval() for deterministic BatchNorm/Dropout, so drop to the
        # non-cuDNN RNN path for just this forward+backward. No-op on CPU.
        with torch.backends.cudnn.flags(enabled=False):
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
            logits[:, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)      # [1, C, 1, 1]
        cam = torch.relu((weights * self.activations).sum(dim=1))    # [1, F', T']
        cam = cam[0].cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)     # normalise to [0, 1]
        return cam, probs


def compute_ensemble_gradcam(models, x, class_idx, target_layer_attr="conv2"):
    """Averages Grad-CAM heatmaps and predicted probabilities across multiple models -- for
    Stage 5, this means every fold's model, since a confirmation-set patient was never seen by
    ANY of them (they're all equally out-of-sample), unlike Stage 3 where each fold explained
    only its own held-out patients. x: [1, freq, time] tensor, already on the right device."""
    import numpy as np
    cams, probs_list = [], []
    for model in models:
        cam_engine = GradCAM(model, getattr(model, target_layer_attr))
        was_training = model.training
        model.eval()
        cam, probs = cam_engine(x, class_idx=class_idx)
        model.train(was_training)
        cams.append(cam)
        probs_list.append(probs)
    return np.mean(cams, axis=0), np.mean(probs_list, axis=0)


def compute_gradcam_batch(model, features_batch, device, target_class, target_layer=None):
    """Computes a Grad-CAM heatmap for every item in a batch (looped -- Grad-CAM's backward
    hooks are per-sample, not batchable) and returns them stacked, plus each sample's predicted
    probabilities. features_batch: [B, freq, time] tensor (not yet on device)."""
    target_layer = target_layer or model.conv2
    cam_engine = GradCAM(model, target_layer)

    was_training = model.training
    model.eval()

    cams, probs_list = [], []
    for i in range(features_batch.shape[0]):
        x = features_batch[i:i + 1].to(device)
        cam, probs = cam_engine(x, class_idx=target_class)
        cams.append(cam)
        probs_list.append(probs)

    model.train(was_training)
    import numpy as np
    return np.stack(cams), np.stack(probs_list)
