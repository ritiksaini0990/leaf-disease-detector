import torch
import torch.nn.functional as F
import numpy as np
import cv2


class GradCAM:

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        self.forward_hook = target_layer.register_forward_hook(
            self._save_activation
        )

        self.backward_hook = target_layer.register_full_backward_hook(
            self._save_gradient
        )

    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, class_index=None):

        self.model.zero_grad()

        output = self.model(input_tensor)

        if class_index is None:
            class_index = torch.argmax(
                output,
                dim=1
            ).item()

        score = output[:, class_index]

        score.backward()

        gradients = self.gradients
        activations = self.activations

        weights = torch.mean(
            gradients,
            dim=(2, 3),
            keepdim=True
        )

        cam = torch.sum(
            weights * activations,
            dim=1
        )

        cam = F.relu(cam)

        cam = (
            cam.squeeze()
            .detach()
            .cpu()
            .numpy()
        )

        cam = cv2.resize(
            cam,
            (224, 224)
        )

        cam = cam - np.min(cam)

        if np.max(cam) > 0:
            cam = cam / np.max(cam)

        return cam