# Third-Party Notices

## Upstream detector code

AI Foto adapts portions of `lynote-ai/ai-image-detector`, licensed under MIT. The upstream license text is included in [`LICENSE.ai-image-detector`](LICENSE.ai-image-detector). Source: <https://github.com/lynote-ai/ai-image-detector>.

## Detector models and weights

- **UnivFD / UniversalFakeDetect:** Original method by Utkarsh Ojha, Yuheng Li, and Yong Jae Lee, CVPR 2023. Official implementation and code license: <https://github.com/WisconsinAIVision/UniversalFakeDetect>. The classifier checkpoint is sourced from the DeepSafe mirror: <https://huggingface.co/siddharthksah/deepsafe-weights>. DeepSafe identifies the UniversalFakeDetect classifier and OpenAI CLIP ViT-L/14 weights as works of their respective original authors. OpenAI CLIP source: <https://github.com/openai/CLIP>.
- **Hugging Face detector:** `capcheck/ai-image-detection`, Apache-2.0, model card: <https://huggingface.co/capcheck/ai-image-detection>. The card credits Google's ViT base architecture and `dima806/ai_vs_real_image_detection`, also Apache-2.0: <https://huggingface.co/dima806/ai_vs_real_image_detection>.
- **Nonescape Mini:** Adapted from `e3ntity/nonescape`, Apache-2.0: <https://github.com/e3ntity/nonescape>. The upstream model card/repository remains the source of the downloadable weights.
- **Hybrid and Hybrid Plus:** These are arithmetic combinations of the base outputs implemented by AI Foto. They are not separately trained or independently sourced models.

Model files may have additional upstream terms. Obtain model files only from the linked official sources and consult their current model cards before redistribution.

## Python dependencies

Runtime and build dependencies are pinned in [`requirements.txt`](requirements.txt). Each package retains its upstream license metadata in its distribution. The application bundles PyTorch, torchvision, PySide6, Transformers, OpenCLIP, Pillow, python-docx, and their transitive dependencies in the macOS DMG.
