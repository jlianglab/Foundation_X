# Foundation X+

## Data Splits and Bounding Box Annotations
- Data splits and generated COCO-format localization bouding box annotation files can be downloaded through this [Google Form](https://forms.gle/wdiq3s6SNvsd6nn78). <br/>
- The [config.py](https://github.com/jlianglab/Foundation_X/blob/main/Foundation_X%2B/config/config_datasets.py) file defines the directory structure for all datasets and specifies the task head indices associated with each task.
- 
## Pre-trained models
- You can download the pretrained models through this [Google Form](https://docs.google.com/forms/d/e/1FAIpQLScK8x0awPHDWGWnN-pU64nTBWaK0nTKHyl2OcxPGp4rOCsSHg/viewform?usp=publish-editor).

## Loading pre-trained Foundation X+ checkpoints and extracting features
We provide a utility script `load_weights.py` to initialize a Swin-B backbone using our pretrained Foundation X checkpoints. The model only loads the encoder weights from the checkpoint, and supports an optional projection layer.

 #### Example: Load model and extract features
 ```python
from load_weights import build_model

# Path to the pretrained Foundation X+ checkpoint
pretrained_weights = "path/to/weights/ckpt.pth"

# Initialize the model
foundationx_model = build_model(
    pretrained_weights,
    num_classes=0,
    projector_features=256,     # Optional: dimensionality of projection layer
    use_mlp=True
)

foundationx_model.eval()  # Set model to evaluation mode

# extract features from input batch (e.g., [B, 3, 224, 224])
with torch.no_grad():
    features = foundationx_model.forward_features(input_tensor)
```

## Setting-up the multiscaledeformableattention package
- Please follow the steps described in [DINO GitHub Repo](https://github.com/IDEA-Research/DINO) to install the package "multiscaledeformableattention".

## Pretraining Foundation X+ Instructions
**Script:**
- Use the script [`scripts/run_IntegratedModel_Foundation6_ClsLocSeg_v107.sh`](https://github.com/jlianglab/Foundation_X/blob/main/Foundation_X%2B/scripts/run_IntegratedModel_Foundation6_ClsLocSeg_v107.sh) to start pretraining **Foundation X+ model**. <br/>
- This script launches pretraining across all 11 datasets and 20 tasks.
- Update the LOGFILE parameter to specify where pretrained models will be saved.
- Use the --debug flag to run a quick check and identify potential issues before full pretraining.
- Use --resume $RESUME to load pretrained weights and continue training from a specified checkpoint.

**Flexible Dataset Picking:**
- The updated code has the flexibility to choose the dataset and it's task on which the Foundation X to be pretrained on.
- To include all datasets and tasks, use the following parameter in the script:
```
cyclictask=chexpertCLS_nihchestxray14CLS_vindrcxrCLS_nihshenzenCLS_mimic2CLS_tbx11kCLS_node21CLS_candidptxCLS_rsnapneumoniaCLS_chestxdetCLS_siimacrCLS_tbx11kLOC_node21LOC_candidptxLOC_rsnapneumoniaLOC_chestxdetLOC_siimacrLOC_candidptxSEG_chestxdetSEG_siimacrSEG
```

## Acnkowledgement
This research was partially supported by ASU and Mayo Clinic through a Seed Grant and an Innovation Grant, as well as by the NIH under Award Number R01HL128785. The authors are solely responsible for the content, which does not necessarily reflect the official views of the NIH. This work also utilized GPUs provided by ASU Research Computing (SOL), Bridges-2 at the Pittsburgh Supercomputing Center (allocated under BCS190015), and Anvil at Purdue University (allocated under MED220025). These resources are supported by the Advanced Cyberinfrastructure Coordination Ecosystem: Services & Support (ACCESS) program, funded by the National Science Foundation under grants #2138259, #2138286, #2138307, #2137603, and #2138296. We also extend our gratitude to Anirudh Kaniyar Narayana Iyengar for his contributions to collecting localization data, preparing bounding boxes in COCO format, and developing some of the data loaders. Finally, the content of this paper is covered by patents pending.


## Contact
For any questions, feel free to reach out: <br/>
Email: nuislam (at) asu.edu
