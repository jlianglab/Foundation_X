![](https://github.com/jlianglab/Foundation_X/blob/main/Figures/Title_Logo.png)
<!-- # Integrating Classification, Localization, and Segmentation through Lock-Release Pretraining Strategy for Chest X-ray Analysis -->

Developing deep-learning models for medical imaging requires large, annotated datasets, but the heterogeneity of annotations across tasks presents significant challenges. Foundation X is an end-to-end framework designed to train a multi-task foundation model by leveraging diverse expert-level annotations from multiple public datasets. It introduces a Cyclic & Lock-Release pretraining strategy alongside a student-teacher learning paradigm to enhance knowledge retention while mitigating overfitting. Trained on 11 chest X-ray datasets, Foundation X seamlessly integrates classification, localization, and segmentation tasks. Experimental results demonstrate its ability to maximize annotation utility, improve cross-dataset and cross-task learning, and achieve superior performance in disease classification, localization, and segmentation.

![](https://github.com/jlianglab/Foundation_X/blob/main/Figures/Foundation%20X%20-%20Method_Figure.png)

## Publication
**Foundation X: Integrating Classification, Localization, and Segmentation through Lock-Release Pretraining Strategy for Chest X-ray Analysis** <br/>
[Nahid Ul Islam](https://scholar.google.com/citations?hl=en&user=uusv5scAAAAJ&view_op=list_works&sortby=pubdate)<sup>1</sup>, [DongAo Ma](https://scholar.google.com/citations?user=qbLjOGcAAAAJ&hl=en)<sup>1</sup>, [Jiaxuan Pang](https://scholar.google.com/citations?user=hvE5HSoAAAAJ&hl=en)<sup>1</sup>, [Shivasakthi Senthil Velan](https://scholar.google.com/citations?user=mAmlLQUAAAAJ&hl=en)<sup>1</sup>, [Michael B Gotway](https://www.mayoclinic.org/biographies/gotway-michael-b-m-d/bio-20055566)<sup>2</sup>, and [Jianming Liang](https://scholar.google.com/citations?user=rUTf4hgAAAAJ&hl=en)<sup>1</sup><br/>
<sup>1</sup>Arizona State University, <sup>2</sup>Mayo Clinic<br/>
[Winter Conference on Applications of Computer Vision (WACV-2025)](https://wacv2025.thecvf.com/)<br/>
[Paper](https://openaccess.thecvf.com/content/WACV2025/papers/Islam_Foundation_X_Integrating_Classification_Localization_and_Segmentation_through_Lock-Release_Pretraining_WACV_2025_paper.pdf) | [Supp](https://openaccess.thecvf.com/content/WACV2025/supplemental/Islam_Foundation_X_Integrating_WACV_2025_supplemental.pdf) | [Poster](https://github.com/jlianglab/Foundation_X/blob/main/Figures/FoundationX_POSTER.pdf) | [Code](https://github.com/jlianglab/Foundation_X) | [Presentation Slides](https://github.com/jlianglab/Foundation_X/blob/main/Figures/Foundation%20X%20-%20Presentation_Slides.pdf) | [Presentation](https://youtu.be/eT3vatrU8MU?si=nzJc0JdCS6Q1txGh)

## Datasets
1. [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/)
2. [NIH ChestX-ray14](https://www.kaggle.com/datasets/nih-chest-xrays/data)
3. [VinDr-CXR](https://vindr.ai/datasets/cxr)
4. [NIH Schenzhen CXR](https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Shenzhen-Hospital-CXR-Set/index.html)
5. [MIMIC-II](https://archive.physionet.org/mimic2/)
6. [TBX11k](https://www.kaggle.com/datasets/usmanshams/tbx-11)
7. [NODE21](https://node21.grand-challenge.org/Data/)
8. [CANDID-PTX](https://figshare.com/articles/dataset/CANDID-PTX/14173982)
9. [RSNA Pneumonia](https://www.rsna.org/rsnai/ai-image-challenge/rsna-pneumonia-detection-challenge-2018)
10. [ChestX-Det](https://service.tib.eu/ldmservice/dataset/chestx-det)
11. [SIIM-ACR](https://www.kaggle.com/competitions/siim-acr-pneumothorax-segmentation)
12. [CheXmask VinDr-CXR](https://www.nature.com/articles/s41597-024-03358-1)
13. [VinDr-RibCXR](https://vindr.ai/datasets/ribcxr)
14. [NIH Montgomery](https://www.nih.gov/about-nih/nih-montgomery-county-leased-facilities)
15. [JSRT](http://db.jsrt.or.jp/eng.php)
<br/>
<p align="left">
    <img src="https://github.com/jlianglab/Foundation_X/blob/main/Figures/Foundation X - Dataset_Collection.png" width=60% height=60%>
    <br/><em>We pretrain our Foundation X model using 11 publicly available chest X-ray datasets, as shown in the first 11 datasets in the table. Although not every dataset contains all three types of annotations—classification, localization, and segmentation—we leverage all available annotations to maximize the model’s learning potential. Among these datasets, all include classification ground truths, six provide localization bounding box annotations, and three offer segmentation masks for diseases. Furthermore, we utilize organ localization and segmentation datasets from VinDr-CXR, VinDr-RibCXR, NIH Montgomery, and JSRT for target task fine-tuning. Here, the organ segmentation masks for VinDr-CXR were sourced from the CheXmask database. We also fine-tuned VinDr-CXR with local labels for the disease localization task.</em>
</p>

## Data Splits and Bounding Box Annotations
- Data splits and generated COCO-format localization bouding box annotation files can be downloaded through this [Google Form](https://forms.gle/wdiq3s6SNvsd6nn78). <br/>

## Pre-trained models
You can download the pretrained models through this [Google Form](https://forms.gle/7ynYFcoiKYYwQNWG8).

## Setting-up the multiscaledeformableattention package
Please follow the steps described in [DINO GitHub Repo](https://github.com/IDEA-Research/DINO) to install the package "multiscaledeformableattention".

## Pretraining Instructions
- Follow the script [`scripts/run_IntegratedModel_Foundation6_ClsLocSeg_v102.sh`](https://github.com/jlianglab/Foundation_X/blob/main/scripts/run_IntegratedModel_Foundation6_ClsLocSeg_v102.sh) to start pretraining **Foundation X model**. <br/>
- Make sure to update the data direcotry in the files [`datasets/coco.py`](https://github.com/jlianglab/Foundation_X/blob/main/datasets/coco.py) (for localization tasks) and [`datasets_medical.py`](https://github.com/jlianglab/Foundation_X/blob/main/datasets_medical.py) (for classification and segmentation tasks). <br/>
- If Classification Heads need to be increased or decreased the file [`models/dino/swin_transformer_CyclicSegmentation.py`](https://github.com/jlianglab/Foundation_X/blob/main/models/dino/swin_transformer_CyclicSegmentation.py#L637) should be modified.
- If Segmentation Heads need to be increased or decreased the file [`models/dino/swin_transformer_CyclicSegmentation.py`](https://github.com/jlianglab/Foundation_X/blob/main/models/dino/swin_transformer_CyclicSegmentation.py#L672) should be modified.
- If the number of Localization Decoders needs to be adjusted, the following code snippet must be modified. Currently, the code reflects 6 Localization Decoders. <br/>
    * [`models/dino/dino_F6.py Line159`](https://github.com/jlianglab/Foundation_X/blob/main/models/dino/dino_F6.py#L159) <br/>
    * [`models/dino/dino_F6.py Line221`](https://github.com/jlianglab/Foundation_X/blob/main/models/dino/dino_F6.py#L221) <br/>
    * [`models/dino/dino_F6.py Line252`](https://github.com/jlianglab/Foundation_X/blob/main/models/dino/dino_F6.py#L252) <br/>
    * [`models/dino/deformable_transformer_F6.py Line153`](https://github.com/jlianglab/Foundation_X/blob/main/models/dino/deformable_transformer_F6.py#L153) <br/>
    * [`models/dino/deformable_transformer_F6.py Line201`](https://github.com/jlianglab/Foundation_X/blob/main/models/dino/deformable_transformer_F6.py#L201) <br/>
    * [`models/dino/deformable_transformer_F6.py Line236`](https://github.com/jlianglab/Foundation_X/blob/main/models/dino/deformable_transformer_F6.py#L236) <br/>
    * [`models/dino/deformable_transformer_F6.py Line259`](https://github.com/jlianglab/Foundation_X/blob/main/models/dino/deformable_transformer_F6.py#L259) <br/>
    * [`models/dino/deformable_transformer_F6.py Line282`](https://github.com/jlianglab/Foundation_X/blob/main/models/dino/deformable_transformer_F6.py#L282) <br/>

<br/>

## Major results from our work
**1. Foundation X maximizes performance improvements during pretraining by utilizing all available annotations for classification, localization, and segmentation.**
<p align="left">
<img src="https://github.com/jlianglab/Foundation_X/blob/main/Figures/Result_FoundationX_PretrainingResults.png" width=75% height=75%>
</p>

**2. Foundation X enhances performance when jointly trained for organ localization and segmentation and excels during finetuning.**
<p align="left">
<img src="https://github.com/jlianglab/Foundation_X/blob/main/Figures/Result_FoundationX_LocSeg_VinDrCXR.png" width=50% height=50%>
</p>
<p align="left">
<img src="https://github.com/jlianglab/Foundation_X/blob/main/Figures/Result_FoundationX_Seg_OtherDatasets.png" width=50% height=50%>
</p>

**3. Foundation X excels in few-shot learning and shows strong performance across training samples.**
<p align="left">
<img src="https://github.com/jlianglab/Foundation_X/blob/main/Figures/Result_FoundationX_FewShot.png" width=70% height=70%>
</p>

**4. Foundation X maximizes performance with cross-dataset and cross-task learning.**
<p align="left">
<img src="https://github.com/jlianglab/Foundation_X/blob/main/Figures/Result_FoundationX_CrossTask_CrossDataset.png" width=70% height=70%>
</p>

**5. Foundation X full finetuning outperforms head-only finetuning and baseline models.**
<p align="left">
<img src="https://github.com/jlianglab/Foundation_X/blob/main/Figures/Result_FoundationX_Discussion_Localization_VinDrCXR.png" width=70% height=70%>
</p>

<br/>

## Citation
If you use this code or use our pre-trained models for your research, please cite our paper:

```
@InProceedings{Islam_2025_WACV,
    author    = {Islam, Nahid Ul and Ma, DongAo and Pang, Jiaxuan and Velan, Shivasakthi Senthil and Gotway, Michael and Liang, Jianming},
    title     = {Foundation X: Integrating Classification Localization and Segmentation through Lock-Release Pretraining Strategy for Chest X-ray Analysis},
    booktitle = {Proceedings of the Winter Conference on Applications of Computer Vision (WACV)},
    month     = {February},
    year      = {2025},
    pages     = {3647-3656}
}
```


## Acnkowledgement
This research was partially supported by ASU and Mayo Clinic through a Seed Grant and an Innovation Grant, as well as by the NIH under Award Number R01HL128785. The authors are solely responsible for the content, which does not necessarily reflect the official views of the NIH. This work also utilized GPUs provided by ASU Research Computing (SOL), Bridges-2 at the Pittsburgh Supercomputing Center (allocated under BCS190015), and Anvil at Purdue University (allocated under MED220025). These resources are supported by the Advanced Cyberinfrastructure Coordination Ecosystem: Services & Support (ACCESS) program, funded by the National Science Foundation under grants #2138259, #2138286, #2138307, #2137603, and #2138296. We also extend our gratitude to Anirudh Kaniyar Narayana Iyengar for his contributions to collecting localization data, preparing bounding boxes in COCO format, and developing some of the data loaders. Finally, the content of this paper is covered by patents pending.


## Contact
For any questions, feel free to reach out: <br/>
Email: nuislam (at) asu.edu


## License
Released under the [ASU GitHub Project License](https://github.com/jlianglab/Foundation_X/blob/main/LICENSE)
