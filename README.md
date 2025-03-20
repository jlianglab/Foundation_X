![](https://github.com/jlianglab/Foundation_X/blob/main/Figures/Title_Logo.png)
<!-- # Integrating Classification, Localization, and Segmentation through Lock-Release Pretraining Strategy for Chest X-ray Analysis -->

Developing deep-learning models for medical imaging requires large, annotated datasets, but the heterogeneity of annotations across tasks presents significant challenges. Foundation X is an end-to-end framework designed to train a multi-task foundation model by leveraging diverse expert-level annotations from multiple public datasets. It introduces a Cyclic & Lock-Release pretraining strategy alongside a student-teacher learning paradigm to enhance knowledge retention while mitigating overfitting. Trained on 11 chest X-ray datasets, Foundation X seamlessly integrates classification, localization, and segmentation tasks. Experimental results demonstrate its ability to maximize annotation utility, improve cross-dataset and cross-task learning, and achieve superior performance in disease classification, localization, and segmentation.

![](https://github.com/jlianglab/Foundation_X/blob/main/Figures/Foundation%20X%20-%20Method_Figure.png)

## Publications
**Foundation X: Integrating Classification, Localization, and Segmentation through Lock-Release Pretraining Strategy for Chest X-ray Analysis** <br/>
[Nahid Ul Islam](https://scholar.google.com/citations?hl=en&user=uusv5scAAAAJ&view_op=list_works&sortby=pubdate)<sup>1</sup>, [DongAo Ma](https://scholar.google.com/citations?user=qbLjOGcAAAAJ&hl=en)<sup>1</sup>, [Jiaxuan Pang](https://scholar.google.com/citations?user=hvE5HSoAAAAJ&hl=en)<sup>1</sup>, [Shivasakthi Senthil Velan](https://scholar.google.com/citations?user=mAmlLQUAAAAJ&hl=en)<sup>1</sup>, [Michael B Gotway](https://www.mayoclinic.org/biographies/gotway-michael-b-m-d/bio-20055566)<sup>2</sup>, and [Jianming Liang](https://scholar.google.com/citations?user=rUTf4hgAAAAJ&hl=en)<sup>1</sup><br/>
<sup>1</sup>Arizona State University, <sup>2</sup>Mayo Clinic<br/>
[Winter Conference on Applications of Computer Vision (WACV-2025)](https://wacv2025.thecvf.com/)<br/>
[Paper](https://openaccess.thecvf.com/content/WACV2025/papers/Islam_Foundation_X_Integrating_Classification_Localization_and_Segmentation_through_Lock-Release_Pretraining_WACV_2025_paper.pdf) | [Supp](https://openaccess.thecvf.com/content/WACV2025/supplemental/Islam_Foundation_X_Integrating_WACV_2025_supplemental.pdf) | [Poster](https://github.com/jlianglab/Foundation_X/blob/main/Figures/FoundationX_POSTER.pdf) | [Code](https://github.com/jlianglab/Foundation_X)

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
    <img src="https://github.com/jlianglab/Foundation_X/blob/main/Figures/Foundation X - Dataset_Collection.png" width=70% height=70%>
    <br/><em>### [We pretrain our Foundation X model using 11 publicly available chest X-ray datasets, as shown in the first 11 datasets in the table. Although not every dataset contains all three types of annotations—classification, localization, and segmentation—we leverage all available annotations to maximize the model’s learning potential. Among these datasets, all include classification ground truths, six provide localization bounding box annotations, and three offer segmentation masks for diseases. Furthermore, we utilize organ localization and segmentation datasets from VinDr-CXR, VinDr-RibCXR, NIH Montgomery, and JSRT for target task fine-tuning. Here, the organ segmentation masks for VinDr-CXR were sourced from the CheXmask database. We also fine-tuned VinDr-CXR with local labels for the disease localization task.]</em>
</p>

## Major results from our work
**1. Foundation X maximizes performance improvements by utilizing all available annotations for classification, localization, and segmentation.**
<p align="left">
<img src="https://github.com/jlianglab/Foundation_X/blob/main/Figures/Result_FoundationX_PretrainingResults.png" width=70% height=70%>
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

## Data organization and preprocessing steps
Data splits and generated COCO-format annotation files can be found at the following locations:<br/>
[Classification/Segmentation Data split](https://www.dropbox.com/scl/fo/j26ybbjq27sfuc075lo2a/AFyr1fdpC0L7ZCPSeX-ZfeE?rlkey=wkbx6r8c7wjhdwp6uxog0k91s&st=5n17jsan&dl=0)<br/>
[Coco Format Localization Annotation Files](https://www.dropbox.com/scl/fo/gqsj733jn9iaw52zebm2r/AC25bQU0pQDhX4FDOEDuolY?rlkey=wgynvi1qyt5yfswwif2md8rdq&st=tnm19p4c&dl=0)<br/>

## Pre-trained models
You can download the pretrained models from [here](https://www.dropbox.com/scl/fo/nin8bu3cygdmdrmafuidl/AA_xaDmfd2o9aMTT_ZXGklM?rlkey=hdxjv89qk0u47dhh3sxsna8iy&st=hg78rq19&dl=0).


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

## License
Released under the [ASU GitHub Project License](https://github.com/jlianglab/Foundation_X/blob/main/LICENSE)
