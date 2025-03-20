![](https://github.com/jlianglab/Foundation_X/blob/main/Figures/Title_Logo.png)
<!-- # Integrating Classification, Localization, and Segmentation through Lock-Release Pretraining Strategy for Chest X-ray Analysis -->

Developing deep-learning models for medical imaging requires large, annotated datasets, but the heterogeneity of annotations across tasks presents significant challenges. Foundation X is an end-to-end framework designed to train a multi-task foundation model by leveraging diverse expert-level annotations from multiple public datasets. It introduces a Cyclic & Lock-Release pretraining strategy alongside a student-teacher learning paradigm to enhance knowledge retention while mitigating overfitting. Trained on 11 chest X-ray datasets, Foundation X seamlessly integrates classification, localization, and segmentation tasks. Experimental results demonstrate its ability to maximize annotation utility, improve cross-dataset and cross-task learning, and achieve superior performance in disease classification, localization, and segmentation.

![](https://github.com/jlianglab/Foundation_X/blob/main/Figures/Foundation%20X%20-%20Method_Figure.png)

## Publications
**Foundation X: Integrating Classification, Localization, and Segmentation through Lock-Release Pretraining Strategy for Chest X-ray Analysis** <br/>
[Nahid Ul Islam](https://scholar.google.com/citations?hl=en&user=uusv5scAAAAJ&view_op=list_works&sortby=pubdate)<sup>1</sup>, [DongAo Ma](https://scholar.google.com/citations?user=qbLjOGcAAAAJ&hl=en)<sup>1</sup>, [Jiaxuan Pang](https://scholar.google.com/citations?user=hvE5HSoAAAAJ&hl=en)<sup>1</sup>, [Shivasakthi Senthil Velan](https://scholar.google.com/citations?user=mAmlLQUAAAAJ&hl=en)<sup>1</sup>, [Michael B Gotway](https://www.mayoclinic.org/biographies/gotway-michael-b-m-d/bio-20055566)<sup>2</sup>, and [Jianming Liang](https://scholar.google.com/citations?user=rUTf4hgAAAAJ&hl=en)<sup>1</sup><br/>
<sup>1</sup>Arizona State University, <sup>2</sup>Mayo Clinic<br/>
[Winter Conference on Applications of Computer Vision (WACV-2025)](https://wacv2025.thecvf.com/)<br/>
[Paper](https://openaccess.thecvf.com/content/WACV2025/papers/Islam_Foundation_X_Integrating_Classification_Localization_and_Segmentation_through_Lock-Release_Pretraining_WACV_2025_paper.pdf) | [Poster](https://github.com/jlianglab/Foundation_X/blob/main/Figures/FoundationX_POSTER.pdf) | [Code](https://github.com/jlianglab/Foundation_X)

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

## Major results from our work
**1. Foundation X maximizes performance improvements by utilizing all available annotations for classification, localization, and segmentation.
**
<p align="left">
<img src="https://github.com/jlianglab/Foundation_X/blob/main/Figures/Result_FoundationX_PretrainingResults.png" width=70% height=70%>
</p>
