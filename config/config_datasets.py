DATASETS_CONFIG = {
    # NIH ChestX-ray14 - Classification - Localization
    "cls_nih_root": "/scratch/jliang12/data/nih_xray14/images/images/",
    "cls_nih_trainList": "data/xray14/official/train_official.txt",
    "cls_nih_valList": "data/xray14/official/val_official.txt",
    "cls_nih_testList": "data/xray14/official/test_official.txt",

    "loc_nih_testTag": "nihLOC_test",
    "loc_nih_root": "/data/jliang12/shared/dataset/NIH_Localization/bbox_img",
    "loc_nih_testList": "/data/jliang12/shared/dataset/NIH_Localization/nih_bbox_coco_1024_fromTensorcsv.json",

    # CheXpert - Classification
    "cls_chexpert_root": "/scratch/jliang12/data/",
    "cls_chexpert_trainList": "/scratch/nuislam/Model_Checkpoints/data_files_splits/ark6_classificationTask/CheXpert-v1.0_train.csv",
    "cls_chexpert_valList": "/scratch/nuislam/Model_Checkpoints/data_files_splits/ark6_classificationTask/CheXpert-v1.0_valid.csv",
    "cls_chexpert_testList": "/scratch/nuislam/Model_Checkpoints/data_files_splits/ark6_classificationTask/CheXpert_test_official.csv",

    # VINDrCXR - Classification
    "cls_vindrcxr_root": "/scratch/jliang12/data/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0/",
    "cls_vindrcxr_trainList": "/scratch/nuislam/Model_Checkpoints/data_files_splits/ark6_classificationTask/VinDrCXR_train_pe_global_one.txt",
    "cls_vindrcxr_valList": "/scratch/nuislam/Model_Checkpoints/data_files_splits/ark6_classificationTask/VinDrCXR_test_pe_global_one.txt",
    "cls_vindrcxr_testList": "/scratch/nuislam/Model_Checkpoints/data_files_splits/ark6_classificationTask/VinDrCXR_test_pe_global_one.txt",

    "loc_vindrcxr_trainTag": "vindrcxr_train",
    "loc_vindrcxr_testTag": "vindrcxr_test",
    "loc_vindrcxr_trainRoot": "/scratch/jliang12/data/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0/train_jpeg",
    "loc_vindrcxr_testRoot": "/scratch/jliang12/data/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0/test_jpeg",
    "loc_vindrcxr_trainList": "data/vindrcxr_detection/VinDrCXR_Kaggle_14Diseases_TRAIN_modified_wbf.json",
    "loc_vindrcxr_testList": "data/vindrcxr_detection/VinDrCXR_Kaggle_14Diseases_TEST_modified_wbf.json",

    # ShenzenCXR - Classification
    "cls_shenzencxr_root": "/scratch/jliang12/data/ShenzhenHospitalXray/ChinaSet_AllFiles/CXR_png/",
    "cls_shenzencxr_trainList": "/scratch/nuislam/Model_Checkpoints/data_files_splits/ark6_classificationTask/ShenzenCXR_train_data.txt",
    "cls_shenzencxr_valList": "/scratch/nuislam/Model_Checkpoints/data_files_splits/ark6_classificationTask/ShenzenCXR_valid_data.txt",
    "cls_shenzencxr_testList": "/scratch/nuislam/Model_Checkpoints/data_files_splits/ark6_classificationTask/ShenzenCXR_test_data.txt",

    # Mimic-CXR - Classification
    "cls_mimiccxr_root": "/scratch/jliang12/data/MIMIC_jpeg/physionet.org/files/mimic-cxr-jpg/2.0.0/",
    "cls_mimiccxr_trainList": "/scratch/nuislam/Model_Checkpoints/data_files_splits/ark6_classificationTask/mimic-cxr-2.0.0-train.csv",
    "cls_mimiccxr_valList": "/scratch/nuislam/Model_Checkpoints/data_files_splits/ark6_classificationTask/mimic-cxr-2.0.0-validate.csv",
    "cls_mimiccxr_testList": "/scratch/nuislam/Model_Checkpoints/data_files_splits/ark6_classificationTask/mimic-cxr-2.0.0-test.csv",

    # TBX11K - Classification - Localization
    "cls_tbx11k_root": "/scratch/jliang12/data/tbx11k/TBX11K/",
    "cls_tbx11k_trainList": "lists/TBX11K_train.txt",
    "cls_tbx11k_valList": None,
    "cls_tbx11k_testList": "lists/TBX11K_val.txt",

    "loc_tbx11k_trainTag": "tbx11k_catagnostic_train",
    "loc_tbx11k_testTag": "tbx11k_catagnostic_test",
    "loc_tbx11k_root": "/scratch/jliang12/data/tbx11k/TBX11K/imgs",
    "loc_tbx11k_trainList": "/scratch/jliang12/data/tbx11k/TBX11K/annotations/all_train_2.json",
    "loc_tbx11k_testList": "/scratch/jliang12/data/tbx11k/TBX11K/annotations/all_val_2.json",

    # NODE21 - Classification - Localization
    "cls_node21_root": "/scratch/jliang12/data/NODE21/",
    "cls_node21_trainList": "data/node21_dataset/train.txt",
    "cls_node21_valList": None,
    "cls_node21_testList": "data/node21_dataset/test.txt",

    "loc_node21_trainTag": "node21_noduleDataset_train",
    "loc_node21_testTag": "node21_noduleDataset_test",
    "loc_node21_root": "/scratch/jliang12/data/NODE21/png_images",
    "loc_node21_trainList": "/scratch/jliang12/FoundationX_localization_bbox_annotation_collections/NODE21/Node21_Nodule_Bbox_Ann_train_2.json",
    "loc_node21_testList": "/scratch/jliang12/FoundationX_localization_bbox_annotation_collections/NODE21/Node21_Nodule_Bbox_Ann_test_2.json",

    # ChestX-Det - Classification - Localization - Segmentation
    "cls_chestxdet_root": "/scratch/jliang12/data/ChestX-Det/",
    "cls_chestxdet_trainList": "/scratch/jliang12/FoundationX_localization_bbox_annotation_collections/ChestX-Det_Disease/ChestX_det_train_NAD_v2.json",
    "cls_chestxdet_valList": None,
    "cls_chestxdet_testList": "/scratch/jliang12/FoundationX_localization_bbox_annotation_collections/ChestX-Det_Disease/ChestX_det_test_NAD_v2.json",

    "loc_chestxdet_trainTag": "chestxdet_train",
    "loc_chestxdet_testTag": "chestxdet_test",
    "loc_chestxdet_trainRoot": "/scratch/jliang12/data/ChestX-Det/train",
    "loc_chestxdet_testRoot": "/scratch/jliang12/data/ChestX-Det/test",
    "loc_chestxdet_trainList": "/scratch/jliang12/FoundationX_localization_bbox_annotation_collections/ChestX-Det_Disease/ChestX_det_train_NAD_v2.json",
    "loc_chestxdet_testList": "/scratch/jliang12/FoundationX_localization_bbox_annotation_collections/ChestX-Det_Disease/ChestX_det_test_NAD_v2.json",

    "seg_chestxdet_trainRoot": "/scratch/jliang12/data/ChestX-Det/train/",
    "seg_chestxdet_trainList": "/scratch/jliang12/data/ChestX-Det/train_binary_mask/",
    "seg_chestxdet_testRoot": "/scratch/jliang12/data/ChestX-Det/test/",
    "seg_chestxdet_testList": "/scratch/jliang12/data/ChestX-Det/test_binary_mask/",

    # RSNAPneumonia - Classification - Localization
    "cls_rsnapneumonia_root": "/scratch/jliang12/data/rsna-pneumonia-detection-challenge/stage_2_train_images_png/",
    "cls_rsnapneumonia_trainList": "/scratch/nuislam/Model_Checkpoints/data_files_splits/rsna_pneumonia/RSNAPneumonia_train.txt",
    "cls_rsnapneumonia_valList": "/scratch/nuislam/Model_Checkpoints/data_files_splits/rsna_pneumonia/RSNAPneumonia_val.txt",
    "cls_rsnapneumonia_testList": "/scratch/nuislam/Model_Checkpoints/data_files_splits/rsna_pneumonia/RSNAPneumonia_test.txt",

    "loc_rsnapneumonia_trainTag": "rsnaPneumoniaDetection_Train",
    "loc_rsnapneumonia_testTag": "rsnaPneumoniaDetection_Valid",
    "loc_rsnapneumonia_root": "/scratch/jliang12/data/rsna-pneumonia-detection-challenge/stage_2_train_images_png",
    "loc_rsnapneumonia_trainList": "/scratch/jliang12/FoundationX_localization_bbox_annotation_collections/RSNA_Pneumonia_Detection_Challenge/rsnaPneumoniaDetection_Train.json",
    "loc_rsnapneumonia_valList": "/scratch/jliang12/FoundationX_localization_bbox_annotation_collections/RSNA_Pneumonia_Detection_Challenge/rsnaPneumoniaDetection_Valid.json",
    "loc_rsnapneumonia_testList": "/scratch/jliang12/FoundationX_localization_bbox_annotation_collections/RSNA_Pneumonia_Detection_Challenge/rsnaPneumoniaDetection_Test.json'",

    # SIIM-ACR - Classification - Localization - Segmentation
    "cls_siimacr_root": "/scratch/jliang12/data/siim_pneumothorax_segmentation/",
    "cls_siimacr_trainList": "/scratch/nuislam/Model_Checkpoints/data_files_splits/siimacr_ptx/SIIMPTX_cls_train.txt",
    "cls_siimacr_valList": "/scratch/nuislam/Model_Checkpoints/data_files_splits/siimacr_ptx/SIIMPTX_cls_val.txt",
    "cls_siimacr_testList": "/scratch/nuislam/Model_Checkpoints/data_files_splits/siimacr_ptx/SIIMPTX_cls_test.txt",

    "loc_siimacr_trainTag": "siimacr_train",
    "loc_siimacr_testTag": "siimacr_val",
    "loc_siimacr_trainRoot": "/scratch/jliang12/data/siim_pneumothorax_segmentation/train_jpeg",
    "loc_siimacr_valRoot": "/scratch/jliang12/data/siim_pneumothorax_segmentation/val_jpeg",
    "loc_siimacr_testRoot": "/scratch/jliang12/data/siim_pneumothorax_segmentation/test_jpeg",
    "loc_siimacr_trainList": "/scratch/nuislam/Model_Checkpoints/localization_bbox_annotation_collections/SIIM-ACR_Pneumothorax/siim_pneumothorax_train_coco.json",
    "loc_siimacr_valList": "/scratch/nuislam/Model_Checkpoints/localization_bbox_annotation_collections/SIIM-ACR_Pneumothorax/siim_pneumothorax_val_coco.json",
    "loc_siimacr_testList": "/scratch/nuislam/Model_Checkpoints/localization_bbox_annotation_collections/SIIM-ACR_Pneumothorax/siim_pneumothorax_test_coco.json",

    "seg_siimacr_trainRoot": "/scratch/jliang12/data/siim_pneumothorax_segmentation/train_jpeg",
    "seg_siimacr_trainList": "data/pxs/train.txt",
    "seg_siimacr_testRoot": "/scratch/jliang12/data/siim_pneumothorax_segmentation/test_jpeg",
    "seg_siimacr_testList": "data/pxs/test.txt",

    # CANDID-PTX - Classification - Localization - Segmentation
    "cls_candidptx_root": "/scratch/jliang12/data/CANDID-PTX/dataset/",
    "cls_candidptx_trainList": "/scratch/nuislam/Model_Checkpoints/data_files_splits/candid_ptx/CANDIDPTX_cls_train.txt",
    "cls_candidptx_valList": "/scratch/nuislam/Model_Checkpoints/data_files_splits/candid_ptx/CANDIDPTX_cls_val.txt",
    "cls_candidptx_testList": "/scratch/nuislam/Model_Checkpoints/data_files_splits/candid_ptx/CANDIDPTX_cls_test.txt",

    "loc_candidptx_trainTag": "candidptx_pneumothorax_train_full",
    "loc_candidptx_testTag": "candidptx_pneumothorax_val",
    "loc_candidptx_root": "/scratch/jliang12/data/CANDID-PTX/png",
    "loc_candidptx_trainList": "/scratch/nuislam/Model_Checkpoints/localization_bbox_annotation_collections/CANDID_PTX/CANDID_PTX_train_1.json",
    "loc_candidptx_valList": "/scratch/nuislam/Model_Checkpoints/localization_bbox_annotation_collections/CANDID_PTX/CANDID_PTX_valid_1.json",
    "loc_candidptx_testList": "/scratch/nuislam/Model_Checkpoints/localization_bbox_annotation_collections/CANDID_PTX/CANDID_PTX_test_1.json",

    "seg_candidptx_root": "/scratch/jliang12/data/CANDID-PTX/dataset",
    "seg_candidptx_trainList": "data/candid_ptx/train.txt",
    "seg_candidptx_testList": "data/candid_ptx/test.txt",


    # RSNA PE - Classification - Localization
    "cls_rsnape_root": "/scratch/nuislam/Data/RawInputImages_Exams_ODDS_EVENS_indiNP_256each_224by224",
    "cls_rsnape_trainList": "/scratch/nuislam/Data/RSNA_Exam_Odds_Evens_Train_256List.txt",
    "cls_rsnape_valList": "/scratch/nuislam/Data/RSNA_Exam_Odds_Evens_Val_256List.txt",
    "cls_rsnape_testList": None,

    "loc_rspect_trainTag": "rspect_train",
    "loc_rspect_testTag": "rspect_test",
    "loc_rspect_root": "/scratch/jliang12/data/RSNA_PE_RSPECT/localization_png_augmented_files_224by224",
    "loc_rspect_trainList": "/scratch/jliang12/data/RSNA_PE_RSPECT/RSPECT_384_Train_Loc_png_v2_50perBigger_224by224.json",
    "loc_rspect_valList": None,
    "loc_rspect_testList": "/scratch/jliang12/data/RSNA_PE_RSPECT/RSPECT_61_Test_Loc_png_v2_50perBigger_224by224.json",

    # RadFusion PE - Classification
    "cls_radfusion_trainRoot": "/scratch/jliang12/data/CT_RadFusion/RawInputImages_Exams_ODDS_EVENS_indiNP_256each_224by224",
    "cls_radfusion_testRoot": "/scratch/jliang12/data/CT_RadFusion/RawInputImages_Exams_ODDS_EVENS_indiNP_256each_TEST_224by224",
    "cls_radfusion_trainList": "/scratch/jliang12/data/CT_RadFusion/RadFusion_Exam_Odds_Evens_Train+Val_256List_v1.txt",
    "cls_radfusion_valList": "/scratch/jliang12/data/CT_RadFusion/RadFusion_Exam_Odds_Evens_Test_256List.txt",
    "cls_radfusion_testList": None,

    # INSPECT - Classification
    "cls_inspect_Root": "/scratch/jliang12/data/CT_INSPECT_26gb/INSPECT_Exams_ODDs_EVENs_indiNP_256each_224by224_PixelOrg/",
    "cls_inspect_testRoot": "/data/jliang12/nuislam/CT_INSPECT_26gb/INSPECT_Official_TrainList.csv",
    "cls_inspect_trainList": None,
    "cls_inspect_valList": "/data/jliang12/nuislam/CT_INSPECT_26gb/INSPECT_Official_TestList.csv",
    "cls_inspect_testList": None,

    # FUMPE - Classification - Localization - Segmentation
    "cls_fumpe_root": "/scratch/jliang12/data/FUMPE/",
    "cls_fumpe_trainList": "",
    "cls_fumpe_valList": "",
    "cls_fumpe_testList": None,

    "loc_fumpe_trainTag": "fumpe_train",
    "loc_fumpe_testTag": "fumpe_test",
    "loc_fumpe_root": "/scratch/jliang12/data/FUMPE/png_images_224by224",
    "loc_fumpe_trainList": "/scratch/jliang12/data/FUMPE/Train_BBox_List_v2_50perBigger_224by224.json",
    "loc_fumpe_valList": None,
    "loc_fumpe_testList": "/scratch/jliang12/data/FUMPE/Test_BBox_List_v2_50perBigger_224by224.json",

    "seg_fumpe_rootDir": "/scratch/jliang12/data/FUMPE/",
    "seg_fumpe_root": "/scratch/jliang12/data/FUMPE/np_images_224by224/",
    "seg_fumpe_mask": "np_masks_224by224/",

    # PE CAD 91 - Classification - Localization - Segmentation
    "cls_pecad91_root": "/scratch/jliang12/data/PE_CAD_91_Challenge/",
    "cls_pecad91_trainList": "",
    "cls_pecad91_valList": "",
    "cls_pecad91_testList": None,

    "loc_pecad91_trainTag": "pecad91_train",
    "loc_pecad91_testTag": "pecad91_test",
    "loc_pecad91_root": "/scratch/jliang12/data/PE_CAD_91_Challenge/png_images_v3_224by224",
    "loc_pecad91_trainList": "/scratch/jliang12/data/PE_CAD_91_Challenge/Train_BBox_List_v2_50perBigger_v2_224by224.json",
    "loc_pecad91_valList": None,
    "loc_pecad91_testList": "/scratch/jliang12/data/PE_CAD_91_Challenge/Test_BBox_List_v2_50perBigger_v2_224by224.json",

    "seg_pecad91_rootDir": "/scratch/jliang12/data/PE_CAD_91_Challenge/",
    "seg_pecad91_root": "/scratch/jliang12/data/PE_CAD_91_Challenge/np_images_v3_224by224/",
    "seg_pecad91_mask": "/scratch/jliang12/data/PE_CAD_91_Challenge/np_masks_v2_224by224/",

    # # x
    # "cls_x_root": "",
    # "cls_x_trainList": "",
    # "cls_x_valList": "",
    # "cls_x_testList": "",
}