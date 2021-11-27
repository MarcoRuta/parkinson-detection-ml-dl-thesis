# parkinson_detection_spiral_test


```
├─ dataset
│  ├─ augmented_dataset 
│  │  ├─ testing
│  │  │  ├─ healthy
│  │  │  └─ parkinson
│  │  └─ training
│  │     ├─ healthy
│  │     └─ parkinson
│  │ 
│  │ 
│  │ 
│  ├─ image_dataset @dataset NIATS, di sole immagini
│  │  └─ spiral
│  │     ├─ testing
│  │     │  ├─ healthy
│  │     │  └─ parkinson
│  |     |  
│  │     └─ training
│  │        ├─ healthy
│  │        └─ parkinson
│  │ 
│  │ 
│  │    
│  └─ numerical_dataset @dataset UCI, numerico
│     │ 
│     │
│     ├─ extracted_data.csv 
│     │
│     ├─ extracted_data_fancy.xlsxp 
│     │
│     ├─ hw_dataset 
│     │  ├─ control
│     │  └─ parkinson
│     │
│     └─ hw_drawings 
│        ├─ Dynamic Spiral Test
│        └─ Static Spiral Test
│             
│  
├─ README.md
│
│
│
├─ results
│  ├─ CNN Results
│  │  ├─ epoch_ custom CNN.png
│  │  ├─ epoch_ResNet.png
│  │  ├─ epoch_VGG.png
│  │  ├─ history_custom CNN.png
│  │  ├─ history_ResNet.png
│  │  ├─ history_VGG.png
│  │  ├─ ROC_custom CNN.png
│  │  ├─ ROC_ResNet.png
│  │  └─ ROC_VGG.png
│  ├─ image_processing_results
│  │  ├─ hog_example.png
│  │  └─ image_processing_res.png
│  └─ numerical_features_results
│     ├─ corr_matrix.png
│     └─ numerical_feature_res.png
│
│
│
├─ scripts
│  ├─ image_processing
│  │  ├─ CNN_classification.py
│  │  ├─ custom_cnn.h5
│  │  ├─ image processing.py 
│  │  ├─ Resnet_classification.py
│  │  ├─ resnet_spiral.h5
│  │  ├─ spiral_weights.h5
│  │  ├─ V01HE01.png
│  │  ├─ vgg16_spiral.h5
│  │  └─ VGG_classification.py
│  │
│  │
│  └─ numerical_features
│     ├─ classification.py
│     └─ data_extraction.py
└─ tesi
   └─ Capitoli 1 e 3.pdf

```