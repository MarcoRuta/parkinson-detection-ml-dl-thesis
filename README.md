# parkinson_detection_spiral_test


```
├─ dataset
│  ├─ augmented_dataset @dataset in cui a run time vengono salvate e successivamente eliminate le immagini prodotte attraverso data augmentation
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
│  │ 
│  │     └─ training
│  │        ├─ healthy
│  │        └─ parkinson
│  │ 
│  │ 
│  │    
│  └─ numerical_dataset @dataset UCI, numerico
│     │ 
│     │
│     ├─ extracted_data.csv @features estratte dal dataset UCI attraverso lo script data_extraction.py
│     │
│     ├─ extracted_data_fancy.xlsxp @versione human friendly di extracted_data
│     │
│     ├─ hw_dataset @dati numerici
│     │  ├─ control
│     │  └─ parkinson
│     │
│     └─ hw_drawings @immagini relative ai dati numerici, sono presenti solo in parte (25/78)
│        ├─ Dynamic Spiral Test
│        └─ Static Spiral Test
│             
│  
└─ scripts
   ├─ image_processing @script utilizzati per eseguire image processing sul dataset NIATS
   │  ├─ CNN_classification.py @data augmentation, estrazione HOG e classificazione attraverso rete neurale
   |  ├─ image processing.py @data augmentation, estrazione HOG e classificazione attraverso diversi classificatori
   │  ├─ spiral_weights.h5 @pesi della rete neurale
   │  └─ V01HE01.png @immagine presa dal dataset
   │  
   │  
   │  
   └─ numerical_features
      ├─ classification.py @estrazione delle features dai dati del dataset UCI
      └─ data_extraction.py @classificazione attraverso diversi classificatori

Nella cartella results sono riportate le immagini con i risultati per:

- classificatori su dataset UCI
- classificatori su dataset NIATS
- rete neurale su dataset NIATS
```