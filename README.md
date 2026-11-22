.
├── scripts/
│   ├── dimensional_model/
│   │   ├── Dimensional_bert.ipynb          # BERT VAD training & inference (Google Colab / GPU)
│   │   ├── data/                           # EmoBank training data
│   │   └── output/                         # VAD predictions (*_OUTIP.xlsx, *_VY.xlsx)
│   ├── componential_model/
│   │   ├── componential_roberta.ipynb      # RoBERTa GoEmotions training & inference (Google Colab / GPU)
│   │   ├── data/                           # GoEmotions train/dev/test splits
│   │   └── output/                         # Discrete emotion predictions (*_OUTIP.xlsx, *_VY.xlsx)
│   └── STM/
│       ├── Data/                           # OUTIP & VulnYouth English corpora and CORA survey CSV
│       ├── STM_ott26_OUTIP.R               # Text preprocessing & 20-topic STM estimation (OUTIP)
│       ├── STM_ott26_VY.R                  # Survey join, preprocessing & 20-topic STM estimation (VY)
│       ├── 0_creating_dataset_OUTIP.R      # Merges VAD, GoEmotions, and STM topics (OUTIP)
│       ├── 0_creating_dataset_VY.R         # Merges VAD, GoEmotions, CORA survey, and STM topics (VY)
│       ├── analysis_OUTIP.R                # Statistical analysis & visualizations (OUTIP)
│       ├── analysis_VY.R                   # Statistical analysis & visualizations (VY)
│       └── output/
│           ├── OUTIP/                      # STM models (.rds), merged datasets, summaries, figures/
│           └── VY/                         # STM models (.rds), merged datasets, summaries, figures/
├── .gitignore
└── README.md