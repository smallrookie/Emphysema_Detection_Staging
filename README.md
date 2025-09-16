# Self-Supervised Anomaly Detection and Staging for CT Emphysema

## Abstract
Emphysema, a diffuse and heterogeneous phenotype of chronic obstructive pulmonary disease (COPD), carries substantial morbidity and elevates lung cancer risk. CT facilitates detection and progression assessment, yet existing deep learning methods rely on large-scale annotated datasets. Unsupervised anomaly detection (UAD) methods offer an alternative but struggle with general synthetic anomalies and weak emphysema semantics. In this study, we propose a self-supervised framework trained exclusively on non-emphysema CT scans, using synthetically generated lesions with emphysema-inspired morphology to guide pixel-level anomaly modeling. We introduce EDLNet, an encoder-decoder architecture with spatial-channel refinement and adaptive feature fusion, and we apply an unsupervised manner for emphysema staging. Multi-center evaluations show that our framework outperforms existing UAD approaches in detection and localization, while achieving a mean staging accuracy of 93.13% and a macro AUROC of 99.08%. These results demonstrate the potential of annotation-free emphysema modeling for clinical application.

## Quick Start
### 1. Installation
```bash
git clone https://github.com/smallrookie/Emphysema_Detection_Staging.git
cd Emphysema_Detection_Staging
conda create -n emphysema_detection_staging python=3.10
conda activate emphysema_detection_staging
pip install -r requirements.txt
```

### 2. Training
```bash
# training
python train.py --gate --fup --scrb --tr_path your_dataset_path --save your_save_path
```

## Datasets

[Dataset C (INSPECT)](https://som-shahlab.github.io/inspect-website) and [D (CT-RATE)](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE) are accessible online. The full emphysema annotations used in this work are not publicly available; instead, we provide partial annotations as illustrative examples, available at: [Google Drive](https://drive.google.com/drive/folders/1oB5nJVe1-iPJPFQ_1rXGxYWbqktwXlXs?usp=drive_link).

## Contact
For questions or collaborations:

- Email: rookiefcb@gmail.com
