# Comprehensive-Analysis-and-Resolution-of-Data-Imbalance-in-CRISPR-Off-Target-Prediction-Tasks
![Figure 1](https://github.com/gzrgzx/Comprehensive-Analysis-and-Resolution-of-Data-Imbalance-in-CRISPR-Off-Target-Prediction-Tasks/assets/48210803/9bdc4ab2-1957-443a-81d1-410f130e826a)
![Figure 2](https://github.com/gzrgzx/Comprehensive-Analysis-and-Resolution-of-Data-Imbalance-in-CRISPR-Off-Target-Prediction-Tasks/assets/48210803/fa0dd79b-4fcc-47d7-a03b-2915f8e9da50)
PREREQUISITE

CrisprDNT was conducted by TensorFlow version 2.3.2 and python version 3.6.

Following Python packages should be installed:

* numpy
- pandas
* scikit-learn
- TensorFlow
* Keras

Data Description:

* dataset1->Doench et al.(Protein knockout detection)
* dataset2->Haeussler et al.(PCR, Digenome-Seq and HTGTS)
* dataset3->Cameron et al.(SITE-Seq)
* dataset4->Tasi et al.(GUIDE-seq)
* dataset5->Kleinstiver et al(GUIDE-seq)
* dataset6->Listgarten et al.(GUIDE-seq)
* dataset7->Chuai et al.(GUIDE-Seq, Digenome-Seq,HTGTS,BLESS and IDLVs)
* dataset8->Chuai et al.(GUIDE-Seq, Digenome-Seq,HTGTS,BLESS and IDLVs)

*Code Description
    * data_process(coding scheme)
        * create_coding_scheme.py->Create CrisprDNT, CRISPR_IP, CRISPR_Net and CNN_std encoding.
    * model
        * model_network.py->CrisprDNT, CRISPR_IP, CRISPR_Net and CNN_std network and data imbalance code.
    * train&test
        * experiment.py->code to reproduce the experiments with CrisprDNT, CRISPR_IP, CRISPR_Net, and CNN_std.
