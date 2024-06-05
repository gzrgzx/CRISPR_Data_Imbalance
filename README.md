# A Systematic Method for Solving Data Imbalance in CRISPR Off-Target Prediction Tasks

![Figure 1](https://github.com/gzrgzx/Comprehensive-Analysis-and-Resolution-of-Data-Imbalance-in-CRISPR-Off-Target-Prediction-Tasks/assets/48210803/9bdc4ab2-1957-443a-81d1-410f130e826a)
![Figure 2](https://github.com/gzrgzx/CRISPR_Data_Imbalance/assets/48210803/eb1b27f8-fb97-43be-b2ae-205751e69d82)

PREREQUISITE

CrisprDNT was conducted by TensorFlow version 2.3.2 and python version 3.6.

Following Python packages should be installed:

* numpy
- pandas
* scikit-learn
- TensorFlow
* Keras


Usage

For a new dataset：
* You can process the dataset by calling code/data_process/create_coding_scheme.py to get the coding format needed by CrisprDNT, CRISPR_IP, CRISPR_Net, CNN_std, CnnCrispr and DL-CRISPR encoding.
* The network structure of the six models CrisprDNT, CRISPR_IP, CRISPR_Net, CNN_std, CnnCrispr and DL-CRISPR as well as the data imbalance method are available from code/train&test/experiment.py and code/model/model_network.py.


Data Description:

* dataset1->Doench et al.(Protein knockout detection)
* dataset2->Haeussler et al.(PCR, Digenome-Seq and HTGTS)
* dataset3->Cameron et al.(SITE-Seq)
* dataset4->Tasi et al.(GUIDE-seq)
* dataset5->Kleinstiver et al(GUIDE-seq)
* dataset6->Listgarten et al.(GUIDE-seq)
* dataset7->Chuai et al.(GUIDE-Seq, Digenome-Seq,HTGTS,BLESS and IDLVs)
* dataset8->Chuai et al.(GUIDE-Seq, Digenome-Seq,HTGTS,BLESS and IDLVs)

* Code Description
    * data_process(coding scheme)
        * create_coding_scheme.py->Create CrisprDNT, CRISPR_IP, CRISPR_Net, CNN_std, CnnCrispr and DL-CRISPR encoding.
    * model
        * newnetwork.py->CrisprDNT, CRISPR_IP, CRISPR_Net, CNN_std, CnnCrispr and DL-CRISPR network and data imbalance code.
    * train&test
        * experiment_2.py->code to reproduce the experiments with CrisprDNT, CRISPR_IP, CRISPR_Net, CNN_std, CnnCrispr and DL-CRISPR.


saved_model Description(The saved model is exemplified by CrisprDNT, where the model is trained on the first type of dataset and tested on the second type of dataset respectively):
* encodedmismatchtype14x231&2&3&4_Listgarten_22gRNAwithoutTsai.pkl+CrisprDNT_GHM.h5->Processing the CrisprDNT model using GHM techniques.
* encodedmismatchtype14x231&2&3&4_Listgarten_22gRNAwithoutTsai.pkl+CrisprDNT_focal_loss.h5->Processing the CrisprDNT model using focal_loss techniques.
* encodedmismatchtype14x231&2&3&4_Listgarten_22gRNAwithoutTsai.pkl+CrisprDNT_undersampling.h5->Processing the CrisprDNT model using undersampling techniques.
* encodedmismatchtype14x231&2&3&4_Listgarten_22gRNAwithoutTsai.pkl+CrisprDNT_oversampling.h5->Processing the CrisprDNT model using oversampling techniques.
* encodedmismatchtype14x231&2&3&4_Listgarten_22gRNAwithoutTsai.pkl+CrisprDNT_ADASYN.h5->Processing the CrisprDNT model using ADASYN techniques.
* encodedmismatchtype14x231&2&3&4_Listgarten_22gRNAwithoutTsai.pkl+CrisprDNT_SMOTE.h5->Processing the CrisprDNT model using SMOTE techniques.
