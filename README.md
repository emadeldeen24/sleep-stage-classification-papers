# sleep-stages-classification-papers

### This repo contains a list of papers for sleep stage classification using deep learning based techniques starting from 2017-afterwards.
### I classified them *generally* according to the proposed DL technique.
If you have any suggested papers, please contact me <br/>
emad0002{at}e.ntu.edu.sg



## Review papers
year | Paper | PDF
------------ | ------------- | ------------ 
2022 | Representations of temporal sleep dynamics: Review and synthesis of the literature <br/> *Sleep Medicine Reviews* | [PDF](https://www.sciencedirect.com/science/article/pii/S1087079222000247)     
2022 | A comprehensive evaluation of contemporary methods used for automatic sleep staging <br/> *Biomedical Signal Processing and Control* | [PDF](https://www.sciencedirect.com/science/article/abs/pii/S174680942200341X).    
2021 | Automatic Sleep Staging: Recent Development, Challenges, and Future Directions  <br/> *Arxiv* | [PDF](https://arxiv.org/pdf/2111.08446.pdf)    
2020 | Automated Detection of Sleep Stages Using Deep Learning Techniques: A Systematic Review of the Last Decade (2010–2020) <br/> *Applied Sciences* | [PDF](https://www.mdpi.com/2076-3417/10/24/8963)   
2019 | Automated sleep scoring: A review of the latest approaches <br/> *Sleep Medicine Reviews* | [PDF](https://www.sciencedirect.com/science/article/abs/pii/S1087079218301746) 
2019 | A review of automated sleep stage scoring based on physiological signals for the new millennia <br/> *Computer Methods and Programs in Biomedicine* | [PDF](https://www.sciencedirect.com/science/article/abs/pii/S0169260718313865)   



## CNN only
year | Paper | PDF | Code
------------ | ------------- | ------------ | -------------
2022 | Enhancing Contextual Encoding With Stage-Confusion and Stage-Transition Estimation for EEG-Based Sleep Staging ([v2](https://arxiv.org/pdf/2203.12590.pdf)) <br/> *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)* | [PDF](https://ieeexplore.ieee.org/abstract/document/9746353) | [github](https://github.com/ku-milab/TransSleep)       
2021 | SalientSleepNet: Multimodal Salient Wave Detection Network for Sleep Staging <br/> *IJCAI-21* | [PDF](https://www.ijcai.org/proceedings/2021/0360.pdf) | [github](https://github.com/ziyujia/SalientSleepNet)  
2020 | SleepPrintNet: A Multivariate Multimodal Neural Network based on Physiological Time-series for Automatic Sleep Staging <br/> *IEEE Transactions on Artificial Intelligence* | [PDF](https://ieeexplore.ieee.org/document/9357954) | [github](https://github.com/xiyangcai/SleepPrintNet)    
2020 | Orthogonal convolutional neural networks for automatic sleep stage classification based on single-channel EEG <br/> *Computer Methods and Programs in Biomedicine* | [PDF](https://www.sciencedirect.com/science/article/pii/S0169260719311617) | -    
2020 | TRIER: Template-Guided Neural Networks for Robust and Interpretable Sleep Stage Identification from EEG Recordings <br/> *Arxiv* | [PDF](https://arxiv.org/pdf/2009.05407.pdf) | -     
2020 | Computation-Efficient Multi-Model Deep Neural Network for Sleep Stage Classification <br/> *Asia Service Sciences and Software Engineering Conference* | [PDF](https://dl.acm.org/doi/abs/10.1145/3399871.3399887) | -    
2019 | U-Time: A Fully Convolutional Network for Time Series Segmentation Applied to Sleep Staging <br/>  *Advances in Neural Information Processing Systems (NeurIPS)* | [PDF](https://arxiv.org/pdf/1910.11162.pdf) | [github](https://github.com/perslev/U-Time)
2019 | Deep learning for automated feature discovery and classification of sleep stages <br/> *IEEE Transactions On Computational Biology and Bioinformatics* | [PDF](https://pdfs.semanticscholar.org/d8ae/c06b259bbdb79e9e55b6eb04c7060ce9b55c.pdf) | -     
2019 | A Deep Learning Model for Automated Sleep Stages Classification Using PSG Signals <br/> *International Journal of Environmental Research and Public Health* | [PDF](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6406978/) | -     
2019 | Joint Classification and Prediction CNN Framework for Automatic Sleep Stage Classification <br/> *IEEE Transactions on Biomedical Engineering* | [PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8502139) | [github](https://github.com/pquochuy/MultitaskSleepNet)    
2019 | An Image Based Prediction Model for Sleep Stage Identification <br/> *IEEE International Conference on Image Processing (ICIP)* | [PDF](https://ieeexplore.ieee.org/abstract/document/8803026) | -     
2018 | DNN Filter Bank Improves 1-Max Pooling CNN for Single-Channel EEG Automatic Sleep Stage Classification <br/> *IEEE Engineering in Medicine and Biology Society (EMBC)* | [PDF](https://ieeexplore.ieee.org/document/8512286) | -     
2018 | Deep residual networks for automatic sleep stage classification of raw polysomnographic waveforms <br/> *IEEE Engineering in Medicine and Biology Society (EMBC)* | [PDF](https://arxiv.org/pdf/1810.03745.pdf) | -    

## CNN with LSTM
year | Paper | PDF | Code
------------ | ------------- | ------------ | -------------
2021 | CCRRSleepNet: A Hybrid Relational Inductive Biases Network for Automatic Sleep Stage Classification on Raw Single-Channel EEG <br/> *Brain Sciences*| [PDF](https://www.mdpi.com/2076-3425/11/4/456) | [github](https://github.com/nengwp/CCRRSleepNet)
2020 | An Automatic Sleep Staging Method Using a Multi-head and Sequence Network <br/> *International Conference on Biological Information and Biomedical Engineering*| [PDF](https://dl.acm.org/doi/pdf/10.1145/3403782.3403797) | -    
2020 | Intra- and inter-epoch temporal context network (IITNet) using sub-epoch features for automatic sleep scoring on raw single-channel EEG <br/> *Biomedical Signal Processing and Control*| [PDF](https://www.sciencedirect.com/science/article/pii/S1746809420301932) | -    
2020 | TinySleepNet: An Efficient Deep Learning Model for Sleep Stage Scoring based on Raw Single-Channel EEG <br/> *International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC)* | [PDF](https://ieeexplore.ieee.org/document/9176741) | [github](https://github.com/akaraspt/tinysleepnet)    
2020 | Temporal dependency in automatic sleep scoring via deep learning based architectures: An empirical study <br/> *International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC)*| [PDF](https://ieeexplore.ieee.org/abstract/document/9176356)  | -   
2019 | End-to-end Sleep Staging with Raw Single Channel EEG using Deep Residual ConvNets <br/> *IEEE EMBS International Conference on Biomedical & Health Informatics (BHI)*| [PDF](https://arxiv.org/pdf/1904.10255.pdf)  | -   
2018 | A Structured Learning Approach with Neural Conditional Random Fields for Sleep Staging <br/> *IEEE International Conference on Big Data*| [PDF](https://arxiv.org/pdf/1807.09119.pdf) | -    
2018 | Expert-level sleep scoring with deep neural networks <br/> *Journal of the American Medical Informatics Association*| [PDF](https://academic.oup.com/jamia/article/25/12/1643/5185596) | -    
2018 | Deep Convolutional Network Method for Automatic Sleep Stage Classification Based on Neurophysiological Signals <br/> *International Congress on Image and Signal Processing, BioMedical Engineering and Informatics (CISP-BMEI)*| [PDF](https://ieeexplore.ieee.org/abstract/document/8633058) | -    
2018 | Learning Sleep Stages from Radio Signals: A Conditional Adversarial Architecture <br/> *International Conference on Machine Learning (ICML)*| [PDF](http://sleep.csail.mit.edu/files/rfsleep-paper.pdf) | -     
2017 | A deep learning architecture for temporal sleep stage classification using multivariate and multimodal time series <br/> *IEEE Transactions on Neural Systems and Rehabilitation Engineering*| [PDF](https://arxiv.org/pdf/1707.03321.pdf) | -    
2017 | DeepSleepNet: a Model for Automatic Sleep Stage Scoring based on Raw Single-Channel EEG  <br/> *IEEE Transactions on Neural Systems and Rehabilitation Engineering* | [PDF](https://arxiv.org/pdf/1703.04046.pdf) | [python2](https://github.com/akaraspt/deepsleepnet) [python3](https://github.com/genaris/deepsleepnet) 



## CNN with LSTM with attention
year | Paper | PDF | Code
------------ | ------------- | ------------ | -------------
2020 | XSleepNet: Multi-View Sequential Model for Automatic Sleep Staging <br/> *IEEE Transactions on Pattern Analysis and Machine Intelligence* | [PDF](https://arxiv.org/pdf/2007.05492.pdf) | -    
2020 | An Automatic Sleep Staging Model Combining Feature Learning and Sequence Learning <br/> *International Conference on Advanced Computational Intelligence (ICACI)*| [PDF](https://ieeexplore.ieee.org/abstract/document/9177520) | -    
2019 | SeqSleepNet: End-to-End Hierarchical Recurrent Neural Network for Sequence-to-Sequence Automatic Sleep Staging <br/> *IEEE Transactions on Neural Systems and Rehabilitation Engineering* | [PDF](https://arxiv.org/pdf/1809.10932.pdf) | [github](https://github.com/pquochuy/SeqSleepNet) 
2019 | SleepEEGNet: Automated sleep stage scoring with sequence to sequence deep learning approach <br/> *Plos One* | [PDF](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0216456) | [github](https://github.com/MousaviSajad/SleepEEGNet)  
2018 | Automatic Sleep Stage Classification Using Single-Channel EEG: Learning Sequential Features with Attention-Based Recurrent Neural Networks <br/> *IEEE Engineering in Medicine and Biology Society (EMBC)* | [PDF](https://ieeexplore.ieee.org/document/8512480) | -    



## Attention-based
year | Paper | PDF | Code
------------ | ------------- | ------------ | -------------
2021 | An Attention-based Deep Learning Approach for Sleep Stage Classification with Single-Channel EEG <br/> *IEEE Transactions on Neural Systems and Rehabilitation Engineering* | [PDF](https://ieeexplore.ieee.org/document/9417097) | [github](https://github.com/emadeldeen24/AttnSleep)     
2020 | Convolution- and Attention-Based Neural Network for Automated Sleep Stage Classification <br/> *International Journal of Environmental Research and Public Health*| [PDF](https://www.mdpi.com/1660-4601/17/11/4152)  | -     
2020 | A Residual Based Attention Model for EEG Based Sleep Staging <br/> *IEEE Journal of Biomedical and Health Informatics*| [PDF](https://ieeexplore.ieee.org/document/9022981) | -   
2018 | Multivariate Sleep Stage Classification using Hybrid Self-Attentive Deep Learning Networks <br/> *IEEE International Conference on Bioinformatics and Biomedicine (BIBM)*| [PDF](https://cse.buffalo.edu/~lusu/papers/BIBM2018.pdf) | -     


## Transfer learning
year | Paper | PDF | Code
------------ | ------------- | ------------ | -------------
2021 | RobustSleepNet: Transfer learning for automated sleep staging at scale <br/> *IEEE Engineering in Medicine and Biology Society (EMBC)*| [PDF](https://arxiv.org/abs/2101.02452) | [github](https://github.com/Dreem-Organization/RobustSleepNet)       
2020 | MetaSleepLearner: A Pilot Study on Fast Adaptation of Bio-signals-Based Sleep Stage Classifier to New Individual Subject Using Meta-Learning <br/> *IEEE Journal of Biomedical and Health Informatics*| [PDF](https://arxiv.org/pdf/2004.04157.pdf) | [github](https://github.com/IoBT-VISTEC/MetaSleepLearner)      
2019 | Towards More Accurate Automatic Sleep Staging via Deep Transfer Learning <br/> *IEEE Transactions on Biomedical Engineering* | [PDF](https://arxiv.org/pdf/1907.13177.pdf) | [github](https://github.com/pquochuy/sleep_transfer_learning)   
2018 | Multichannel Sleep Stage Classification and Transfer Learning using Convolutional Neural Networks <br/> *IEEE Engineering in Medicine and Biology Society (EMBC)*| [PDF](https://ieeexplore.ieee.org/document/8512214) | -     
2017 | deep convolutional neural networks for interpretable analysis of eeg sleep stage scoring <br/> *International Workshop on Machine Learning for Signal Processing (MLSP)* | [PDF](https://arxiv.org/pdf/1710.00633.pdf) | -     


## Personalized sleep staging
year | Paper | PDF | Code
------------ | ------------- | ------------ | -------------
2020 | Personalized automatic sleep staging with single-night data: a pilot study with Kullback–Leibler divergence regularization <br/> *IOP science*| [PDF](https://iopscience.iop.org/article/10.1088/1361-6579/ab921e)  
2018 | Personalizing deep learning models for automatic sleep staging <br/> *Arxiv*| [PDF](https://arxiv.org/pdf/1801.02645.pdf)  


## Domain Adaptation
year | Paper | PDF | Code
------------ | ------------- | ------------ | -------------
2022 | ADAST: Attentive Cross-domain EEG-based Sleep Staging Framework with Iterative Self-Training <br/> *IEEE Transactions on Emerging Topics in Computational Intelligence* | [PDF](https://arxiv.org/pdf/2107.04470.pdf) | [github](https://github.com/emadeldeen24/ADAST)    
2022 | From unsupervised to semi-supervised adversarial domain adaptation in electroencephalography-based sleep staging <br/> *Journal of Neural Engineering* | [PDF](https://iopscience.iop.org/article/10.1088/1741-2552/ac6ca8/meta) | -     
2021 | Transferring structured knowledge in unsupervised domain adaptation of a sleep staging network <br/> *IEEE Journal of Biomedical and Health Informatics* | [PDF](https://ieeexplore.ieee.org/abstract/document/9513578) | -    
2021 | Unsupervised sleep staging system based on domain adaptation <br/> *Biomedical Signal Processing and Control* | [PDF](https://www.sciencedirect.com/science/article/abs/pii/S1746809421005346) | -     
2020 | Attentive Adversarial Network for Large-Scale Sleep Staging <br/> *Machine Learning for Healthcare (MLHC)*| [PDF](http://proceedings.mlr.press/v126/nasiri20a/nasiri20a.pdf) | -    



## Self-Supervised Learning
year | Paper | PDF | Code
------------ | ------------- | ------------ | -------------
2021 | Time-Series Representation Learning via Temporal and Contextual Contrasting <br/> *IJCAI-21* | [PDF](https://www.ijcai.org/proceedings/2021/0324.pdf) | [github](https://github.com/emadeldeen24/TS-TCC)     
2021 | Self-supervised Electroencephalogram Representation Learning for Automatic Sleep Staging <br/> *Arxiv* | [PDF](https://arxiv.org/ftp/arxiv/papers/2110/2110.15278.pdf) | [github](https://github.com/ycq091044/ContraWR)   
2021 | SleepPriorCL: Contrastive Representation Learning with Prior Knowledge-based Positive Mining and Adaptive Temperature for Sleep Staging <br/> *Arxiv* | [PDF](https://arxiv.org/pdf/2110.09966.pdf) | -       
2021 | Self-Supervised Learning for Sleep Stage Classification with Predictive and Discriminative Contrastive Coding <br/> *IEEE International Conference on Acoustics, Speech and Signal Processing* | [PDF](https://ieeexplore.ieee.org/document/9414752) | -     
2021 | Self-supervised Contrastive Learning for EEG-based Sleep Staging <br/> *International Joint Conference on Neural Networks (IJCNN)* | [PDF](https://ieeexplore.ieee.org/document/9533305) | [github](https://github.com/XueJiang16/ssl-torch)    



## Manual features + Deep learning
year | Paper | PDF | Code
------------ | ------------- | ------------ | -------------
2020 | Automatic Sleep Stage Classification using Marginal Hilbert Spectrum Features and a Convolutional Neural Network <br/> *International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC)*| [PDF](https://ieeexplore.ieee.org/abstract/document/9175460) | -   
2018| Mixed Neural Network Approach for Temporal Sleep Stage Classification <br/> *IEEE Transactions on Neural Systems and Rehabilitation Engineering*| [PDF](https://arxiv.org/pdf/1610.06421.pdf) | -   
2017 | SLEEPNET: Automated Sleep Staging System via Deep Learning <br/> *Arxiv*| [PDF](https://arxiv.org/pdf/1707.08262.pdf) | -   



## Graph 
year | Paper | PDF | Code
------------ | ------------- | ------------ | -------------
2022 | An Attention-Guided Spatiotemporal Graph Convolutional Network for Sleep Stage Classification <br/> *Life* | [PDF](https://www.mdpi.com/2075-1729/12/5/622/htm) | -     
2021 | Multi-View Spatial-Temporal Graph Convolutional Networks with Domain Generalization for Sleep Stage Classification <br/> *IEEE Transactions on Neural Systems and Rehabilitation Engineering* | [PDF](https://arxiv.org/pdf/2109.01824v1.pdf) | [github](https://github.com/ziyujia/MSTGCN)    
2020 | A Graph-Temporal fused dual-input Convolutional Neural Network for Detecting Sleep Stages from EEG Signals <br/> *IEEE Transactions on Circuits and Systems II: Express Briefs*| [PDF](https://ieeexplore.ieee.org/abstract/document/9159668) | -     
2020 | GraphSleepNet: Adaptive Spatial-Temporal Graph Convolutional Networks for Sleep Stage Classification <br/> *IJCAI-20* | [PDF](https://www.ijcai.org/Proceedings/2020/0184.pdf) | [github](https://github.com/ziyujia/GraphSleepNet)    



## Data Augmentations
year | Paper | PDF | Code
------------ | ------------- | ------------ | -------------
2020 | EEG data augmentation: towards class imbalance problem in sleep staging tasks <br/> *IOP science*| [PDF](https://iopscience.iop.org/article/10.1088/1741-2552/abb5be/meta) | -     
