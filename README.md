# Multimodal-Egocentric-Action-Recognition
**Date**: 19/02/2024

**Authors**: Manuel Escobar Ferrer, Pablo Muñoz Salazar, Santiago Romero Aristizabal

For the complete paper, please refer to the `paper.pdf` file.

## Introduction
The action recognition task on computer vision is related to the categorization and identification of human actions in a video sequence. Focus on this area [8,11,14] has emerged due to its application on many areas, such as security and behavioral analysis, and due to the significant advances in computer vision of the last decade.

Based on the current efforts on action recognition and the increasing availability of sensors that can be mounted on an actor (e.g. GoPro, Google Lenses, Meta Smart Glasses), a major sub-field has emerged called Egocentric Action Recognition (EAR), focusing on first-person point of view (POV) scenarios. Recent first-person video datasets [4, 10, 13] provide huge amounts of data to be used for train- ing EAR models. Additionally to visual data, multi-modal approaches taking advantage of non-visual sensors, such as Electromyography (EMG) [10], can also be used.

Apart from classification, EAR can be used for several other tasks, such as R3D [16], which uses a pre-trained model on diverse human video data to learn robotic manip- ulation tasks. This model uses the Ego4D dataset [13], con- taining different modalities such as audio, IMU, 3D pose, different POVs, etc. Other applications, such as IMU2CLIP [15] and AUDIO2CLIP, use IMU sensors and audio to generate video and text explanations.

Therefore, the primary goal of our study is to get familiar with the EAR concept, datasets, and the subsequent imple- mentation of EAR on two common datasets: Epic-Kitchens [4] and ActionSense [10]. Specifically, we intend to explore two modalities: RGB streams of first-person point of view (POV) videos and EMG data, which has not been explored extensively on this context.

Using RGB streams of data, previous studies have ob- tained remarkable results on the video action recognition task [8, 14, 16]. Although all of them use RGB frames, which is the main modality of videos and images, some of them use different modalities, such as audio, optical flow, and/or gaze. Another major modality is optical flow, with recent promising approaches that combine RGB, Flow, and audio [12].

As many different models, datasets, training parameters, and evaluation standards are used, it is difficult to establish general baseline for action recognition tasks. Nevertheless, works such as [1], give a clear idea of the best performing models and guidelines. For example, the performance of a model can be affected by important parameters such as the number of frames to be analysed per clip or the backbone used by the feature extraction model. Therefore, this study provides some insights related to the settings that will be used for the feature extraction and training of the classifier described for this project.

Based on this, we will use a pre-trained I3D model on the kinetics dataset [19] for action recognition, which uses an Inception backbone as in [8]. While recent advance- ments in action recognition have led to models achieving impressive accuracy [1, 3, 5, 7], I3D was chosen due to its popularity and extensive use. This model will be used as an RGB feature extractor for the Epic-Kitchens [4] and Action- Sense [10] datasets. As of the classification task, different models will be implemented, analyzing their performances on the Epic-Kitchens [4] dataset. Then, we extended this classification to the ActionSense dataset [10], combining the previously explored model with a newly trained model for the EMG modality, demonstrating EMG data’s useful- ness on the EAR context.

Our main results can be summarized on the following:

- For the EPIC Kitchen dataset, we showed the benefit of using a pre-trained feature extractor and how a simple classifier is enough to applying it to another task.
- For the EPIC Kitchen dataset, we had an equal result with a Multi layer Preceptron and a transformer, as our best models. The MLP also had a better performance across different configurations, making it the overall best classifier.
- For the ActionSense dataset, the EMG modality demonstrated to be more important than the RGB stream by a significant margin. Nevertheless, the best results were obtained by using a multi-modal approach, combining the EMG and RGB streams of data using weighted late-fusion.
