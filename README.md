# Learning to Gaze by Watching Videos
## A Deep Reinforcement Learning Approach on Gaze Control through Self-generated Reward from Learned Features

Undergraduate Special Project for B.S. Computer Engineering at Electrical and Electronics Institute, University of the Philippines Diliman

Abstract:

Deep learning has achieved state of the art results in many problems in computer vision through deep convolutional neural networks. However, to achieve these results, extensive labeling of large amounts of data is necessary in order to train a deep convolutional neural
networks. For humans, we do not need labeling on our environment in order to learn vision rather we learn from the features we see in the environment. Instead of extensive labeling on data, a network can trained using self-supervised methods to learn how to extract necessary features in an image. Using the learned features from a specic object in a large scale dataset, i.e. faces and human bodies, the network can then learn to locate these features in other images. This method can be applied to the localization and tracking problem in computer vision. For localization, the features are searched on a wide eld of view image while on tracking, deep reinforcement learning was used such that reward generation is self-generated based on the learned features of the network. Even without labels on where the objects are in the video, the network was able to localize or track them even as they were moving around the wide field of view video.

Sample results are at https://1drv.ms/f/s!Ao8Y5FscWK9in7cEdKnqKRJvniP_Sg
