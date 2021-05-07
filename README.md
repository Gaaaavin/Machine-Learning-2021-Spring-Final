# Machine-Learning-2021-Spring-Final
Xinhao Liu &amp; Wenbin Qi's Final Project for Machine Learning 2021 Spring

# Machine-Learning-2021-Spring-Final
Xinhao Liu &amp; Wenbin Qi's Final Project for Machine Learning 2021 Spring
## File Structure
### Dataset
<u>**Remember to unzip (use tar command) these datasets to the file where you are running your code**</u>
This File includes thre datasets:
#####100face.tar.gz: 
This is the original dataset that contains 2022 faces with 100 identities

#####100face_random_mask.tar.gz:
This is the dataset generated from 100 face. We generate a virtual mask on approximately 50% of the pictures

#####30unmask+70masked.tar.gz:
This dataset contains 30 identities from 100face.tar.gz and 70 identities from the downloaded dataset with all pictures have face mask

###Methods

This file contains two possible methos that we can use (not sure which is better). One is simply use a VGG net to map the pictures to features, the other is use a ResNet and then apply the CBAM method (we use ResNet becuase it's the only code I can find). For both methods, we use center loss as the loss function and to predict which class a picture belongs to. 

**To run either method, you only need to run the train.py or cbam_train.py file. The global parameters are defined in the beginning of the file and is quite straightforward**

###Archive
This file contains some of the outcomes of the previous training. The two pth file is for loading models. The two png files are the initial result of the two methods on the 100face_random_mask dataset.
