# Owkin_challenge

## Data visualization
As described in the notebook `Supervised_survival_prediction`, I have first had a look at the data and plotted some survival curves with Kaplan-Meier Estimators. I have also assessed the proportions of each histology category since there probably exists a strong correlation on the cancer type and survival time. Moreover, it is important to know that the number of patients that escaped the study is close to the number of patients who died. Indeed building a standard regressor would have been a solution but since half of the patients escaped the study, we might not be able to do so. Thus, I have used a cox model in much of the following. Another interesting notion is that in order to take the histology into account, one needs to encode it. To do so, we can use the LabelEncoder of Scikit-Learn but we have to mind the differences between each encoded label (1 is closer to 2 than to 4). We can also use a One Hot Encoder to get rid of this issue. However, by plotting the survival curves associated with each histology category, we can see that some cancers are closer in terms of survival than others. Thus, I have decided to use it as an additive information to label the categories since it seems relevant for our model.

## Reproduce the baseline

As presented on the [site of the challenge](https://challengedata.ens.fr/participants/challenges/33/), using some radiomics and clinical data yields a good CI score : 0.691. Thus I have tried to reproduce this baseline using the same parameters. This has allowed me to understand a bit better the differences between lifelines' and scikit-survival's implementations of a cox model. Since they have many differences and only allow to perform a Ridge regression at best, I would like to build my own cox regressor. However, I didn't have time and this would an interesting next step. The performance on the training set (80% of the training data) is : **0.704** (CI) against on the validation set (20% of the training data) : **0.668** (CI). Unfortunately, the model overfits a little and I don't reproduce the exact baseline but the result is close and I guess that considering the whole training set instead of the splitted one would yield the result. A first criticism could be that I should have used cross validation to find the best parameters. However, on the one hand lifelines' implementation doesn't allow to play with a Ridge parameter or anything and thus a cross validation is useless, and on the other hand scikit-survival only has alpha to find which is not that hard and tricky so it doesn't really matter whether to use a cross validation or not. In order to find the best ridge parameter in scikit-survival, I built a `golden section search` function instead of the classic `grid search` or `random search`. This approach works when making the assumption that the curve to minimize (resp. maximize) is convex (resp. concave). It is just a bit more fancy and I used it because it saves much computation time and because I don't really give too much interest in this parameter tuning.

Then, I also keep track of the proportions for each origin in order to control how biased are my splits. Additionaly, I considered normalizing the data or not, both can be easily used with the code I developed. In some cases I used it and in some others I don't since it doesn't improve the results and generalization... 

A finer description of my work and comments can be found in the notebook.

## Use the images to improve the predictions : 2D
### Full image

The objective of this challenge is also to make use of the images to improve the survival predictions. Hence, I have decided to first use Transfer Learning and pretrained model to help me with this task. An issue is that to the best of my knowledge, there are no pretrained 3D models... Thus, I decided to use a 2D model such as ResNet50 pretrained on imagenet (not the best choice perhaps since we work on tumors here and not animals but since this model is trained to look at shapes and textures, it makes sense to use it as a first approach to extract features).

Then in order to extract only one 2D image per 3D volume, I built a function using the masks to get the slice containing the highest number of tumor related pixels (given the assumption that a tumor is some kind of a convex volume in 3D).

Doing so, we extract 2048 x 7 x 7 features from our resnet50 and thus we can play a bit with the operation to apply on the 7 x 7 feature maps (mainly a mean operation but anyway...). After applying this selected operation, we get 2048 features which is far too much for our little linear regressor (~ cox model). I apply a PCA to reduce this number, either to a fixed number for lifelines since it doesn't work with too many parameters, or asking it to keep 99% of the variance. 

We then get **0.880** on the train set and **0.676** on the validation set. The model clearly overfits, but let's not spend too much time on this.

### Applied on the masked image

We just multiply by the mask to get the masked image containing only information about the tumor : We then get **0.910** on the train set and **0.688** on the validation set. The model clearly overfits, and the results are a little bit better than earlier but how the model overfits is dramatic !

## Use the images to improve the predictions : 2,5D

The same idea as before is applied in this section. The only modification is that we want to make use of the 3D property of the data. To do so, I consider the *biggest* slice (containing the most information about the tumor) along each dimension. I get 2048 x 3 features and then apply a PCA as before.

We then get **0.884** on the train set and **0.693** on the validation set. This is better than before, but still not that impressive. *This is the model with which I obtained the best score on the public test set.*

## Use the images to improve the predictions : 3D
### Regressor
#### Direct Prediction

Now, we would like to make full use of the 3D property of our data and not only what is called 2,5D. Why do we create our own model ? Well, first of all to the best of my knowledge there isn't any pretrained 3D model available in the litterature, and moreover, a model pretrained on imagenet as before might not be very good to study medical images. For these reasons, we will build our own 3D model using dilated kernels (their efficiency in 3D has been proven in the litterature or at least they are widely used).

Usually Deep Learning is  used for classification. Thus we could train a network to predict the histology of our cohort and then use the network to extract features. But is this approach really a good one ? Indeed, a representation of the images learned by a network trying to predict which tumour it is might not be a good representation to predict survival (cf the survival curves plotted before)...

> A better approach could be instead to remove the final softmax or sigmoid and choose to have only one final neuron. This way our network would be able to perform regression. We then train this network to predict patients' survival : such a network is more likely to get better representations of the tumours for our task. If we do have such a network and that it is trained, then we would have two options : either use the predictions to get a CI score (but in this case the network don't get the subtle issue of patients leaving the study) or use it to extract relevant features for survival prediction that could be passed into a cox model as we did before.

I built a small CNN with two 3D convolutions with dilated kernels and strides of 2 (with a pooling between them) in order to limit the computations and to make it trainable on my laptop (since I had no GPU available). However, in further work I would have worked on Colab or AWS to use the GPUs and train a bigger network. The final part of the model is composed of 3 linear layers with just one neuron at the end and no sigmoid or softmax activation in order to perform regression.

What could be nice is to implement data augmentation for 3D volumes but does it work that easily ? Moreover, it is important that augmented images are still possible images we could come across (stay in the same distribution) ! In our case, we can see the ribs sometimes and the patient seem to be always in the same position with its organs located at certain places. Augmenting the data with rotations and symmetry might not make that much sense... I didn't have time to dive in those considerations but I have proposed a code to use it (which works in 2D but might require small changes to be applied to 3D).

**Training :**
The training settings are descibed in the notebook. I used a learning rate scheduler and SGD Optimizer (Adam might have yield better results though). The process goes well and even if there is no notable improvements in the training curves after a few epochs, I figured out that even if the loss doesn't change, the results in terms of C-index are much better ! This explains why I used 100 epochs.

The results however are really bad with **0.577** on the validation set... Indeed, one might argue that the model never knows which patients escaped the study and which died... Therefore, it might not understand that the Survival Time given isn't until death for each patient, which results in very bad performances.

#### Using only patients who died

Considering only patients for whom `Event==1` is a possible solution but yields very bad results since we consider only half of the patients to train (huge loss of data !). And finally we get **0.547** on the validation set. Extracting features for this doesn't work well neither.

#### Feature Extraction

This approach should be much better theoretically. We applied exactly the same process described above but instead of using a ResNet pretrained on Imagenet, we use the trained 3D CNN. This approach yields the second best result we have : **0.715** on the train set and **0.711** on the validation set ! The model probably overfits on the validation set. *Keep at mind that all these results are obtained using only 80% of the available training data, but to have better scores on the leaderboard, I would probably have retrained the model on the whole dataset.*

### Classifier

Given the metric score (CI), we don't need to predict the exact survival in days in order to get a good score ! Thus, we will use the argsort function of numpy/torch to get labels for classification. This approach might be interesting since the model should learn features representations of data able to distinguish which patients as more risk than another. I defined the exact same CNN as before but this time we consider a number of neurons in output equal to the batch size. Indeed, we want to classify all the patients of a certain batch against each other.

> The major issue about it is the fact that we train the model to predict an order on a certain batch size and thus we cannot apply it directly to the whole dataset. Moreover, when we classify types of tumors 2 images of the same class should be similar. But here, two images leading to the same position in different batches might not be from the same class and so on... 

Thus, we only use it to extract features ! However, an interesting approach to investigate would be to see it as a merge sort problem. Indeed, since we have many batches of patients which are sorted : we consider all the first ones and sort them and go on this way considering all the first ones and the second one of the *best batch*... But this would have issues and thus it requires investigation (and time !).

The performances of this method using feature extraction are impressive : **0.771** on the training set and **0.737** on the validation set. The model thus doesn't seem to overfit that much ! However, when submitting on the test set, I obtained **O.59** which is very bad...

## Conclusion

It has been really great working on this problem and would have been nice to build my own cox model and study the different proposed further steps (for instance merging the sorted batches and find a better way to fix the issue of not knowing how to manage patients who escaped the study in the CNNs training...). But another very important and key remaining step is the error analysis ! Indeed I have had good scores on my validation for multiple models but always much worse ones on the public test set... In order to get better scores, I would have investigated this part, but with the time I had I preferred to focus on the approach which was much more interesting.
