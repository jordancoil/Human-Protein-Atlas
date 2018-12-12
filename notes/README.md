# Human Protein Atlas Notes

## Public Kernels

### Terms

**LB (Leader Board)**: The score achieved on the public leaderboard

### https://www.kaggle.com/iafoss/pretrained-resnet34-with-rgby-0-460-public-lb
#### Score: 0.460 LB
- Uses ResNet34 as a base model, but mentions that since the classified images are much different than ImageNet substantial retraining of the entire model is needed
- Since this is a multilabel and multiclass problem we need to consider a few things
  - SOFTMAX cannot be used as an output layer, because it encourages single label prediction
  - Typically for multilabel prediction Sigmoid is used, **(Not sure what the following means exactly)** _"However, combining the sigmoid with the loss function (like in BCE with logits loss or in Focal loss used in this kernel) allows log(sigmoid) optimization of the numerical stability of the loss function. Therefore, sigmoid is also removed."_
- This competition has a strong data imbalance. eg. "Nucleoplasm" is a common class, but classes like "Rods & Rings" are very rare.
  - in multilabel multiclass problems, there is always an issue with imbalance. ie. predicting 1 out of 10 classes given the same number of examples of each class classes, you have 1 positive and 9 negative examples. We need to use a loss function that accounts for this
  - Focal loss is chosen here and in many other kernels. it comes from this paper: https://arxiv.org/pdf/1708.02002.pdf
  - Focal loss had good results on the airbus ship detection competition (for image segmentation): https://www.kaggle.com/iafoss/unet34-dice-0-87
- This competition has 4 channels of input (RGBY), but ImageNet pretrained models only accept 3 channels (RGB) as input. Also, the dataset is too small to train something like ResNet34 from scratch.
  - This kernel solves this by changes the first layer from 7x7 3->64 to 7x7 4->64 while keeping weights from 3->64 and setting weights for the new Y channel to 0
  - There is a comment on this kernel saying that rather than initializing the Y channel with zeros, you can simply copy the weights from one of the other channels [0-2] because ResNet34 is already trained on simple shapes, so this should give you a better result Which apparently you can do with something like this: `layers[0].weight = torch.nn.Parameter(torch.cat((w,w[:,:1,:,:]),dim=1))`
- in order to find the optimal learning rate this kernel runs a function that plots the loss over different learning rates. The optimal learning rate lies that the minimum of the plotted curve, but before divergence
- During training, the pre-trained weights are kept frozen for a while to avoid corruption due to the random initialization of the head layers
  - **not sure how long they are frozen for though...I think for 1 epoch?**
- (After 1 epoch?) unfreeeze all the weights
  - a trick used in this kernel is to lower the learning rate by a factor of 10 for the first layers (**for how long? i don't know**). This is done because the images in the competition are significantly different from ImageNet
  - _"Another trick is learning rate annealing. Periodic lr increase followed by slow decrease drives the system out of steep minima (when lr is high) towards broader ones (which are explored when lr decreases) that enhances the ability of the model to generalize and reduces overfitting. The length of the cycles gradually increases during training."_
- There is some more stuff about "thresholding" at the bottom of this kernel which i don't really understand, but apparently it can increase the LB score of this kernel to `0.466`
