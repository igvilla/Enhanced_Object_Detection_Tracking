# Enhanced_Object_Detection_Tracking
The objective of this research is to conduct a comparative analysis of the performance of baseline YOLO
(You Only Look Once) and the more complex DETR (DEtection TRansformer) models in the context of automotive object
detection. A key aspect of this study is to evaluate whether the integration of optical flow as a heuristic can enhance the models’
performance, both in terms of accuracy and computational efficiency.
Our research focuses on two primary contributions: firstly, to establish a comprehensive understanding of how YOLO and
DETR models perform under various automotive conditions, and secondly, to assess the impact of incorporating optical flow
on these models. This approach aims to leverage the motion detection capabilities of optical flow to potentially improve the
real-time detection and tracking accuracy in dynamic driving environments.
To date, our experiments have involved rigorous testing of these models on diverse datasets representing a range of
driving scenarios. Preliminary results indicate a notable difference in the performance of YOLO and DETR, with each model
exhibiting unique strengths and limitations. While we expected the addition of optical flow to hurt accuracy but improve
efficiency, it actually appears to have improved accuracy with no change to model efficiency, including in complex traffic
situations with high object movement. This study is pivotal in identifying the most effective model and heuristic combination,
promising to enhance the reliability and safety of autonomous driving systems.

### NOTE: 
This information (along with usefu images and tables) can be found in the attached written report for the project, 
"Enhanced_Object_Detection_Tracking.pdf". Please refer to this document for a detailed description and explanation of the project.


## Data

We worked with the “Driving Video with Object Tracking” dataset, which is a subset of the
Berkeley DeepDrive dataset. This dataset came with 1000 videos and label csv file, which includes one row for every
bounding box in every frame (sampled at 5hz from 30hz) for each video. From this, we randomly sampled 100 videos,
80 for training and 20 for testing. Each set had an even distribution of object labels. The labels included: car, pedestrian,
truck, bus, bicycle, rider, other vehicle, motorcycle, other person, trailer, and train. However, due to ambiguity in some
labels, we consolidated our labels into: car, person, truck, bus, bicycle, motorcycle, and train. Trailer and other vehicle
were dropped due to poor labeling and infrequency, such as a grocery shopping cart being labeled ”other vehicle” in the dataset.


## Data Cleaning

In the preparation of our object detection dataset, bounding box coordinates provided in
(x1, y1, x2, y2) format were converted to COCO’s (x, y, w, h) format to facilitate compatibility with our model’s
input requirements. We faced challenges with frame extraction due to discrepancies in frame rates between the provided
videos and labels. By sampling every sixth frame to match the 5Hz annotation rate from a 30Hz video, we synchronized
our data with the provided bounding boxes.
Our dataset was split using Multilabel Stratified Shuffle Split, ensuring a balanced class distribution, and we
employed the Iterative Stratification library, accessible at this GitHub repository. Augmentations were applied to enrich
the training data, including hue and brightness variations, grayscale conversion, and flipping. Test data were kept simple with 
resizing and tensor conversion. A normalization step was also introduced to centralize bounding box coordinates
and scale them according to image dimensions, enhancing the quality and consistency of the input data for our models.

Some of the initial data preprocessing steps can be seen in the "preprocessing.ipynb" notebook, while additional data manipulation steps
are performed in each of the individual notebooks as well for training of the models. 

## Performance Metrics

To optimize the matching of predicted and ground truth objects, we adopt a bipartite matching loss.
This loss function allows for an optimal one-to-one assignment between predictions and ground truths, minimizing
misclassifications and improving localization accuracy. The dataset comprises 100 videos, partitioned into an 80-20
split for training and testing, respectively, with frames meticulously extracted for processing.
Our choice of performance metric is the mean Average Precision (mAP), which is pivotal in object detection tasks
due to its robust evaluation across varying threshold levels for classification and localization accuracy. The mAP offers
a comprehensive measure, considering both precision and recall, making it an excellent gauge of a model’s effectiveness
in distinguishing and accurately placing bounding boxes around different object classes within the intricate environment
of driving scenarios. mAP also calculates precision and recall across different thresholds and across all labels, which
can then be averaged to represent the overall model performance for all objects of interest. Through this investigation,
we seek to establish the comparative strengths of YOLO and DETR models, as well as the added value of bipartite
matching loss, in pushing the boundaries of precision in object detection for autonomous vehicles.


## Models

### YOLO

The most up-to-date YOLO model (YOLO v8) was used for object detection in this project. This model
was initially tested on the data with its pre-trained model weights. Since this model was trained to detect and predict 80
different class labels, while the goal of this project was limited to predicting only 7 of the 80 class labels, the initial
testing and validation was expected to perform slightly worse with irrelevant label predictions. The next step involved
retraining the model weights for YOLO v8 using our custom training set. Each frame of the training set of videos was
passed through the YOLO model in batches of 64 to undergo training and calculate optimal weights for minimization of
the 3 losses calculated during training: box loss, classification loss (CLS), and distribution focal loss (DFL). Box loss
focuses on the error in bounding box coordinate location (with respect to the ground truth bounding box locations),
CLS focuses on the error in classifying detected objects correctly, and DFS focuses on the precision of the precision of
the bounding box predictions in order to optimize a distribution of bounding box boundaries.

As mentioned, YOLO was initially run on the test set of videos before finetuning the model
weights to our specific use case of video data. This initial performance, as expected, was slightly worse than what was
obtained after training. This initial mean average precision (mAP) improved from 0.122 to 0.17. All steps discussed for the training 
and validation of YOLO on the dataset can be found in the "yolo_v8.ipynb" notebook. 

### DETR

After pre-processing the data into the appropriate format for DETR, we loaded the pre-trained model
from Facebook via the pytorch hub. As described above, we used bipartite matching loss for training and evaluation
of DETR, as is done in literature. Before training on the model for fine tuning, we input our test data into the model
to obtain a baseline loss for comparison. We then trained on DETR, using a maximum number of queries of 65 and
a batch size for the dataset of 80. This was chosen due to a maximum of 45 bounding boxes within the training data,
plus room for some more boxes to be detected for test. Our training fine tunes the pre-trained DETR model to work
specifically for our traffic video dataset, and we aimed to observed a reduction in loss. After training the DETR model
to minimize the loss function described above, we use the fine tuned model for test-set object detection, as well as to be
incorporated into the optical flow heuristic.

The DETR model without finetuning performed very poorly on the dataset and was unable to
identify bounding boxes for object detection. After finetuning, we saw a notable improvement in performance, with the
loss decreasing from 3.45 to 0.88. All steps discussed for the training 
and validation of YOLO on the dataset can be found in the "detr-3.ipynb" notebook. 

### Optical Flow Integration

Optical flow is computed using the opencv library in Python. We computed the
trajectory of a bounding box from one image to the next using the four corner points of the box. This integrates the
sequential nature of video data into predictions. Initial testing of optical flow on the dataset can be encountered in the 
"optical_flow.ipynb" notebook. 

A heuristic was designed for full video data to integrate with the chosen models. This differs from standalone DETR
or YOLO, which predict on a frame of a video. The general goal of this is to use the chosen model as little as possible.
We want to reduce the overall computational cost by limiting the need to execute the chosen model (YOLO or DETR),
so optical flow can be used to track bounding box movement instead. After a certain number of frames of just optical
flow, we then check the chosen model to see if the optical flow box predictions still align. As part of our heuristic, we
define a padding variable, which begins at the value of 1. For the first frame of the video, we use the chosen model to
predict bounding boxes. We then predict for subsequent frames using just optical flow, but check back with the chosen
model after padding number of frames. When padding is 1, this implies we use the chosen model at every frame.
However, an additional aspect of the heuristic is our approach to this variable. Every time we check optical flow against
the chosen model, we compare the bounding boxes with box IOU (intersection over union). If the box IOU is above a
threshold, we imply that the optical flow bounding boxes are sufficiently close to the model. At this step, we increase
the padding variable by 1 (i.e. padding could be increased from 1 to 2 to 3 and so on, increased the number of steps
before we check the chosen model. However, if the box IOU is below the threshold, we reset the padding back to 1.

After training the models, integrating optical flow into the overall architecture resulted
in improved performance overall. Although it improved for both models, the improvement was more drastic for YOLO,
improving the mAP from 0.17 to 0.2896. DETR improved by far less (0.39 to 0.407).
We also chose to explore model time complexity by exploring each of the models for one video in the data set. This
was done to mimic a real-time model, which is just one ”video” with object detection. We used the same video, which
had 101 frames (sub-sampled from a 20 second video), across each models, finding the MAP and timing for each model.
As expected, YOLO performed better in terms of speed, both with and without optical flow integration. The integration of DETR with 
optical flow can be found in the "model_optical_flow_combined.ipynb" notebook, while integration of optical flow with YOLO is found
in the same notebook used for validation and testing of YOLO alone, "yolo_v8.ipynb".

Finally, we can see a clear understanding of the success of our DETR and YOLO models on object detection as they
occur in sampled images. Both DETR and YOLO predictions actually match up well with the
true bounding boxes. This occurs for images at night, with objects at a distance, and many objects to detect.
Overall, there are some notable conclusions from our results. First, we see an imrpovement for both YOLO and
DETR from pre-training to finetuning. This means that our trained models are better suited for the traffic dataset we are
working with. The images show this as well, as much of object detection relies on the qualitiative aspect of actually
seeing the object detection succeed in the images. This gives us an understanding of how these models may accurately
detect images in a real environment. While these are notable results on their own, the incorporation of optical flow
provides an enhances analysis. The optical flow model does not improve efficiency as expected. However, it has either
very similar or improved results on MAP compared to respective standalone models. The time similarity is likely due
to a weakness in the heuristic, where the chosen model is being used at most iterations. We can continue to explore the optical flow heuristic and change the
thresholds as well as the mechanism for determining padding values, to hopefully achieve the results we expected. As
an additional note, we also saw a remarkable improvement in the MAP value for YOLO with optical flow. We compute
MAP for DETR and any optical flow models with a custom built MAP function. However, YOLO has MAP built in.
This could explain the difference in metrics as well. These factors are explored in the discusion section.
Overall, our results, both qualitative and quantitive, show successful object detection for each model, with DETR
consistently performing better but slower, as was expected. The optical flow integration works but does not impact
results significantly.

## Discussion and Further Improvements 

Our model had a few key limitations, including training time and model comparability. Because the YOLO and
DETR models were pre-trained, and we used packages for each to conduct fine-tuning, we used each models commonly
used loss function. YOLO had MAP built into it, but DETR did not. Additionally, their loss functions differ slightly. In
the future, we aim to develop more thorough metrics for an enhanced comparability of the data.
Additionally, we hope to train on a larger set of data. In this case, given the limited time and complexity of the
problem at hand as well as limited resources, the data set used was only of 100 driving car videos, 80 for training and
20 for testing. If presented with more time for more training (e.g. more epochs of training on a larger training set of videos),
the mAP scores for both models could be expected to
improve significantly, as more training data could be expected to reduce the bias of the model’s performance.
Beyond improving metrics and training, we also would hope to expand upon the optical flow heuristic. Currently,
optical flow does not improve time, which we believe is due to frequent use of the chosen model. To get the true benefits
of an optical flow addition to the model, we need to spend significant time designing the heuristic so that optical flow
can be used more and the chosen model will not be computed as much. We hope to maintain the strong accuracy the
optical flow model provides, while seeing the improvmeent in efficiency we expected.
Finally, we can extend this problem in the future to further mimic a real-time environment. With a better dataset and
further exploration of the sequential video data, we can see how these models truly perform in a more realistic setting.
This differs from our primarily frame-based approach to object detection. Integrating these models into real-time data
will give us the best sense of its impact on object detection and the improvement of it for autonomous driving.


