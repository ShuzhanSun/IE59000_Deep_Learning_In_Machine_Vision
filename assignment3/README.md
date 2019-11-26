# Assignment 3

## You are required to do this assignment on google cloud. Please use Python 3.6 or above. Your personal computers can be either in Windows 10 or Ubuntu or OSX. Questions 3, 4 and 5 require GPU resources for faster processing. Please set up Google Cloud VM mahines with either K80 or P100 GPUs. 

## Download the dataset
1. Download assignmen3.zip from Blackboard.
2. Upload the assignment3.zip to the google cloud. Use the 'upload' button on jupyter notebook interface.
3. Unzip assignment3.zip. Copy paste the following command in the google cloud terminal. Make sure to modify ```path_to_gcloud``` appropriately in the following command.
```
unzip path_to_gcloud/assignment3.zip
```
4. Navigate to the 'assignment3' folder
```
cd path_to_gcloud/assignment3/
```
5. Download the dataset. In this homework, you will use COCO captioning data, pretrained SqueezeNet model (only for Tensorflow) and some ImageNet validation images. 
```
	cd ie590/datasets
	./get_assignment3_data.sh
```

## Miscellaneous
1. Make sure to start the IPython notebook server from the assignment3 directory, with the ``` jupyter notebook``` command. 
2. Note that questions 3, 4, and 5 can be either done in Pytorch or TensorFlow. You will not be awarded extra points if you do a question both in both the frameworks. 

## Questions
### 1. Image Captioning with Vanilla RNNs (25 points)

The Jupyter notebook `RNN_Captioning.ipynb` will walk you through the implementation of an image captioning system on MS-COCO using vanilla recurrent networks.

### 2. Image Captioning with LSTMs (30 points)

The Jupyter notebook `LSTM_Captioning.ipynb` will walk you through the implementation of Long-Short Term Memory (LSTM) Networks, and apply them to image captioning on MS-COCO.

### 3. Network Visualization: Saliency maps, Class Visualization, and Fooling Images (15 points)

The Jupyter notebooks `NetworkVisualization-TensorFlow.ipynb` / `NetworkVisualization-PyTorch.ipynb` will introduce the pretrained SqueezeNet model, compute gradients with respect to images, and use them to produce saliency maps and fooling images. Please complete only one of the notebooks (TensorFlow or PyTorch). No extra credit will be awardeded if you complete both notebooks.

### 4. Style Transfer (15 points)

In the Jupyter notebooks `StyleTransfer-TensorFlow.ipynb` / `StyleTransfer-PyTorch.ipynb` you will learn how to create images with the content of one image but the style of another. Please complete only one of the notebooks (TensorFlow or PyTorch). No extra credit will be awardeded if you complete both notebooks.

### 5. Generative Adversarial Networks (15 points)

In the Jupyter notebooks `GANS-TensorFlow.ipynb` / `GANS-PyTorch.ipynb` you will learn how to generate images that match a training dataset, and use these models to improve classifier performance when training on a large amount of unlabeled data and a small amount of labeled data. Please complete only one of the notebooks (TensorFlow or PyTorch). No extra credit will be awarded if you complete both notebooks. 

## Working on google cloud
* Follow the [instructions](https://github.com/cs231n/gcloud/) here to setup the google cloud.
* Install requirements.txt on the command line
```
pip3 install --user -r requirements.txt
```

## Submission instructions
* Create the PDF reports of each of the ipython notebooks on jupyter. 
* Save the PDF reports in `assignment3` folder. 
* Run the following on the terminal to create a zip file of your solutions. 
```
zip -r assignment3_firstname_lastname.zip . -x "*.git*" "*ie590/datasets*" ".env/*"
```
* Submit the `assignment3_firstname_lastname.zip` on the Blackboard. 