# MobileNetv2_Netvlad
Netvlad using lightweight framework mobilenet_v2 for feature extraction！The repository includes :</br>
1. Inference demo
2. Pre-trained model
3. Sample image
4. Totorial and description </br>
The repository is created by **Ze Huang**
## **Usage**
First, make sure you have installed **python 3.8** environment and downloaded this project repository.<br/>
Next, run the following command in the home directory to install the python package needed to use this model
```
pip install -r requirements.txt
```
Run `python demo.py` to get the feature and vlad vector of the sample image without CUDA.
You can also specify the image path or using CUDA by runing the following command.
```
python demo.py --image='imagepath' --cuda
```
## **Details**
| item | value |
|--|--|
|  **model**|mobilenet_v2 + netvlad  |
|  **size**|18.50M  |
|  **number of clusters**|64  |
|  **training data**|Pittsburgh  |
|  **accuracy**|**R@1**=80.07% **R@5**=91.25% **R@10**=93.99%  |
|  **time of load network**|**cpu**=0.111s **gpu**=0.148s  |
|  **time of inference**(single image)|**cpu**=0.136s **gpu**=0.020s  |
|  **GPU**|TITAN XP  |
