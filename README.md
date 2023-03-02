# CV-Error-Calibration



## Setup

1. Install Python 3.7-3.9 because Pytorch currently only supports these versions.
2. Create and activate a virtual environment using `virtualenv`. [PyPi](https://pypi.org/project/virtualenv/).
3. Install the requirements using `pip install -r requirements.txt`.

## Assumptions
### Folder Structure

The script implicitly assumes that the CV Project being tested has the following structure.

```
project
│
└───test
│   │  
│   └───class1
│   │    │...
│   └───class2
│   │    │...  
│      
│   
└───train
│   │  
│   └───class1
│   │    │...
│   └───class2
│   │    │...      
│      
│       
└───models
    │
    └───tiny
    │   └───model
    │   │    │model_tiny.pt
    │   └───transform
    │   │    │transform_tiny.pickle
    │
    └───small
    │   └───model
    │   │    │model_small.pt
    │   └───transform
    │   │    │transform_tiny.pickle 
```

Therefore, the following must occur
- The test data and the model components are kept in a subfolders that are in the same parent directory. 
- Each of the models are in a seperate subfolder with the appropriate name
- The model and the transform function are both in specific subdfolders. 

### Model

The script can currently handle Pytorch models and Torchvision's ConvNeXt models. This script does NOT train the model and pre-trained models must be saved with a '.pt' extention (saved as `torch.save(model, PATH)`).

### Transformation Function

The script leverages `datasets.ImageFolder` to create a datalaoder, `transform_tiny.pickle` in the example directory is a function (`data_transform` below)that was used in the training process when using  `datasets.ImageFolder` as below:
```
dataset = datasets.ImageFolder(data_path, transform = data_transform())
```

## Usage 

After setup has completed and a project that satisfies the above assumptions,  run:  

```
python test.py <data>
```  

or  

```
python test.py <data> --model <model>
```  

replacing `<data>` with the path to the dataset and `<model>` with the model size (tiny, small, base, large). In the subfolder above `<data>` would be `~/project/test`. 


The does the following:
- Compute expected calibration error (ECE) max calibration error (MCE) for the model over the specified data (with bins (k) set to 10).
- Produces calibration error plot (stored in `~/project/results`).
- Prints the confusion matrix and produces a heat map plot of the confusion matrix (stored in `~/project/results`)
- Identifies the false positives and save all false postive images in a seperate folder (`~/project/results/false_positves`) with the same folder structure as specified above
- Takes the class probabilities of the false positves and uses dimensionality reduction techniques PCA and t-SNE to reduce the dimensions (down to 2) and visualise patterns in false postives.
- Takes the RGB values of the false positves and flattens it so PCA and t-SNE can reduce the dimensions (down to 2) and visualise patterns in false postives.

## Limitations
The following are limitations of the code:

- A large portion of the code is producing inference on the images. This has the potential to be completed in parralel which would increase computational time significantly. 
- The majority of the computation is completed by Pytorch, which can be computed on a GPU (if one exists). This code does not spcify the device which calculations are completed on. 