# Cifar 1k project 

## Data classification from 1000 examples on the CIFAR-10 dataset

This project investigates how good a deep learning model can get on the classic CIFAR-10 image classification task using only 1000 examples. To get good results, I used transfer learning: ResNet models were pretrained on the ImageNet classification data and used as a starting point for this project  

_main.py_ runs the full end-to-end training in the following steps: 

* Proxy training: First proxy model is trained on the full CIFAR-10 dataset 
* Subset selection: Then it selects the 1k subset (coreset) of the data based on some criteria
* Core training: finally a new model is trained on the 1k core dataset 


## Usage

1. clone the repo:  
from your terminal type <code> git clone https://github.com/Bencee16/cifar_1k </code>

2. create virtualenv from yml file:  
<code> conda env create -f environment.yml </code>

3. to run the training use type <code> python main.py </code> from the directory where main.py is located with additional arguments:
* --batch_size (default=64)
* --num_epochs_proxy (default=20)
* --num_epochs_core (default=150)
* --model_type_proxy (default="resnet18")
* --model_type_core (default="resnet50")
* --use_pretrained_proxy (default=True)
* --use_pretrained_core (default=True)
* --freeze_weights_proxy (default=False)
* --freeze_weights_core (default=True)
* --coreset_size (default=1000)
* --coreset_selection_method (default="reverse_forgetting_events")
* --saving (default=True)
* --continue_from_selection (default=False)
* --dropout (default=False)

For coreset selection you can choose from methods _forgetting_events_ _(An Empirical Study of Example Forgetting during Deep Neural Network Learning, Toneva et al., 2019)_ , _random_ or _reverse_forgetting_events_. 

To skip the proxy model training, use the argument _--continue_from_selection True_, for selection the algorithm will use the forgetting statistics calculated on a pretrained proxy model (ResNet18 architecture, 92% test accuracy)  

During first run _main.py_ downloads the CIFAR-10 dataset into a _data_ folder

### Results
For models on whole dataset:  

| model         | Test accuracy |
| ------------- | ------------- |
| ResNet18      | 92.1%         |
| ResNet50      | 93.0%         |

For models on 1k samples (ResNet50 architecture):  

| model                                | test_accuraxy |
| -------------------------------------| ------------- |
| forgetting_events selection          | 34.8%         |
| random selection                     | 60.5%         |
| reverse fe selection                 | 66.0%         |
| reverse fe with freeezed conv layers | 68.3%         |



