## Datasets

### Data preprocessing

(1) **[AD dataset](https://osf.io/jbysn/)** contains EEG recordings from 12 patients with Alzheimer's disease and 11 healthy controls. Each patient has an average of 30.0 $\pm$ 12.5 trials. Each trial corresponds to a 5-second interval with 1280 timestamps (sampled at 256Hz) and includes 16 channels. 

(3) **[TDBrain dataset](https://brainclinics.com/resources/)** contains EEG recordings of 1274 patients with 33 channels (500 Hz) during EC (Eye closed) and EO (Eye open) tasks. The dataset consists of 60 types of diseases, and it is possible for a patient to have multiple diseases simultaneously. This paper focuses on a subset of the dataset, specifically 25 patients with Parkinson's disease and 25 healthy controls. 

(3) [NerveDamage](https://physionet.org/content/emgdb/1.0.0/)** consists of 204 single-channel EMG records from the tibialis anterior muscle of three volunteers that are healthy, suffering from neuropathy, and suffering from myopathy, respectively. The sampling rate (frequency) is 4K Hz, and each record encompasses 1500 sampling points. We segment one record into 6 samples without overlapping, where each sample has 250 observations. Each patient is a classification category, and the classification objective is to determine which volunteer each sample belongs to.

### Data source

The processed datasets can be manually downloaded at the following links.

* AD dataset: https://figshare.com/ndownloader/files/43196127
* NerveDamage: https://www.timeseriesclassification.com/description.php?Dataset=NerveDamage

Since TDBrain is not a public dataset, we do not provide a download link here. The users need to request permission to download on the TDBrain official website and process the raw data.

## Requirements

The recommended requirements are specified as follows:

* Python 3.10
* Jupyter Notebook
* scikit-learn==1.2.1
* torch==1.13.1+cu116
* matplotlib==3.7.1
* numpy==1.23.5
* scipy==1.10.1
* pandas==1.5.3
* wfdb==4.1.0
* neurokit2==0.2.4

The dependencies can be installed by:

```bash
pip install -r requirements.txt
```

## Running the code

We provide jupyter notebook examples for each dataset. To train and evaluate COMET on a dataset, simply run `DatasetName_Method_example.ipynb`, such as `AD_Example_COMET.ipynb`.  After integrating with LTCR,  you can see the `AD_Example_COMET_LTCR.ipynb`.

After training and evaluation, the pre-training model and fine-tuning model can be found in `test_run/models/DatasetName/`; and the logging file for validation and test results can be found in  `test_run/logs/DatasetName/`. You could modify all the parameters, the working and logging directory in `config_files/DatasetName_Configs.py`.
