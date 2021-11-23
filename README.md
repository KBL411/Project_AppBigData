# Project Application of Big Data

## Dependencies management

	pip install poetry
	poetry install
	
Poetry will install all the package store in the .toml
### Installation Problem
If you have an problem of installation check that you are in the right directory. If yes do Windows+R and go to 

	C:\Users\"User_Name"\AppData\Local\pypoetry

Delete the Cache folder and re-run the installation

## Poetry function
### pyproject.toml
It contains the list of packages that we need in the virtual environment. 
In fact, it does the same thing as the "requirements.txt" file.

To add a package we will just make a shell command in the root of our project :
	`poetry add [package_name]`
	
And if you need to remove one :
	`poetry remove [package_name]`
### poetry.lock
If you don't want the latest version of your package or just an specific version you have to configure it here.

## 



## Naming Convention
### Function
Use a lowercase word or words. Separate words by underscores to improve readability.
`function`,  `my_function`

###  Variable
Use a lowercase single letter, word, or words. Separate words with underscores to improve readability.
`x`,  `var`,  `my_variable`

### Class
Start each word with a capital letter. Do not separate words with underscores. This style is called camel case.
`Model`,  `MyClass`

### Method
Use a lowercase word or words. Separate words with underscores to improve readability.
`class_method`,  `method`

### Constant
Use an uppercase single letter, word, or words. Separate words with underscores to improve readability.
`CONSTANT`,  `MY_CONSTANT`,  `MY_LONG_CONSTANT`

### Module

Use a short, lowercase word or words. Separate words with underscores to improve readability.
`module.py`,  `my_module.py`

### Package

Use a short, lowercase word or words. Do not separate words with underscores.
`package`,  `mypackage`

## MlFlow usage
### Train tracking
To launch mlflow in poetry environment, launch this command in the project directory:
	
	poetry run mlflow ui

Then simply click on the link displayed in the console (usually http://127.0.0.1:5000/)

All launch are displayed in the ui. To see which model has been used, look at the `source` column

To launch another train, launch in the project directory the following command:

	poetry run python train[GradientBoosting/RandomForest/XGBoost].py random_state n_estimators

Choose the appropriate file depending on the model you are interested in.

`random_state` and `n_estimators` are parameters used by the programs and are replaces by respectively 42 and 100 if not specified.

Note : XGBoost algorithm do not use `n_estimators`.

### Model serving API
In order make a trained model available to use via a REST API, get the hash of the model training and run following command in project directory:

	poetry run mlflow models serve –m mlruns\0\[model hash]\artifacts\model -p [port to use]

A hash is similar to this `1f8180d11f6146389b4b0a4a08401c1e` and you can use port `1234` for instance.

## ML Interpretability (SHAP)
Visualize explanations for a specific point of a data set.

We plot the SHAP values for the 10th observation:
<img src="C:\Users\Rayan\source\Project_AppBigData\for_README\force_plot.png"/>

A dependence scatter plot to show the effect of a single feature across the whole dataset:
<img src="C:\Users\Rayan\source\Project_AppBigData\for_README\dependence_plot.png"/>

Visualize a summary plot for each class on the whole dataset.
<img src="C:\Users\Rayan\source\Project_AppBigData\for_README\summary_plot .png"/>


