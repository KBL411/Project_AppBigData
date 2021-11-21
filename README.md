# Project Application of Big Data

## Dependencies management

	pip install poetry
	poetry install
	
Poetry will install all the package store in the .toml

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

## ML Interpretability (SHAP)
<br Visualize explanations for a specific point of a data set />
We plot the SHAP values for the 10th observation
<img src="https://github.com/KBL411/Project_AppBigData/blob/main/force_plot.png" />
<br A dependence scatter plot to show the effect of a single feature across the whole dataset/>
<img src="https://github.com/KBL411/Project_AppBigData/blob/main/dependence_plot.png" />
<br Visualize explanations for all points of your data set at once />



