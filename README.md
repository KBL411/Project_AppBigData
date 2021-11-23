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

	poetry run mlflow models serve –m mlruns\0\[model hash]\artifacts\model -p [port to use] --no-conda

A hash is similar to this `1f8180d11f6146389b4b0a4a08401c1e` and you can use port `1234` for instance.

We can query the model at the address displayed in terminal.
For instance, we execute this command on one line of our test file:

	curl -X POST -H "Content-Type:application/json; format=pandas-split" --data "{\"columns\": [\"CNT_CHILDREN\", \"AMT_INCOME_TOTAL\", \"AMT_CREDIT\", \"AMT_ANNUITY\", \"AMT_GOODS_PRICE\", \"REGION_POPULATION_RELATIVE\", \"DAYS_BIRTH\", \"DAYS_EMPLOYED\", \"DAYS_REGISTRATION\", \"DAYS_ID_PUBLISH\", \"FLAG_EMP_PHONE\", \"FLAG_WORK_PHONE\", \"FLAG_CONT_MOBILE\", \"FLAG_PHONE\", \"FLAG_EMAIL\", \"CNT_FAM_MEMBERS\", \"REGION_RATING_CLIENT\", \"REGION_RATING_CLIENT_W_CITY\", \"HOUR_APPR_PROCESS_START\", \"REG_REGION_NOT_LIVE_REGION\", \"REG_REGION_NOT_WORK_REGION\", \"LIVE_REGION_NOT_WORK_REGION\", \"REG_CITY_NOT_LIVE_CITY\", \"REG_CITY_NOT_WORK_CITY\", \"LIVE_CITY_NOT_WORK_CITY\", \"EXT_SOURCE_2\", \"OBS_30_CNT_SOCIAL_CIRCLE\", \"DEF_30_CNT_SOCIAL_CIRCLE\", \"OBS_60_CNT_SOCIAL_CIRCLE\", \"DEF_60_CNT_SOCIAL_CIRCLE\", \"DAYS_LAST_PHONE_CHANGE\", \"FLAG_DOCUMENT_3\", \"FLAG_DOCUMENT_4\", \"FLAG_DOCUMENT_5\", \"FLAG_DOCUMENT_6\", \"FLAG_DOCUMENT_7\", \"FLAG_DOCUMENT_8\", \"FLAG_DOCUMENT_9\", \"FLAG_DOCUMENT_10\", \"FLAG_DOCUMENT_11\", \"FLAG_DOCUMENT_12\", \"FLAG_DOCUMENT_13\", \"FLAG_DOCUMENT_14\", \"FLAG_DOCUMENT_15\", \"FLAG_DOCUMENT_16\", \"FLAG_DOCUMENT_17\", \"FLAG_DOCUMENT_18\", \"FLAG_DOCUMENT_19\", \"FLAG_DOCUMENT_20\", \"FLAG_DOCUMENT_21\", \"AMT_REQ_CREDIT_BUREAU_HOUR\", \"AMT_REQ_CREDIT_BUREAU_DAY\", \"AMT_REQ_CREDIT_BUREAU_WEEK\", \"AMT_REQ_CREDIT_BUREAU_MON\", \"AMT_REQ_CREDIT_BUREAU_QRT\", \"AMT_REQ_CREDIT_BUREAU_YEAR\", \"NAME_CONTRACT_TYPE\", \"CODE_GENDER\", \"FLAG_OWN_CAR\", \"FLAG_OWN_REALTY\", \"NAME_TYPE_SUITE\", \"NAME_INCOME_TYPE\", \"NAME_EDUCATION_TYPE\", \"NAME_FAMILY_STATUS\", \"NAME_HOUSING_TYPE\", \"WEEKDAY_APPR_PROCESS_START\", \"ORGANIZATION_TYPE\"],\"data\":[[0, 0.024640657084188913, 0.2476595744680851, 0.10245343025897319, 0.19148936170212766, 0.2573801121029697, 0.33189268073641687, 0.03954471578705324, 0.782058848326448, 0.8720856962822936, 1, 0, 1, 0, 1, 0.05, 0.5, 0.75, 0.7826086956521738, 0, 0, 0, 0, 0, 0, 0.9235719769752415, 0, 0, 0, 0, 0.6010089429030039, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0.25, 0.25, 0.2, 0.8333333333333333, 0.49122807017543857]]}" http://127.0.0.1:1234/invocations

## ML Interpretability (SHAP)
Visualize explanations for a specific point of a data set.

We plot the SHAP values for the 10th observation:
<img src="https://github.com/KBL411/Project_AppBigData/for_README/force_plot.png"/>

A dependence scatter plot to show the effect of a single feature across the whole dataset:
<img src="https://github.com/KBL411/Project_AppBigData/blob/main/for_README/force_plot.png>

Visualize a summary plot for each class on the whole dataset.
<img src="https://github.com/KBL411/Project_AppBigData/blob/main/for_README/summary_plot%20.png"/>


