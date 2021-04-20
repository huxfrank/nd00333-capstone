# Potentially Deadly Mushroom Classification

# Table of Contents
1. [Project Set Up and Installation](#setup)
2. [Dataset](#data)
	1. [Overview](#overview)
	2. [Task](#task)
	3. [Access](#access)
3. [Automated ML](#auotml)
	1. [Results](#automl-results)
4. [Hyperparameter Tuning](#hyper)
	1. [Results](#hyper-results)
5. [Model Deployment](#deploy)
6. [Screencast](#screencast)
7. [Standout Suggestions](#standout)

This dataset is called the Mushroom Classification dataset from Kaggle, source: https://www.kaggle.com/uciml/mushroom-classification.
The goal of this project is to be able to accurately identify poisonous and edible mushrooms based on the features and qualities provided. I will 
be using both AutoML and SKLearn+Hyperdrive to determine which method will produce a model with the highest accuracy.
"This dataset includes descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms in the Agaricus and Lepiota Family 
Mushroom drawn from The Audubon Society Field Guide to North American Mushrooms (1981). Each species is identified as definitely edible, 
definitely poisonous, or of unknown edibility and not recommended. This latter class was combined with the poisonous one. The Guide clearly states 
that there is no simple rule for determining the edibility of a mushroom; no rule like "leaflets three, let it be'' for Poisonous Oak and Ivy." 

## Project Set Up and Installation <a name="setup" />
My project has no special installation steps. I ran it on my personal AzureML account on my local machine. The notebooks are written so it can be 
run on any valid AzureML subscription.
I did find it easier to upload the dataset directly to Azure but as is showcased in my Hyperdrive training script, I can also pull the data 
directly from my github.

## Dataset <a name="data" />

### Overview <a name="overview" />
This dataset was obtained from Kaggle and it was donated to UCI ML in 1987. Although the dataset is over 30 years old, it has been updated in the 
last 5 years and is still a great real-life example of a classification problem.
The dataset contains 23 different features as follows and the goal is to predict the class:

classes: edible=e, poisonous=p  
cap-shape: bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s  
cap-surface: fibrous=f,grooves=g,scaly=y,smooth=s  
cap-color: brown=n,buff=b,cinnamon=c,gray=g,green=r,pink=p,purple=u,red=e,white=w,yellow=y  
bruises: bruises=t,no=f  
odor: almond=a,anise=l,creosote=c,fishy=y,foul=f,musty=m,none=n,pungent=p,spicy=s  
gill-attachment: attached=a,descending=d,free=f,notched=n  
gill-spacing: close=c,crowded=w,distant=d  
gill-size: broad=b,narrow=n  
gill-color: black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y  
stalk-shape: enlarging=e,tapering=t  
stalk-root: bulbous=b,club=c,cup=u,equal=e,rhizomorphs=z,rooted=r,missing=?  
stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s  
stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s  
stalk-color-above-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y  
stalk-color-below-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y  
veil-type: partial=p,universal=u  
veil-color: brown=n,orange=o,white=w,yellow=y  
ring-number: none=n,one=o,two=t  
ring-type: cobwebby=c,evanescent=e,flaring=f,large=l,none=n,pendant=p,sheathing=s,zone=z  
spore-print-color: black=k,brown=n,buff=b,chocolate=h,green=r,orange=o,purple=u,white=w,yellow=y  
population: abundant=a,clustered=c,numerous=n,scattered=s,several=v,solitary=y  
habitat: grasses=g,leaves=l,meadows=m,paths=p,urban=u,waste=w,woods=d  

### Task <a name="task" />
I will be attempting to determine if a mushroom is poisonous or edible based on the attributes listed above. I decided to use all the features 
given except veil-color and gill attachment because they have the same value for all rows.
I was not sure which features will contribute the most to the determination of poisonous or edible so I initially trained my model with all the 
features and then retrained it without veil color and gill attachment to see if it would make any difference.
Ultimately I decided, based on the results that it was ok to remove those features. They had very low importance in my runs.
The original dataset file also had t and f as values for some of the features so I made sure to convert those to booleans to give those features 
the proper representation.

### Access <a name="access" />
I used Dataset.Tabular.from_delimited_files to access this dataset in my workspace. I uploaded the dataset to my github and then proceeded to use 
the URL for the raw .csv data as my datapath.
You will see in my Hyperdrive training script that I pull the data directly from my github and utilize it for the hyperparameter tuning run.

## Automated ML <a name="automl" />
For this project, I started with a lot of the default automl configuration settings as the previous two projects.
I decided to start somewhere familiar and then if I wasn't getting the results I was expecting or I was getting results I wasn't satisfied with, I 
would tweak the parameters and configuration settings as I went but I discovered that a lot of the settings from the last two projects worked well 
for this one as well.
I used BanditPolicy (terminates runs where the primary metric is smaller than (Metric + Metric*Slack_Factor)) as my early termination method 
because it allows for early termination of low performance runs. I chose it over the other two options because it is the most aggressive about 
saving money and as I was on a personal Azure account, I didn't want to run up a large bill. Budget aside, I had no issues obtaining satisfying 
results with BanditPolicy so I decided to not change it.
I used Random Parameter Sampling because it supports early termination of low performance runs and it was the best choice of my options because I 
wasn't doing a lot of runs which Bayesian sampling needs (ideally 20+). I also wanted to sweep over a range of continuous hyperparameter values so 
grid sampling was out because it only supports discrete hyperparameter values.

### Results <a name="automl-results" />
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning <a name="hyper" />
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the 
hyperparameter search


### Results <a name="hyper-results" />
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment <a name="deploy" />
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording <a name="screencast" />
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions <a name="standout" />
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
