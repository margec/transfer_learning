# Finetune a pretrained image classification CNN to classify dog breed

**TODO:** Write a short introduction to your project.

This project took a RESNET18 pretrained network, as a fixed feature extractor, and replaced its last FC layer with a new one with random weights, and only train that layer with a set of dog images, goal is to predict the breed of an input dog image.

AWS SageMaker is used for different tasks throughout the project:

- hyperparameters tuning
- model training (hooked with SageMaker debugger and profiler) 
- deploy trained model
- inference hosting

## Project Set Up and Installation
**OPTIONAL:** If your project has any special installation steps, this is where you should put it. To turn this project into a professional portfolio project, you are encouraged to make your `README` detailed and self-explanatory. For instance, here you could explain how to set up your project in AWS and provide helpful screenshots of the process.

## Dataset

### Overview
**TODO**: Explain about the data you are using and where you got it from.

i am using a dataset from the below Udacity S3 bucket: 

https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip

it has a total of 8351 images of dogs, they are of 133 breeds, so there are 133 classes.

### Access
**TODO**: Explain how you are accessing the data in AWS and how you uploaded it

- download the zip file from Udacity S3 bucket to my SageMaker Studio JupyterLab space

- unzip the files (which have already been split into train, validate, and test sets)

- use AWS CLI to copy / upload the files to a S3 bucket i created for this project

- let the training job know where the training data is by calling HyperparameterTuner or PyTorch estimator fit({ "training": our_s3_uri }) with inputs dict that points the "training" channel to our S3 bucket


## Hyperparameter Tuning
**TODO**: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search

- RESNET18 was chosen as the pretrained model because i want to start with something simpler (with 18 layers) which will take shorter time to train, and move up to more complicated ones like RESNET50 if needed

- hyperparameters 
  1. learning rate (0.0005, 0.005), i chose a range around a well known optimal value of 0.001
  2. batch size [16, 32, 128, 256], i included both smaller sizes and bigger sizes and skipping the middle one like 64, also i tune learning rate and batch size together because they are highly correlated
  3. epochs (the number of iterations) [5, 10], the idea is again to start with smaller numbers which will be faster

Remember that your README should:
- Include a screenshot of completed training jobs

```
screen_shots -> Hyperparameter_tuning_job_completed.png
screen_shots -> Training_jobs_during_hyperparameter_tuning_completed.png
screen_shots -> Training_job_with_best_hyperparameters_completed.png
```

- Logs metrics during the training process

Hyperparameter tuning took 15 mins, while the training job afterwards (with the best hyperparameters) took 11 mins. 

- Tune at least two hyperparameters

i tuned 3 hyperparameters.

- Retrieve the best best hyperparameters from all your training jobs

the best is from job 001:

```
{
 'batch_size': '"16"',
 'epochs': '"5"',
 'learning_rate': '0.0006684340499967713'
} 
```

## Debugging and Profiling
**TODO**: Give an overview of how you performed model debugging and profiling in Sagemaker

- we define the (debugging and profiling) rules and set up hook configuration (how often we save the tensors), and pass them to the estimator
- then in the container (entry point) code
  - we create a hook and register it with both the model and loss function
  - we pass the hook to both train and test subroutines
  - at training time, set the hook mode to TRAIN
  - at testing time, set the hook mode to EVAL

### Results
**TODO**: What are the results/insights did you get by profiling/debugging your model?

the debugger has reported an issue of poor weight initialization, which is not a usual complaint for transfer learning, i suspect there might be a domain mismatch between the pretrained set and our training set.

**TODO** Remember to provide the profiler html/pdf file in your submission.

```
profiler-output -> profiler-report.html
```

## Model Deployment
**TODO**: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

the trained model was deployed to a single instance of type 'ml.m5.large'.

how to query the endpoint: 

- read a test image (jpeg) file into a chunk of bytes
- call the Predictor interface predict() method with the chunk of bytes as parameter, and gets returned a prediction result tensor

**TODO** Remember to provide a screenshot of the deployed active endpoint in Sagemaker.

```
screen_shots -> SageMaker_deployed_endpoint.png
```


## Standout Suggestions
**TODO (Optional):** This is where you can provide information about any standout suggestions that you have attempted.
