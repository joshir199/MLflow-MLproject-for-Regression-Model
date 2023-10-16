# MLflow-MLproject-for-Regression-Model
Understanding and implementing MLflow MLproject to encapsulate model, dependencies, configurations and data of a ML system for reproducibility.

# Challenge : How to reproduce the exact set of experiment that was done during development stage to any other environment ?
=> MLproject of MLflow is used to package data science/ML code in a format that enables reproducible runs on any platform.

![](https://github.com/joshir199/MLflow-project-for-Regression-Model/blob/main/MLproject%20of%20MLflow%20.png)

************************************

Mlproject contains environment definition details, parameters and class to train/validate in a YAML format.

The file can specify a name and a Conda or Docker environment, as well as more detailed information about each entry point. Specifically, each entry point defines a command to run and parameters to pass to the command (including data types).




```bash
name: My Project

python_env: python_env.yaml
# or
# conda_env: my_env.yaml
# or
# docker_env:
#    image:  mlflow-docker-example

entry_points:
  main:
    parameters:
      data_file: path
      regularization: {type: float, default: 0.1}
    command: "python train.py -r {regularization} {data_file}"
  validate:
    parameters:
      data_file: path
    command: "python validate.py {data_file}"

```

For more details. visit official ![mlflow](https://mlflow.org/docs/latest/projects.html#).
