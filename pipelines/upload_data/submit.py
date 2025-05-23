"""Federated Learning Cross-Silo pipeline for uploading data to the silos' storages.
This script:
1) reads a config file in yaml specifying the number of silos and their parameters,
2) reads the components from a given folder,
3) builds a flexible pipeline depending on the config,
4) configures each step of this pipeline to write to the right storage.
"""
import os
import argparse
import random
import string
import datetime
import webbrowser
import time
import sys

# Azure ML sdk v2 imports
import azure
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml import MLClient, Output
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.dsl import pipeline
from azure.ai.ml import load_component

# to handle yaml config easily
from omegaconf import OmegaConf


############################
### CONFIGURE THE SCRIPT ###
############################

parser = argparse.ArgumentParser(description=__doc__)

# load the config from a local yaml file
YAML_CONFIG = OmegaConf.load(os.path.join(os.path.dirname(__file__), "config.yaml"))

# path to the components
COMPONENTS_FOLDER = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "components",
)


###########################
### CONNECT TO AZURE ML ###
###########################


def connect_to_aml():
    try:
        credential = DefaultAzureCredential()
        # Check if given credential can get token successfully.
        credential.get_token("https://management.azure.com/.default")
    except Exception as ex:
        # Fall back to InteractiveBrowserCredential in case DefaultAzureCredential not work
        credential = InteractiveBrowserCredential()

    # Get a handle to workspace
    try:
        # tries to connect using cli args if provided else using config.yaml
        ML_CLIENT = MLClient(
            subscription_id= YAML_CONFIG.aml.subscription_id,
            resource_group_name= YAML_CONFIG.aml.resource_group_name,
            workspace_name= YAML_CONFIG.aml.workspace_name,
            credential=credential,
        )

    except Exception as ex:
        print("Could not find config.yaml.")
        # tries to connect using local config.json
        ML_CLIENT = MLClient.from_config(credential=credential)

    return ML_CLIENT


####################################
### LOAD THE PIPELINE COMPONENTS ###
####################################

# Loading the component from their yaml specifications
upload_data_component = load_component(
    source=os.path.join(COMPONENTS_FOLDER, "upload_data", "spec.yaml")
)


########################
### BUILD A PIPELINE ###
########################


def custom_fl_data_path(datastore_name, output_name, iteration_num=None):
    """Produces a path to store the data.

    Args:
        datastore_name (str): name of the Azure ML datastore
        output_name (str): a name unique to this output

    Returns:
        data_path (str): direct url to the data path to store the data
    """
    return (
        f"azureml://datastores/{datastore_name}/paths/federated_learning/{output_name}/"
    )


@pipeline(
    description=f"FL cross-silo upload data pipeline.",
)
def fl_cross_silo_upload_data():
    silos = YAML_CONFIG.federated_learning.silos

    for silo_index, silo_config in enumerate(silos):
        # create step for upload component
        silo_upload_data_step = upload_data_component(
            silo_count=len(silos), silo_index=silo_index
        )
        # if confidentiality is enabled, add the keyvault and key name as environment variables
        silo_upload_data_step.environment_variables = {
            "CONFIDENTIALITY_DISABLE": "True",
        }

        # add a readable name to the step
        silo_upload_data_step.name = f"silo_{silo_index}_upload_data"

        # make sure the compute corresponds to the silo
        silo_upload_data_step.compute = silo_config.compute

        # assign instance type for AKS, if available
        if hasattr(silo_config, "instance_type"):
            if silo_upload_data_step.resources is None:
                silo_upload_data_step.resources = {}
            silo_upload_data_step.resources["instance_type"] = silo_config.instance_type

        # make sure the data is written in the right datastore
        silo_upload_data_step.outputs.raw_train_data = Output(
            type=AssetTypes.URI_FOLDER,
            mode="mount",
            path=custom_fl_data_path(
                silo_config.datastore,
                "ccfraud/raw_train_data",
            ),
        )
        silo_upload_data_step.outputs.raw_test_data = Output(
            type=AssetTypes.URI_FOLDER,
            mode="mount",
            path=custom_fl_data_path(
                silo_config.datastore,
                "ccfraud/raw_test_data",
            ),
        )


pipeline_job = fl_cross_silo_upload_data()

# Inspect built pipeline
print(pipeline_job)

print("Submitting the pipeline job to your AzureML workspace...")
ML_CLIENT = connect_to_aml()
pipeline_job = ML_CLIENT.jobs.create_or_update(
    pipeline_job, experiment_name="fl_demo_upload_data"
)

print("The url to see your live job running is returned by the sdk:")
print(pipeline_job.services["Studio"].endpoint)

webbrowser.open(pipeline_job.services["Studio"].endpoint)
