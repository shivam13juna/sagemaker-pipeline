"""Example workflow pipeline script for abalone pipeline.

                                               . -ModelStep
                                              .
    Process-> Train -> Evaluate -> Condition .
                                              .
                                               . -(stop)

Implements a get_pipeline(**kwargs) method.
"""
import os
import boto3
import sagemaker
import sagemaker.session


from sagemaker.inputs import TrainingInput
from sagemaker.serializers import JSONSerializer
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.pytorch import PyTorch, PyTorchModel
from sagemaker.workflow.model_step import ModelStep
from sagemaker.deserializers import JSONDeserializer
from sagemaker.workflow.properties import PropertyFile
from sagemaker.debugger import TensorBoardOutputConfig
from sagemaker.sklearn import SKLearn, SKLearnProcessor
from sagemaker.pytorch.processing import PyTorchProcessor
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.processing.processing_input import ProcessingInput
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.processing.processing_output import ProcessingOutput
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.parameters import ParameterInteger, ParameterString
from sagemaker.processing import FrameworkProcessor, ProcessingInput, ProcessingOutput



BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def get_sagemaker_client(region):
     """Gets the sagemaker client.

        Args:
            region: the aws region to start the session
            default_bucket: the bucket to use for storing the artifacts

        Returns:
            `sagemaker.session.Session instance
        """
     boto_session = boto3.Session(region_name=region)
     sagemaker_client = boto_session.client("sagemaker")
     return sagemaker_client


def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )

def get_pipeline_session(region, default_bucket):
    """Gets the pipeline session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        PipelineSession instance
    """

    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")

    return PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=default_bucket,
    )

def get_pipeline_custom_tags(new_tags, region, sagemaker_project_arn=None):
    try:
        sm_client = get_sagemaker_client(region)
        response = sm_client.list_tags(
            ResourceArn=sagemaker_project_arn.lower())
        project_tags = response["Tags"]
        for project_tag in project_tags:
            new_tags.append(project_tag)
    except Exception as e:
        print(f"Error getting project tags: {e}")
    return new_tags


def get_pipeline(
    region,
    sagemaker_project_arn=None,
    role=None,
    default_bucket=None,
    model_package_group_name="IntelPackageGroup",
    pipeline_name="IntelPipeline",
    base_job_prefix="Intel",
    
    processing_instance_type="ml.m5.xlarge",
    training_instance_type="ml.m5.xlarge",
):
    """Gets a SageMaker ML Pipeline instance working with on intel data.

    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        an instance of a pipeline
    """
    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

    pipeline_session = get_pipeline_session(region, default_bucket)


    base_job_name = "pipeline-intel-emlo-s13"


    dvc_repo_url = ParameterString(
        name="DVCRepoURL", default_value="codecommit::ap-south-1://intl-emlo-s12"
    )
    dvc_branch = ParameterString(
        name="DVCBranch", default_value="pipeline-processed-dataset"
    )

    input_dataset = ParameterString(
        name="InputDatasetZip", default_value="s3://tmp-datasets/intel_s12.zip"
    )


    sklearn_processor = FrameworkProcessor(
        estimator_cls=SKLearn,
        framework_version="0.23-1",
        instance_type="ml.t3.medium",
        # instance_type="ml.m5.xlarge",
        # instance_type='local',
        instance_count=1,
        base_job_name=f"{base_job_name}-preprocess-intel-dataset",
        sagemaker_session=pipeline_session,
        # sagemaker_session=local_pipeline_session,
        role=role,
        env={
            "DVC_REPO_URL": dvc_repo_url,
            "DVC_BRANCH": dvc_branch,
            # "DVC_REPO_URL": "codecommit::ap-south-1://flower-emlo-s12",
            # "DVC_BRANCH": "processed-dataset-pipeline",
            "GIT_USER": "Shivam Prasad",
            "GIT_EMAIL": "shivam.prasad2015@vitalum.ac.in",
        },
    )

    print("Where are we: ", os.getcwd())
    print("What is in the directory: ", os.listdir("."))
    print("What is in the scripts directory: ", os.listdir("scripts"))
    print("Name of current file: ", __file__)
    print("What is in the current directory: ", os.listdir(os.path.dirname(__file__)))
    processing_step_args = sklearn_processor.run(
        code="preprocess.py",
        source_dir="scripts",
        dependencies=["requirements.txt"],
        inputs=[
            ProcessingInput(
                input_name="data",
                source=input_dataset,
                # source="s3://tmp-datasets/flower_s12.zip",
                destination="/opt/ml/processing/input",
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="train", source="/opt/ml/processing/dataset/train"
            ),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/dataset/test"),
        ],
    )


    processing_step_args


    step_process = ProcessingStep(
        name="PreprocessIntelImageClassifierDataset",
        step_args=processing_step_args,
    )


    step_process


    # ## Train Step


    tensorboard_output_config = TensorBoardOutputConfig(
        s3_output_path=f"s3://{default_bucket}/sagemaker-intel-logs-pipeline-notebook",
        container_local_output_path="/opt/ml/output/tensorboard",
    )


    # ! aws s3 cp --recursive /root/flower-project/flowers s3://sagemaker-ap-south-1-006547668672/testing/training


    # s3://sagemaker-ap-south-1-006547668672/testing/training/


    pt_estimator = PyTorch(
        base_job_name=f"{base_job_name}/training-intel-pipeline",
        source_dir="scripts",
        entry_point="train.py",
        dependencies=["requirements.txt"],
        sagemaker_session=pipeline_session,
        role=role,
        py_version="py38",
        framework_version="1.11.0",
        instance_count=1,
        instance_type="ml.m5.4xlarge",
        tensorboard_output_config=tensorboard_output_config,
        use_spot_instances=True,
        max_wait=5000,
        max_run=4800,
        environment={
            "GIT_USER": "Shivam Prasad",
            "GIT_EMAIL": "shivam.prasad2015@vitalum.ac.in",
        },
    )


    estimator_step_args = pt_estimator.fit(
        {
            # 'train': 's3://sagemaker-ap-south-1-006547668672/testing/training',
            # 'train': step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
            # # 'test': 's3://sagemaker-ap-south-1-006547668672/testing/training'
            # 'test': step_process.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "train"
                ].S3Output.S3Uri,
            ),
            "test": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "test"
                ].S3Output.S3Uri,
            ),
        }
    )


    step_train = TrainingStep(
        name="TrainIntelClassifier",
        step_args=estimator_step_args,
    )


    step_train


    # ## Eval Step


    pytorch_processor = PyTorchProcessor(
        framework_version="1.11.0",
        py_version="py38",
        role=role,
        sagemaker_session=pipeline_session,
        instance_type="ml.t3.medium",
        # instance_type="ml.c5.xlarge",
        # instance_type="ml.m5.xlarge",
        # instance_type='local',
        instance_count=1,
        base_job_name=f"{base_job_name}/eval-intel-classifier-model",
    )


    eval_step_args = pytorch_processor.run(
        code="evaluate.py",
        source_dir="scripts",
        dependencies=["requirements.txt"],
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                # source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                # source="s3://sagemaker-ap-south-1-006547668672/training-flower-pipeline-2022-12-07-03-20-21-157/output/model.tar.gz",
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                # source=step_process.properties.ProcessingOutputConfig.Outputs[
                #     "test"
                # ].S3Output.S3Uri,
                source=step_process.properties.ProcessingOutputConfig.Outputs[
                    "test"
                ].S3Output.S3Uri,
                # source="s3://sagemaker-ap-south-1-006547668672/testing/training",
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation", source="/opt/ml/processing/evaluation"
            ),
        ],
    )


    evaluation_report = PropertyFile(
        name="IntelImageClassifierEvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )
    step_eval = ProcessingStep(
        name="EvaluateIntelImageClassifierModel",
        step_args=eval_step_args,
        property_files=[evaluation_report],
    )


    step_eval


    # ## Model Metrics


    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri="{}/evaluation.json".format(
                step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"][
                    "S3Uri"
                ]
            ),
            # s3_uri="s3://sagemaker-ap-south-1-006547668672/eval-flower-classifier-model-2022-12-07-19-40-04-608/output/evaluation/evaluation.json",
            content_type="application/json",
        )
    )


    # ## Register Model Step (Conditional)


    model_approval_status = ParameterString(
        name="ModelApprovalStatus", default_value="PendingManualApproval"
    )


    model_package_group_name = "IntelImageClassifierModelGroup"


    model = PyTorchModel(
        entry_point="infer.py",
        source_dir="scripts",
        sagemaker_session=pipeline_session,
        role=role,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        # model_data="s3://sagemaker-ap-south-1-006547668672/training-flower-pipeline-2022-12-07-03-20-21-157/output/model.tar.gz",
        framework_version="1.11.0",
        py_version="py38",
    )


    model_step_args = model.register(
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=["ml.t2.medium"],
        transform_instances=["ml.m5.xlarge"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        # approval_status="PendingManualApproval",
        model_metrics=model_metrics,
    )


    step_register = ModelStep(
        name="RegisterImageClassifierModel",
        step_args=model_step_args,
    )


    step_register


    cond_gte = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=step_eval.name,
            property_file=evaluation_report,
            json_path="multiclass_classification_metrics.accuracy.value",
        ),
        right=0.4,
    )

    step_cond = ConditionStep(
        name="CheckAccuracyIntelClassifierEvaluation",
        conditions=[cond_gte],
        if_steps=[step_register],
        else_steps=[],
    )


    # ## Pipeline


    pipeline_name = "notebook-pytorch-pipeline-intel-vadi"


    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[dvc_repo_url, dvc_branch, input_dataset, model_approval_status],
        steps=[step_process, step_train, step_eval, step_cond],
        sagemaker_session=pipeline_session,
        # sagemaker_session=local_pipeline_session,
    )


    upsert_response = pipeline.upsert(
        role_arn=role, description="testing pytorch intel pipeline"
    )


    execution = pipeline.start()


    execution.list_steps()


    execution.describe()
    return pipeline
