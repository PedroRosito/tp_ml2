import mlflow

db_pass = "12345"
db_ip = "34.176.139.210"
db_name = "postgres"
db_user = "postgres"
db_port = "5432"

model_registry_uri = f'postgresql+psycopg2://{db_user}:{db_pass}@{db_ip}:{db_port}/{db_name}'
mlflow.set_tracking_uri(model_registry_uri)
mlflow.set_registry_uri(model_registry_uri)

list_mlflow_experiments = mlflow.search_experiments()
list_experiment_id = list(
    map(
        lambda list_mlflow_experiments: int(
            list_mlflow_experiments.experiment_id),
        list_mlflow_experiments))
last_experiment_id = max(list_experiment_id)


runs = mlflow.search_runs(experiment_ids=[last_experiment_id])

best_model_run_id = runs.sort_values(
    by=['metrics.test_acc'], ascending=False).iloc[0]['run_id']
print(best_model_run_id)

new_model = mlflow.register_model(
    f'runs:/{best_model_run_id}/model', 'ml2_uba')


client = mlflow.client.MlflowClient()
client.transition_model_version_stage('ml2_uba',
                                      new_model.version,
                                      "production",
                                      archive_existing_versions=True)
