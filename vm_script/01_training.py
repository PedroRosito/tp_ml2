import mlflow
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

import pandas as pd

db_pass = "12345"
db_ip = "34.176.139.210"
db_name = "postgres"
db_user = "postgres"
db_port = "5432"

# This db could be an external postgres database
mlflow.set_tracking_uri(
    f'postgresql+psycopg2://{db_user}:{db_pass}@{db_ip}:{db_port}/{db_name}')

# This will fail in databricks because the experiment_id is a random hash

new_experiment_id = 0
list_mlflow_experiments = mlflow.search_experiments()
if len(list_mlflow_experiments):
    list_experiment_id = list(
        map(lambda list_mlflow_experiments: int(
            list_mlflow_experiments.experiment_id),
            list_mlflow_experiments))
    last_experiment_id = max(list_experiment_id)
    new_experiment_id = last_experiment_id + 1

    mlflow.create_experiment(str(new_experiment_id))

df_heart = pd.read_csv("/heart_failure_clinical_records_dataset.csv")

features = [
    'age', 'ejection_fraction', 'serum_creatinine', 'serum_sodium', 'time']
x = df_heart[features]
y = df_heart["DEATH_EVENT"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                    random_state=2)


mlflow.sklearn.autolog(max_tuning_runs=None)


# In[4]:


def log_model(model,
              developer=None,
              experiment_id=None,
              grid=False,
              **kwargs):

    assert developer is not None, 'You must define a developer first'
    assert experiment_id is not None, 'You must define a experiment_id first'

    with mlflow.start_run(experiment_id=experiment_id):

        mlflow.set_tag('developer', developer)

        model = model(**kwargs)
        if grid:
            model = GridSearchCV(model, param_grid=kwargs)

        model.fit(x_train, y_train)
        test_acc = (model.predict(x_test) == y_test).mean()

        mlflow.log_metric('test_acc', test_acc)


# In[5]:

log_model(DecisionTreeClassifier, 'Fede-Pedro-Chris',
          experiment_id=new_experiment_id)
log_model(
    LogisticRegression, 'Fede-Pedro-Chris',
    experiment_id=new_experiment_id, **{'max_iter': 1000})
log_model(
    SVC, 'Fede-Pedro-Chris',
    experiment_id=new_experiment_id,
    **{'C': 0.001, 'class_weight': 'balanced'})
log_model(
    GradientBoostingClassifier, 'Fede-Pedro-Chris',
    experiment_id=new_experiment_id,
    **{'max_depth': 2, 'random_state': 1})

log_model(SVC,
          'Fede-Pedro-Chris',
          experiment_id=new_experiment_id,
          grid=True,
          **{'kernel': ('linear', 'rbf'), 'C': [1, 10]}
          )

# %%
