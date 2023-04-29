import requests
import psycopg2

# Initial config
vm_ip = "34.176.217.52"  # By default the internal ip used by mlflow is 127.0.0.1, but to externalize the model the external  ip of the vm must be written here
vm_port = "5001"

db_pass = "12345"
db_ip = "34.176.139.210"
db_name = "postgres"
db_user = "postgres"
db_port = "5432"


def parse_request(request):

    event_id = request.pop('event_id') if 'event_id' in request else 'no_event_id'

    features = request["data"]
    assert len(features) == 4, 'The request must have the correct ammount of columns (At least)'

    return event_id, features


def save_predictions(event_id, prediction):

    conn_string = f"host='{db_ip}' dbname='{db_name}' user='{db_user}' password='{db_pass}'"
    conn = psycopg2.connect(conn_string)
    cursor = conn.cursor()
    cursor.execute(f"""insert into public.inference(id, value) values('{event_id}',{prediction})""")
    conn.commit()  # <- We MUST commit to reflect the inserted data
    cursor.close()
    conn.close()


def externalized_model(features) -> str:
    headers = {}

    json_request = {'dataframe_split': {'data':[features]}}

    response = requests.post(f'http://{vm_ip}:{vm_port}/invocations', headers=headers, json=json_request)
    response = response.json()["predictions"]

    return response[0]


def check_business_logic(features):
    """
    Here We should put some business logic
    We're going to put the min/max of the iris dataset features
    """
    sepal_length, sepal_width, petal_length, petal_width = features

    sepal_length_condition = (4.3 <= sepal_length) and (sepal_length <= 7.9)
    sepal_width_condition = (2.0 <= sepal_width) and (sepal_width <= 4.4)
    petal_length_condition = (1.0 <= petal_length) and (petal_length <= 6.9)
    petal_width_condition = (0.1 <= petal_width) and (petal_width <= 2.5)

    return not (sepal_length_condition and sepal_width_condition and petal_length_condition and petal_width_condition)


def get_business_prediction(features):
    "Here we could have some function of the feature"

    return -1


def trigger_events(request):
    # This pipeline supposes that we are using it to do just one inference at the time
    # If we want to do more inferences, we need to modify the functions to be able to handle them

    # This is to work inside the lambda fn
    event_id, features = parse_request(request)

    # if check_business_logic(features):
    #     prediction = get_business_prediction(features)

    prediction = externalized_model(features)

    save_predictions(event_id, prediction)

    return str(prediction)


print(trigger_events(
   {"event_id": "17cfe7d5-3cdb-4e62-861d-0371b79f16f2", "data": [0, 0, 0, 0]}
))

print(trigger_events(
   {"event_id": "e1a9d8f8-d553-4b97-83ab-37cd9c73c1fc", "data": [5.1, 3.5, 1.4, 0.2]}
))


# print(trigger_events(
#    {"event_id": "event_id_1", "dataframe_split": {"data":[[10,10,10,10]]}}
#    ))

# print(trigger_events(
#    { "dataframe_split": {"data":[[0,0,0,0]]}}
#    ))

# print(trigger_events(
#    { "dataframe_split": {"data":[[19,10,10,10]]}}
#    ))
