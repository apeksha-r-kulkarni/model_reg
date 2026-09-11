import mlflow as mlf
import joblib

mlf.set_tracking_uri("http://127.0.0.1:5000")

#model logging
model=joblib.load("model.pkl")

#MLflow code
with mlf.start_run() as run:
    #model logging
    mlf.sklearn.log_model(model_uri =model, name="simple classifier")
    #gets the run id of the current run
    run_id=run.info.run_id
    #model registration logic
    result=mlf.register_model(
        model_uri=f"runs:/{run_id}/simple classifier",
        name="simple classifier"
    )
    
