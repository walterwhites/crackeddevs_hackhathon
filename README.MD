# ML Search Function – CrackedDevs Project

## Notebooks

### request_api
This notebook is dedicated to data collection through the CrackedDevs API. It focuses on acquiring and compiling data necessary for further analysis and model training.

### model
This notebook is where the model training takes place. It encompasses the entire process of developing, training, and validating the model, ensuring it performs effectively for its intended purpose.

## Datasets

### jobs_data.csv
This dataset is sourced directly from the CrackedDevs API. It contains the initial, raw data which serves as the foundation for subsequent preprocessing and analysis.

### cleaned_jobs_data.csv
This dataset represents the processed form of the original data. It undergoes various preprocessing steps, such as lemmatization and tokenization, to prepare it for more efficient and accurate model training and analysis.

## Run the API in local

```
pip install fastapi uvicorn
uvicorn api.app:app --reload --port 9876
```

Then navigate to http://127.0.0.1:9876/docs