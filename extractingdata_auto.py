from dotenv import load_dotenv, find_dotenv
import os
import requests
from requests import session

payload={
    "action":"login",
    "username":os.environ.get("KAGGlE_USERNAME"),
    "password":os.environ.get("KAGGLE_PASSWORD")
}

login_url='https://www.kaggle.com/account/login'
train_url="https://www.kaggle.com/c/titanic/download/train.csv"
test_url="https://www.kaggle.com/c/titanic/download/test.csv"

raw_path=os.path.join(os.path.pardir)
test_path=os.path.join(raw_path,'test.csv')
train_path=os.path.join(raw_path,'train.csv')


def extract_data(url, file_path):
    with session() as c:
        c.post(login_url, data=payload)
        with open(file_path, 'w') as handle:
            response=c.get(url, stream=True)
            for block in response.iter_content(1024):
                handle.write(block)
    

dot_env_path=find_dotenv()
load_dotenv(dot_env_path)
username=os.environ.get("KAGGlE_USERNAME")

print(username)

extract_data(train_url, train_path)
extract_data(test_url, test_path)

