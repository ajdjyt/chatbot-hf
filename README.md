# Chatbot app with api using huggingface  
This app runs an api using fastapi and provides a tinyLLaMA-2 model with 1.1B parameters.  
## Dependencies
Install pytorch using  
```
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia  
```
and the other dependencies using  
```
pip install -r requirements.txt  
```
## Running  
```
uvicorn main:app --host 0.0.0.0 --port 8000  
```
Where 0.0.0.0 is to be replaced by the ip you want to provide the api on    
and 8000 is to be replaced by the port  
