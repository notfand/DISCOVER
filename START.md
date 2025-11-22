```shell
python -m venv venv
venv/Scripts/activate
pip install -r requirements.txt
pip install tensorboard==1.15
pip install protobuf==3.20

python test_pde.py KdV MODE
```