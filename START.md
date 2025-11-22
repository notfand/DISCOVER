python版本：3.7.7
```shell
python -m venv venv
venv/Scripts/activate
pip install -r requirements.txt
pip install protobuf==3.20
pip install tensorboard==1.15

python test_pde.py KdV MODE
```