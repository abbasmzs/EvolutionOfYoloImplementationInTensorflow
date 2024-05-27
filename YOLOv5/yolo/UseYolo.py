from model.yolo import Yolo
from configs.config import params

model = Yolo(yaml_dir=params['yaml_dir'])(512)
model.summary()