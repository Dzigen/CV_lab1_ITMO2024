from src.neural_nets import *
from src.utils import *
from src.train_utils import *
from transformers import AutoImageProcessor
import json
from torch.utils.data import DataLoader

##############################

# По конфигу начать обучение
with open(CONFIG_FILE_JSON, 'r', encoding='utf-8') as fd:
    json_obj = json.loads(fd.read())
learn_config = LearningConfig(**json_obj)

##############################

if learn_config.model_name == 'resnet':
    model = ResNet50Classifier(14).to(learn_config.device)
    proc_f = AutoImageProcessor.from_pretrained(RESNET_MODEL_NAME)
    processor = lambda x: torch.tensor(proc_f(x)['pixel_values'][0])
    print("Выбран ResNet-based классификатор")

elif learn_config.model_name == 'mask2former':
    model = Mask2FormerClassifier(14).to(learn_config.device)
    proc_f = AutoImageProcessor.from_pretrained(M2F_MODEL_NAME)
    processor = lambda x: torch.tensor(proc_f(x)['pixel_values'][0])
    print("Выбран Mask2Former-based классификатор")

elif learn_config.model_name == 'mycnn':
    model = MyCNNClassifier(14).to(learn_config.device)
    processor = TRANSFORM_MYCNN
    print("Выбран самописный СNN-based классификатор")

else:
    print("ERROR: Invalid model name!")

##############################
    
train_dataset = CustomCarDataset('train', TT_INFO, processor)
eval_dataset = CustomCarDataset('eval', TT_INFO, processor)

print(len(train_dataset), len(eval_dataset))

# Dataloader
train_dataloader = DataLoader(train_dataset, batch_size=learn_config.batch_size, 
                              shuffle=True, 
                              collate_fn=custom_collate,
                              num_workers=2)
eval_dataloader = DataLoader(eval_dataset, batch_size=learn_config.batch_size,
                              collate_fn=custom_collate,
                              num_workers=2)


###############################

run(learn_config, model, train_dataloader, eval_dataloader)