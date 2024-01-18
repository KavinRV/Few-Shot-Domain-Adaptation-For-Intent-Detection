from datasets import Dataset, DatasetDict
from transformers import RobertaForSequenceClassification, RobertaConfig, RobertaTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding, AutoTokenizer, AutoModelForSequenceClassification
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import datasets
from typing import Optional
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.trainer import is_datasets_available, seed_worker
from datasets import load_dataset


cuda = 2

tokenized_sur_dataset = load_dataset("carnival13/rbrt_test_val_lrg3", split=f"train")


#create a gradient reversal layer model for the re-ranker 
class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None
    
#add the gradient reversal layer to the model
class GRL(nn.Module):
    def __init__(self, alpha):
        super(GRL, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalLayer.apply(x, self.alpha)
    


class ReRanker(RobertaForSequenceClassification):
    def __init__(self, config: RobertaConfig):
        # config.rank_score_index = 32019
        config.n_pass = 10
        # config.output_hidden_states = True
        super().__init__(config)
        self.da = config.da
        self.n_pass = config.n_pass
        if self.da:
            config.num_labels = config.num_domain
            self.domain_classifier = RobertaClassificationHead(config)
            self.grl = GRL(1)


    def forward(self, input_ids=None, attention_mask=None, domain_label=None, pass_label=None, **kwargs):

        batch_size_n, seq_len = input_ids.size()
        batch_size = int(batch_size_n/self.n_pass)

        out = super().forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
#         rank_score = self.rank_head(out.decoder_hidden_states[-1][:, 0, :])
        rank_score = out.logits.view(batch_size, -1)
        loss = None
        da_loss = None
        rank_loss = None

        if domain_label != None:
            # print(out.hidden_states[-1].shape)
            domain_logits = self.domain_classifier(self.grl(out.hidden_states[-1]))
            loss_fct = nn.CrossEntropyLoss()
            da_loss = loss_fct(domain_logits, domain_label.view(-1))
            # wandb.log({"da_loss": da_loss})

        if pass_label != None:
            pass_label = pass_label[::self.n_pass]
            rank_score = rank_score
#             gen_score = out.gpe_score

            loss_fct1 = nn.CrossEntropyLoss()
#             loss_fct2 = nn.CrossEntropyLoss()

            rank_loss = loss_fct1(rank_score, pass_label.view(-1))
#             gen_loss = loss_fct2(gen_score, pass_label.view(-1))

#             loss = rank_loss + gen_loss

            # loss = rank_loss
            # wandb.log({"loss": rank_loss})
        try:
            loss = rank_loss + da_loss if self.da else rank_loss
        except:
            loss = None
        # wandb.log({"overall_loss": loss})

        ret =  SequenceClassifierOutput(
            loss=loss,
            logits=out.logits,
            hidden_states=out.hidden_states,
            attentions=out.attentions
        )
        ret.rank_score = rank_score
        return ret
    


# config.output_hidden_states = True
# config.n_pass = 10
# config.da = True
# config.num_labels = 1
# config.num_domain = 2    
model_ckpt = "rbrt_ls_uda_ep5_large2/checkpoint-4700"
config = RobertaConfig.from_pretrained(model_ckpt)
config.num_labels = 1
model = ReRanker.from_pretrained(model_ckpt, config=config)
tokenized_sur_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

from transformers import AutoTokenizer
mod_ckp = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(mod_ckp)


data_collator = DataCollatorWithPadding(tokenizer)
dl = DataLoader(tokenized_sur_dataset, collate_fn=data_collator, batch_size=40, shuffle=False)

device = f"cuda:{cuda}" if torch.cuda.is_available() else "cpu"
print(device)

from tqdm import tqdm
model = model.to(device)
batch_len = []
labels = []
scores = []

model.eval()
with torch.no_grad():
    for b in tqdm(dl):
        batch_len.append(int(b["input_ids"].size()[0]))
        input_ids = b["input_ids"].to(device)
        attention_mask = b["attention_mask"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
#         outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        rank_score = list(outputs.logits)
        scores += rank_score
        
dct = {"indoml_id": [], "intent_id": []}
scores2 = torch.tensor(scores).view(-1, 150)
prediction = [int(x) for x in list(torch.argmax(scores2, dim=1))]

lst = []
for i, eg in enumerate(prediction):
    dct["indoml_id"].append(i)
    dct["intent_id"].append(eg)
    # print(tokenized_sur_dataset[i])
    lst.append(int(eg == tokenized_sur_dataset[i]["label"]))
    
def dict_to_dataset(dict):
    dataset = datasets.Dataset.from_dict(dict)
    return dataset

print(f"Accuracy: {sum(lst)/len(lst)}")

# pred = dict_to_dataset(dct)
# pred.save_to_disk(f"test")
