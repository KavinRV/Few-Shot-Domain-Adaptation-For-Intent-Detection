from datasets import Dataset, DatasetDict
from transformers import XLMRobertaForSequenceClassification, XLMRobertaConfig, XLMRobertaTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding, AutoTokenizer, AutoModelForSequenceClassification
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaClassificationHead
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import datasets
from typing import Optional
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.trainer import is_datasets_available, seed_worker
from datasets import load_dataset

lk = 1
cuda = 4
gst = 300000

a = (lk-1)*gst
b = (lk)*gst

tokenized_sur_dataset = load_dataset("carnival13/test_da_xlmr", split=f"train[{a}:{b}]")


class ReRanker(XLMRobertaForSequenceClassification):
    def __init__(self, config: XLMRobertaConfig):
        # config.rank_score_index = 32019
        config.n_pass = 10
        # config.output_hidden_states = True
        super().__init__(config)
#         self.rank_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # self.rank_id = config.rank_score_index
        # self.da = config.da
        self.n_pass = config.n_pass


    def forward(self, input_ids=None, attention_mask=None, labels=None, pass_label=None, **kwargs):

        batch_size_n, seq_len = input_ids.size()
        batch_size = int(batch_size_n/self.n_pass)
        labels = None


        out = super().forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
#         rank_score = self.rank_head(out.decoder_hidden_states[-1][:, 0, :])
        rank_score = out.logits.view(batch_size, -1)
        loss = None


        if pass_label != None:
            pass_label = pass_label[::self.n_pass]
            rank_score = rank_score
#             gen_score = out.gpe_score

            loss_fct1 = nn.CrossEntropyLoss()
#             loss_fct2 = nn.CrossEntropyLoss()

            rank_loss = loss_fct1(rank_score, pass_label.view(-1))
#             gen_loss = loss_fct2(gen_score, pass_label.view(-1))

#             loss = rank_loss + gen_loss

            loss = rank_loss

        ret =  SequenceClassifierOutput(
            loss=loss,
            logits=out.logits,
            hidden_states=out.hidden_states,
            attentions=out.attentions
        )
        ret.rank_score = rank_score
        return ret
    
model_ckpt = "rbrt_hard_curr_uda_ep3_corr/final"
model = ReRanker.from_pretrained(model_ckpt)
tokenized_sur_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

from transformers import AutoTokenizer
mod_ckp = "cartesinus/xlm-r-base-amazon-massive-intent-label_smoothing"
tokenizer = AutoTokenizer.from_pretrained(mod_ckp)


data_collator = DataCollatorWithPadding(tokenizer)
dl = DataLoader(tokenized_sur_dataset, collate_fn=data_collator, batch_size=450, shuffle=False)

device = f"cuda:{cuda}" if torch.cuda.is_available() else "cpu"
# assert device == "cuda:2"

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

for i, eg in enumerate(prediction):
    dct["indoml_id"].append(i)
    dct["intent_id"].append(eg)
    
def dict_to_dataset(dict):
    dataset = datasets.Dataset.from_dict(dict)
    return dataset

pred = dict_to_dataset(dct)
pred.save_to_disk(f"test_{lk}")