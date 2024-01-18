from datasets import Dataset, DatasetDict
from transformers import T5ForConditionalGeneration, T5Config, T5Tokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import datasets
from typing import Optional
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.trainer import is_datasets_available, seed_worker
from datasets import load_dataset

lk = 6
cuda = 1

a = (lk-1)*150000
b = (lk)*150000
tokenized_sur_dataset = load_dataset("carnival13/sur_test_rt5_few_8", split=f"train[{a}:{b}]")


class RankT5(T5ForConditionalGeneration):
    def __init__(self, config: T5Config):
        config.rank_score_index = 32019
        config.n_pass = 10
        config.output_hidden_states = False
        super().__init__(config)
#         self.rank_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.rank_id = config.rank_score_index
        self.n_pass = config.n_pass

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None, labels=None, pass_label=None, **kwargs):

        batch_size_n, seq_len = input_ids.size()
        batch_size = int(batch_size_n/self.n_pass)
        labels = None

        # input_ids = input_ids.view(batch_size*n_pass, -1)
        # attention_mask = attention_mask.view(batch_size*n_pass, -1)
        if decoder_input_ids == None and labels == None:
            decoder_input_ids = torch.zeros((batch_size_n, 1), dtype=int).to(input_ids.device)
        
        
        
        if labels != None and decoder_input_ids == None:
#             batch_size, decoder_seq_len = labels.size()
#             labels = labels.view(batch_size, 1, decoder_seq_len).contiguous()
#             labels = labels.expand(batch_size, n_pass, decoder_seq_len).contiguous()

#             labels = labels.view(batch_size*n_pass, -1)
            decoder_input_ids = self._shift_right(labels)
            print(decoder_input_ids.size())


        out = super().forward(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, **kwargs)
#         rank_score = self.rank_head(out.decoder_hidden_states[-1][:, 0, :])
        rank_score = out.logits[:, 0, :] 
        out.rank_score = rank_score[:, self.rank_id].view(-1, self.n_pass)
        loss = None
        assert out.decoder_hidden_states == None

        if pass_label != None:
            pass_label = pass_label[::self.n_pass]
            rank_score = out.rank_score
#             gen_score = out.gpe_score

            loss_fct1 = nn.CrossEntropyLoss()
#             loss_fct2 = nn.CrossEntropyLoss()

            loss = loss_fct1(rank_score, pass_label.view(-1))

        ret =  Seq2SeqLMOutput(
            loss=loss,
            logits=out.logits,
            past_key_values=out.past_key_values,
            decoder_hidden_states=out.decoder_hidden_states,
            decoder_attentions=out.decoder_attentions,
            cross_attentions=out.cross_attentions,
            encoder_last_hidden_state=out.encoder_last_hidden_state,
            encoder_hidden_states=out.encoder_hidden_states,
            encoder_attentions=out.encoder_attentions,
        )
        ret.rank_score = out.rank_score
        return ret
        
    
model_ckpt = "massive_da_eng_sur_rt5/final"
model = RankT5.from_pretrained(model_ckpt)
tokenized_sur_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

from transformers import AutoTokenizer
mod_ckp = "t5-base"
tokenizer = AutoTokenizer.from_pretrained(mod_ckp)
question = "query:"
title = "intent:"
context = "examples:"
eou = "<eou>"
tokenizer.add_tokens([question, title, context, eou], special_tokens=True)


data_collator = DataCollatorForSeq2Seq(tokenizer)
dl = DataLoader(tokenized_sur_dataset, collate_fn=data_collator, batch_size=80, shuffle=False)

device = f"cuda:{cuda}" if torch.cuda.is_available() else "cpu"
# assert device == "cuda:2"

from tqdm import tqdm
model = model.to(device)
batch_len = []
labels = []
scores = []
jdf = 0

model.eval()
with torch.no_grad():
    for b in tqdm(dl):
        batch_len.append(int(b["input_ids"].size()[0]))
        input_ids = b["input_ids"].to(device)
        attention_mask = b["attention_mask"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        if jdf == 0:
            print(outputs.rank_score.view(-1).shape)
            jdf+=1
#         outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        rank_score = list(outputs.rank_score.view(-1))
        scores += rank_score
        del input_ids
        del attention_mask
        del outputs
        
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