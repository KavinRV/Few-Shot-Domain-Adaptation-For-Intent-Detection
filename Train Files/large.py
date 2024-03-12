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
import wandb

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="massive_da",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 2e-05,
    "architecture": "Roberta_CE",
    "dataset": "massive_eng_sur_hard_uda",
    "epochs": 5,
    }
)

tokenized_massive = load_dataset("carnival13/rbrt_uda_large_ep13").remove_columns(["input"])
tokenized_eval = load_dataset("carnival13/rbrt_eval_sur_lrg3").remove_columns(["input"])
jk = []


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
            self.num_domain = config.num_domain
            self.domain_classifier = RobertaClassificationHead(config)
            self.grl = GRL(1.15)
            self.domain_loss = nn.CrossEntropyLoss()
        self.rank_loss = nn.CrossEntropyLoss(label_smoothing=0.1)


    def forward(self, input_ids=None, attention_mask=None, domain_label=None, pass_label=None, **kwargs):

        batch_size_n, seq_len = input_ids.size()
        batch_size = int(batch_size_n/self.n_pass)

        out = super().forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
#         rank_score = self.rank_head(out.decoder_hidden_states[-1][:, 0, :])
        rank_score = out.logits[pass_label != -100].contiguous().view(-1, self.n_pass)
        loss = None
        da_loss = None
        x = 1

        if domain_label != None:
            # print(out.hidden_states[-1].shape)
            domain_logits = self.domain_classifier(self.grl(out.hidden_states[-1])).view(-1, self.n_pass, self.num_domain).mean(dim=1)
            domain_label = domain_label[::self.n_pass]
            da_loss = self.domain_loss(domain_logits, domain_label.view(-1))

        if pass_label != None:
            pass_label = pass_label[pass_label != -100].contiguous()[::self.n_pass]
            rank_score = rank_score

            rank_loss = self.rank_loss(rank_score, pass_label.view(-1))
            if -100 in list(pass_label):
                rank_loss = 1.05 * rank_loss
                x = 1.05

        if torch.isnan(rank_loss):
            # print("rank_loss is nan")
            loss = da_loss
            # rank_loss = 0
        else:
            loss = rank_loss + 0.85*da_loss

        with torch.no_grad():
            wandb.log({"overall_loss": loss, "loss": rank_loss/x, "da_loss": da_loss})

        if not self.training:
            loss = rank_loss

        ret =  SequenceClassifierOutput(
            loss=loss,
            logits=out.logits,
            hidden_states=out.hidden_states,
            attentions=out.attentions
        )
        ret.rank_score = rank_score
        return ret
    

mod_ckp = "roberta-large-mnli"
tokenizer = AutoTokenizer.from_pretrained(mod_ckp)
# question = "query:"
# title = "intent:"
# context = "examples:"
# eou = "<sep>"
# print
# tokenizer.add_tokens([question, title, context, eou], special_tokens=True)
config = RobertaConfig.from_pretrained(mod_ckp)
config.output_hidden_states = True
# config.rank_score_index = tokenizer.convert_tokens_to_ids("<extra_id_80>")
# config.vocab_size = config.vocab_size + 4
config.n_pass = 10
config.da = True
config.num_labels = 1
config.num_domain = 2
model = ReRanker.from_pretrained(mod_ckp, config=config, ignore_mismatched_sizes=True)



batch_size = 1*model.config.n_pass
model_dir = f"rbrt_ls_uda_ep5_large2"

args = TrainingArguments(
    model_dir,
    label_names=["pass_label", "domain_label"],
    evaluation_strategy="steps",
    eval_steps=100,
    logging_strategy="steps",
    logging_steps=100,
    save_strategy="steps",
    save_steps=100,
    learning_rate=4e-6,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.015,
    save_total_limit=3,
    num_train_epochs=1,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    gradient_accumulation_steps=16,
    fp16=True
)


data_collator = DataCollatorWithPadding(tokenizer)


class CustomTrainer(Trainer):
#     def compute_loss(self, model, inputs, return_outputs=False):
# #         print(inputs.get("input_ids").size())
#         outputs = model(**inputs)

#         loss = outputs.loss
#         return (loss, outputs) if return_outputs else loss

    def get_train_dataloader(self) -> DataLoader:
        # """
        # Returns the training [`~torch.utils.data.DataLoader`].

        # Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        # training if necessary) otherwise.

        # Subclass and override this method if you want to inject some custom behavior.
        # """
        
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
#             dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker

        return self.accelerator.prepare(DataLoader(train_dataset, shuffle=False, **dataloader_params))

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        # """
        # Returns the evaluation [`~torch.utils.data.DataLoader`].

        # Subclass and override this method if you want to inject some custom behavior.

        # Args:
        #     eval_dataset (`torch.utils.data.Dataset`, *optional*):
        #         If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
        #         by the `model.forward()` method are automatically removed. It must implement `__len__`.
        # """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = self.data_collator

        if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
#             dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last

        return self.accelerator.prepare(DataLoader(eval_dataset, shuffle=False, **dataloader_params))

trainer = CustomTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_massive["train"],
    eval_dataset=tokenized_eval["train"],
    data_collator=data_collator,
    tokenizer=tokenizer
)

if __name__ == "__main__":
    trainer.train()
    trainer.save_model(f"{model_dir}/final")
    wandb.finish()