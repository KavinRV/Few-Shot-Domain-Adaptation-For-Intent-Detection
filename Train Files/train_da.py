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
import wandb

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="massive_da",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 2e-05,
    "architecture": "XLM-R_CE",
    "dataset": "massive_eng_sur_hard",
    "epochs": 2,
    }
)

tokenized_massive = load_dataset("carnival13/xlmr_hard_curr_uda_ep3_corr").remove_columns(["input"])
tokenized_eval = load_dataset("carnival13/xlmr_eval2").remove_columns(["input"])
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
    


class ReRanker(XLMRobertaForSequenceClassification):
    def __init__(self, config: XLMRobertaConfig):
        # config.rank_score_index = 32019
        config.n_pass = 10
        # config.output_hidden_states = True
        super().__init__(config)
        self.da = config.da
        self.n_pass = config.n_pass
        if self.da:
            config.num_labels = config.num_domain
            self.domain_classifier = XLMRobertaClassificationHead(config)
            self.grl = GRL(1)


    def forward(self, input_ids=None, attention_mask=None, domain_label=None, pass_label=None, **kwargs):

        batch_size_n, seq_len = input_ids.size()
        batch_size = int(batch_size_n/self.n_pass)

        out = super().forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
#         rank_score = self.rank_head(out.decoder_hidden_states[-1][:, 0, :])
        rank_score = out.logits.view(batch_size, -1)
        loss = None
        da_loss = None

        if domain_label != None:
            # print(out.hidden_states[-1].shape)
            domain_logits = self.domain_classifier(self.grl(out.hidden_states[-1]))
            loss_fct = nn.CrossEntropyLoss()
            da_loss = loss_fct(domain_logits, domain_label.view(-1))
            wandb.log({"da_loss": da_loss})

        if pass_label != None:
            pass_label = pass_label[::self.n_pass]
            rank_score = rank_score
#             gen_score = out.gpe_score

            loss_fct1 = nn.CrossEntropyLoss(ignore_index=-100)
#             loss_fct2 = nn.CrossEntropyLoss()

            rank_loss = loss_fct1(rank_score, pass_label.view(-1))
#             gen_loss = loss_fct2(gen_score, pass_label.view(-1))

#             loss = rank_loss + gen_loss

            # loss = rank_loss
            wandb.log({"loss": rank_loss})

        if torch.isnan(rank_loss):
            loss = da_loss
        else:
            loss = rank_loss + da_loss
        wandb.log({"overall_loss": loss})

        ret =  SequenceClassifierOutput(
            loss=loss,
            logits=out.logits,
            hidden_states=out.hidden_states,
            attentions=out.attentions
        )
        ret.rank_score = rank_score
        return ret
    

mod_ckp = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(mod_ckp)
# question = "query:"
# title = "intent:"
# context = "examples:"
# eou = "<sep>"
# print
# tokenizer.add_tokens([question, title, context, eou], special_tokens=True)
config = XLMRobertaConfig.from_pretrained(mod_ckp)
config.output_hidden_states = True
# config.rank_score_index = tokenizer.convert_tokens_to_ids("<extra_id_80>")
# config.vocab_size = config.vocab_size + 4
config.n_pass = 10
config.da = True
config.num_labels = 1
config.num_domain = 2
model = ReRanker.from_pretrained(mod_ckp, config=config, ignore_mismatched_sizes=True)



batch_size = 2*model.config.n_pass
model_dir = f"rbrt_hard_curr_uda_ep3_corr"

args = TrainingArguments(
    model_dir,
    label_names=["pass_label", "domain_label"],
    evaluation_strategy="steps",
    eval_steps=30,
    logging_strategy="steps",
    logging_steps=30,
    save_strategy="steps",
    save_steps=30,
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    gradient_accumulation_steps=32
)


data_collator = DataCollatorWithPadding(tokenizer)


class CustomTrainer(Trainer):
#     def compute_loss(self, model, inputs, return_outputs=False):
# #         print(inputs.get("input_ids").size())
#         outputs = model(**inputs)

#         loss = outputs.loss
#         return (loss, outputs) if return_outputs else loss

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
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
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                by the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
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
