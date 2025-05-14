import json
import time
import pandas as pd
import pathlib
import tiktoken
from utils import *
from data_preprocessing import create_data_loader
from gpt.build_llm.gpt2 import GPTModel
from gpt.model_weights.load_model_weights import load_weights,load_foundational_model

model_type = "gpt2"
current_path = pathlib.Path(__file__).resolve().parent.parent
with open('../base_config.json', 'r') as f:
    configs = json.load(f)
    GPT_CONFIG = configs["base_configs"]
    GPT_CONFIG.update(configs["model_configs"][model_type])

"""
Loading data and preprocessing the data
"""
raw_data = pd.read_csv(current_path/"datasets/Sentiment_data.csv",encoding='latin-1')
raw_data = raw_data[['text','sentiment']]
raw_data.rename(columns={"text":"Text","sentiment":"Label"},inplace=True)
balanced_df = (
    raw_data.groupby('Label', group_keys=False)
      .apply(lambda x: x.sample(n=500, random_state=42))
      .reset_index(drop=True)
)
balanced_df['Label'] = balanced_df['Label'].map({'negative': 0, 'neutral': 1,"positive":2})
train_df, validation_df, test_df = random_split(balanced_df, 0.8, 0.1)

train_loader = create_data_loader(
    df=train_df,
    batch_size=4,
    shuffle=True,
    num_workers=0,
    drop_last=True,
)

val_loader = create_data_loader(
    df=validation_df,
    batch_size=4,
    num_workers=0,
    drop_last=False,
)

test_loader = create_data_loader(
    df=test_df,
    batch_size=4,
    num_workers=0,
    drop_last=False,
)

"""
Downloading the foundational model weights and loading them to out GPT model
"""
state_dict = load_foundational_model(model_type)
model = GPTModel(GPT_CONFIG)
model = load_weights(GPT_CONFIG["n_layers"], model, state_dict)

"""
Freezing some model params and finetuning only the shallow layers
"""

for param in model.parameters():
    param.requires_grad = False

num_classes = 3
model.out_head = torch.nn.Linear(in_features=GPT_CONFIG["emb_dim"], out_features=num_classes)

# Training only the last half of transformer layers, layernorm and output layer
for param in model.trf_blocks[-(GPT_CONFIG["n_layers"]//2):].parameters():
    param.requires_grad = True

for param in model.final_norm.parameters():
    param.requires_grad = True

"""
Training of the model starts here
"""

start_time = time.time()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
num_epochs = 25
train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=50, eval_iter=5,
)

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")

"""
testing the predictions of the model
"""

text_1 = (
    "I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried enough today."
)

tokenizer = tiktoken.get_encoding("gpt2")
predict_sentiment(classify_review(
    text_1, model, tokenizer, device, max_length=train_loader.max_length
))



