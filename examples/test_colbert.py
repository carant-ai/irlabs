from transformers import AutoTokenizer
from irlabs.models.colbert.modeling import BertForColbert

def main():
    model = BertForColbert.from_pretrained("indobenchmark/indobert-base-p1")
    assert isinstance(model, BertForColbert)
    tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
    text = "saya makan nasi goreng"
    token = tokenizer(text, return_tensors = "pt")
    print(f"DEBUGPRINT[4]: test_colbert.py:9: token={token}")
    output = model.forward(**token)
    print(f"DEBUGPRINT[1]: test_colbert.py:8: output={output.shape}")



if __name__ == "__main__":
    main()

