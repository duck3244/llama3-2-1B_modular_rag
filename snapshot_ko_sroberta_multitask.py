from transformers import AutoModel, AutoTokenizer

# 모델 이름 지정
model_name = "jhgan/ko-sroberta-multitask"

# 모델과 토크나이저 다운로드
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 로컬에 모델 저장
model_save_path = "./models/ko-sroberta-multitask"
tokenizer_save_path = "./models/ko-sroberta-multitask"

model.save_pretrained(model_save_path)
tokenizer.save_pretrained(tokenizer_save_path)

print(f"모델이 {model_save_path}에 저장되었습니다.")
print(f"토크나이저가 {tokenizer_save_path}에 저장되었습니다.")