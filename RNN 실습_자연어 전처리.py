import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data, datasets

BATCH_SIZE = 64
lr = 0.001
EPOCHS = 40
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

# IMDB 데이터셋 로딩
# sequential : 순차적인 데이터 셋, batch_first : 신경망에 입력되는 텐서의 첫번째 차원값이 batch size, lower: 소문자화
TEXT = data.Field(sequential=True, batch_first=True, lower=True)
LABEL = data.Field(sequential=False, batch_first=True)

trainset, testset = datasets.IMDB.splits(TEXT, LABEL)

# 워드 임베딩을 위한 워드 딕셔너리 생성
TEXT.build_vocab(trainset, min_freq=5) # min_freq : 최소 5번이상 등장 단어만을 사전에 담음, 5번 미만은 unknown(unk) 로 대체
LABEL.build_vocab(trainset)

# 검증셋 만들기 train 80 % validation 20 %
trainset, valset = trainset.split(split_ratio=0.8)
train_iter, val_iter, test_iter = data.BucketIterator.splits(
    (trainset, valset, testset),
    batch_size=BATCH_SIZE,
    shuffle=True, repeat=False
)

vocab_size = len(TEXT.vocab)
n_classes = 2

print("[학습셋]: %d [검증셋]: %d [테스트셋]: %d [단어수]: %d [클래스 ] %d" % (len(trainset), len(valset), len(testset), vocab_size, n_classes))


class BasicGRU(nn.Module):
    def __init__(self, n_layers, hidden_dim, n_vocab, embed_dim, n_classes, dropout_p=0.2):
        super(BasicGRU, self).__init__()
        print("Building Basic GRU model")
        self.n_layers = n_layers # 은닉 벡터들의 층, 복잡한 모델이 아닌이상 2이하로 정의
        self.embed = nn.Embedding(n_vocab, embed_dim) # n_vocab : 사전에 등재된 단어수  embed_dim : 임베딩된 단어텐서가 지니는 차원값
        self.hidden_dim = hidden_dim #은닉베터의 차원값
        self.dropout = nn.Dropout(dropout_p)

        # RNN은 고질적으로 vanshing gradient 혹은 gradient explosion으로 인한 정보소실 문제가 있음, 그래서 GRU 채용
        self.gru = nn.GRU(embed_dim, self.hidden_dim,
                          num_layers=self.n_layers,
                          batch_first=True)

        self.out = nn.Linear(self.hidden_dim, n_classes)


    def forward(self, x):
        x = self.embed(x) # 한배치속 단어들을 워드임베딩 먼저 수행
        h_0 = self._init_state(batch_size = x.size(0)) # RNN 계열의 신경망은 은닉베터(H_0)를 정의해줘야 된다
        x, _ = self.gru(x, h_0) # 은닉벡터들이 시계열 배열 형태로 반환, 결과값 (batch_size, 입력 x의 길이, hidden_dim)
        h_t = x[:,-1,:]
        self.dropout(h_t)
        logit = self.out(h_t)
        return logit

    def _init_state(self, batch_size=1):
        weight = next(self.parameters()).data #next 반복할수 있을때 해당값 출력, .data 값 얻기
        return weight.new(self.n_layers, batch_size, self.hidden_dim).zero_() #.new 비슷한 타입의 다른 사이즈의 tensor 생성


def train(model, optimizer, train_iter):
    model.train()
    for b, batch in enumerate(train_iter):
        x, y = batch.text.to(DEVICE), batch.label.to(DEVICE) # batch.label은 1또2 값
        y.data.sub_(1) # 0 또는 1로 변환
        optimizer.zero_grad()

        logit = model(x)
        loss = F.cross_entropy(logit, y)
        loss.backward()
        optimizer.step()


def evaluate(model, val_iter):
    model.eval()
    corrects, total_loss = 0, 0
    for batch in val_iter:
        x, y = batch.text.to(DEVICE), batch.label.to(DEVICE)
        y.data.sub_(1)
        logit = model(x)
        loss = F.cross_entropy(logit, y, reduction='sum')
        total_loss += loss.item()
        corrects += (logit.max(1)[1].view(y.size()).data == y.data).sum()

    size = len(val_iter.dataset)
    avg_loss = total_loss / size
    avg_accuracy = 100*corrects/size
    return avg_loss, avg_accuracy


model = BasicGRU(1, 64, vocab_size, 32, n_classes, 0.5).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters())

best_val_loss = None
for e in range(1, EPOCHS+1):
    train(model, optimizer, train_iter)
    val_loss, val_accuracy = evaluate(model, val_iter)

    print("[이폭: %d] 검증 오차:%5.2f | 검증 정확도:%5.2f" % (e, val_loss, val_accuracy))

    if not best_val_loss or val_loss < best_val_loss:
        if not os.path.isdir("snapshot"):
            os.makedirs("snapshot")
        torch.save(model.state_dict(), './snapshot/txtclassification.pt')
        best_val_loss = val_loss

model.load_state_dict(torch.load('./snapshot/txtclassification.pt'))
test_loss, test_acc = evaluate(model, test_iter)
print('테스트 오차: %5.2f | 테스트 정확도: %5.2f' % (test_loss, test_acc))