import os 
import shutil 

import tensorflow as tf
from tensorflow.python.keras.metrics import accuracy
from tensorflow.python.ops.gen_batch_ops import batch 
import tensorflow_hub as hub
import tensorflow_text as text 
from official.nlp import optimization # AdamW Optmizer 사용을 위해

import matplotlib.pyplot as plt 

import numpy as np 

tf.get_logger().setLevel('ERROR')

# IMDB 데이터 세트 다운로드 해주는 구간, 인터넷 영화 데이터베이스에서 가져온 50,000개의 영화 리뷰 텍스트가 포함된 대형 영화 리뷰 데이터 세트이므로
# 가급적이면 좋은 성능을 가진 컴퓨터나 Colab으로 돌려주는 것을 추천드립니다
# 저는.. 그냥 했습니다.. 네.. 그렇다구요.. :)
url = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'

dataset = tf.keras.utils.get_file('aclImdb_v1.tar.gz', url, untar=True, cache_dir='.', cache_subdir='')

dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')

train_dir = os.path.join(dataset_dir, 'train')

# 데이터를 쉽게 로드할 수 있도록, 사용하지 않는 폴더를 제거해주는 부분입니다.
remove_dir = os.path.join(train_dir, 'unsup')
shutil.rmtree(remove_dir)

AUTOTUNE = tf.data.AUTOTUNE
batch_size = 32
seed = 42

# 여기서는 validation_split 인수를 사용하여, 훈련 데이터의 분할을 80:20으로 세팅하고 검증하는 파트를 생성합니다.
# validation_split 및 subset 인수를 사용하는 경우, 임의 시드를 저장하거나 shuffle=False를 전달하여 유효성 검사 및 훈련 분할이 겹치지 않도록 해주어야 합니다.
raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/train',
    batch_size=batch_size,
    validation_split=0.2, # 여기서 분할을 지정해주는 것입니다, 만약 0.1로 했다면 90:10이 되겠지요?
    subset='training',
    seed=seed)

class_names = raw_train_ds.class_names
train_ds = raw_train_ds.cache().prefetch(buffer_size=AUTOTUNE)

val_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    seed=seed)

val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

test_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/test',
    batch_size=batch_size)

test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 상위의 과정을 거치고 난 뒤, 몇가지 리뷰를 예시로 살펴보는 구간입니다.
for text_batch, label_batch in train_ds.take(1):
    for i in range(3):
        print(f'Review: {text_batch.numpy()[i]}')
        label = label_batch.numpy()[i]
        print(f'Label:{label}({class_names[label]})')

# 이제, 여기에서 Tensorflow Hub에서 로드 할 BERT 모델을 선택하고 미세조정하는 구간입니다, 여러 BERT 모델을 사용하실 수 있습니다.

# BERT-Base : Uncased 및 원래 BERT 작성자가 발효한 훈련 된 가중치가 있는 7개로 구성된 베이스 모델 입니다.
# 소형 BERT는 동일한 일반 아키텍처를 갖습니다, 하지만 Transformer 블록 수가 적거나 작아서 속도, 크기 및 품질 간의 균형을 탐색할 수 있습니다.
# ALBERT : 레이어간에 매개 변수를 공유해서 모델 크기(계산 시간이 아닙니다.)를 줄이는 "A Lite BERT"의 4가지 모델을 가지고 있습니다.
# BERT Expert(BERT 전문가) : 모두 BERT 기반 아키텍처를 가지고 있지만, 대상 작업에 더 가깝게 맞추기 위해 서로 다른 사전 교육 도메인 중에서 선택할 수 있는, 총 8개로 이루어진 모델입니다.
# Electra : BERT와 동일한 아키텍처(세가지 다른 크기)를 갖지만 GAN(Generative Adversarial Network)과 유사한 설정에서 판별자로 사전 훈련됩니다.
# Talking_Heads Attention 및 Gated GELU[base, large 두가지가 있습니다.]가 있는 BERT는 Transformer 아키텍처의 핵심 중 두가지가 개선된 모델이라 보시면 됩니다.

bert_model_name = 'small_bert/bert_en_uncased_L-4_H-512_A-8' 

map_name_to_handle = {
    'bert_en_uncased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3',
    'bert_en_cased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/3',
    'bert_multi_cased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/3',
    'small_bert/bert_en_uncased_L-2_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-2_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-2_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-2_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-4_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-4_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-4_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-4_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-6_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-6_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-6_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-6_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-8_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-8_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-8_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-8_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-10_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-10_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-10_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-10_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-768_A-12/1',
    'small_bert/bert_en_uncased_L-12_H-128_A-2':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1',
    'small_bert/bert_en_uncased_L-12_H-256_A-4':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-256_A-4/1',
    'small_bert/bert_en_uncased_L-12_H-512_A-8':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-512_A-8/1',
    'small_bert/bert_en_uncased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-768_A-12/1',
    'albert_en_base':
        'https://tfhub.dev/tensorflow/albert_en_base/2',
    'electra_small':
        'https://tfhub.dev/google/electra_small/2',
    'electra_base':
        'https://tfhub.dev/google/electra_base/2',
    'experts_pubmed':
        'https://tfhub.dev/google/experts/bert/pubmed/2',
    'experts_wiki_books':
        'https://tfhub.dev/google/experts/bert/wiki_books/2',
    'talking-heads_base':
        'https://tfhub.dev/tensorflow/talkheads_ggelu_bert_en_base/1',
}

map_model_to_preprocess = {
    'bert_en_uncased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'bert_en_cased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_cased_preprocess/3',
    'small_bert/bert_en_uncased_L-2_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-2_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-2_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-2_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-4_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-4_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-4_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-4_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-6_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-6_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-6_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-6_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-8_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-8_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-8_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-8_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-10_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-10_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-10_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-10_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-12_H-128_A-2':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-12_H-256_A-4':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-12_H-512_A-8':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'small_bert/bert_en_uncased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'bert_multi_cased_L-12_H-768_A-12':
        'https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3',
    'albert_en_base':
        'https://tfhub.dev/tensorflow/albert_en_preprocess/2',
    'electra_small':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'electra_base':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'experts_pubmed':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'experts_wiki_books':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'talking-heads_base':
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
}

tfhub_handle_encoder = map_name_to_handle[bert_model_name]
tfhub_handle_preprocess = map_model_to_preprocess[bert_model_name]

print(f'BERT model selected           : {tfhub_handle_encoder}')
print(f'Preprocess model auto-selected: {tfhub_handle_preprocess}')

# 위에 각 모델별로 주소를 첨부했습니다, 위의 링크를 따르거나 실행 후 인쇄된 'tfhub.dev' URL을 클릭하시면 됩니다.
# 미세 조정이 더 빠르기 때문에 작은 BERT(매개 변수가 더 적습니다.)로 시작하는 것이 좋습니다.
# 작은 모델을 선호하지만, 더 높은 정확도를 원하실 경우 'ALBERT'를 다음 옵션으로 볼 수 있겠습니다.
# 더 나은 정확도를 원하신다면, 저 위에 설명드린 모델 중 순서 중 하나를 선택하셔서 돌리시면 됩니다 :)

# 물론 제시해드린 모델 외에도 더 크고 더 나은 정확도를 자랑할 수 있는 여러 버전들의 모델들이 있습니다.
# 하지만 단일 GPU에서 미세 조정하기에는 너무 큽니다, 따라서 TPU colab에서 BERT를 사용하여 Solve GLUE 작업 에세이를 수행이 가능한 방법이 있습니다.
# 물론 제한시간 또한 존재합니다, 무제한으로 이용하시기 위해서는 일정의 페이를 지불하셔야 합니다(기간제 텀을 두고 결제가 이루어짐, 1달 간격).

# 이제는 전처리 모델을 세팅하는 구간입니다.
bert_preprocess_model = hub.KerasLayer(tfhub_handle_preprocess)

# 일부 텍스트에 한정하여 전처리 모델을 구축한 후, 시도 및 출력이 이루어지는 구간입니다.
text_test = ['this is such an amazing movie!']
text_preprocessed = bert_preprocess_model(text_test)

print(f'Keys       : {list(text_preprocessed.keys())}')
print(f'Shape      : {text_preprocessed["input_word_ids"].shape}')
print(f'Word Ids   : {text_preprocessed["input_word_ids"][0, :12]}')
print(f'Input Mask : {text_preprocessed["input_mask"][0, :12]}')
print(f'Type Ids   : {text_preprocessed["input_type_ids"][0, :12]}')

# 여기까지 진행하셨다면, BERT 모델이 사용할 전처리의 3가지 출력 모델을 확인하실 수 있으실 겁니다.
# 입력은 128개의 토큰으로 커팅되며, 토큰 수는 맞춤 설정이 가능합니다.
# input_type_ids는 단일 문장 입력이므로 하나의 값(0)만 가집니다.

# 다음은, BERT 모델을 사용하는 구간입니다.
# 사용에 앞서, 자신의 모델에 적용하기 전에 출력을 살펴보려 합니다.
# TF Hub에서 로드하고 반환된 값을 확인해줍니다.

bert_model = hub.KerasLayer(tfhub_handle_encoder)
bert_results = bert_model(text_preprocessed)

print(f'Loaded BERT: {tfhub_handle_encoder}')
print(f'Pooled Outputs Shape:{bert_results["pooled_output"].shape}')
print(f'Pooled Outputs Values:{bert_results["pooled_output"][0, :12]}')
print(f'Sequence Outputs Shape:{bert_results["sequence_output"].shape}')
print(f'Sequence Outputs Values:{bert_results["sequence_output"][0, :12]}')

# 여기서 BERT 모델은 3개의 중요한 키를 반환함을 알 수 있습니다.
# pooled_output : 각 입력 시퀀스를 전체로 나타냅니다, 형태는 [batch_size, H] 이며, 이는 전체 영화 리뷰에 대한 임베딩이라고 생각하시면 됩니다.
# sequence_output : 컨텍스트의 각 입력 토큰을 나타냅니다, 형태는 [batch_size, seq_length, H] 이며, 이는 영화 리뷰의 모든 토큰에 대한 문맥 삽입이라고 보시면 됩니다.
# encoder_outputs : 마지막으로 이것은, L Transformer 블록의 중간 활성화 단계를 가집니다.
#                   outputs["encoder_outputs"][i]는 0 <= i < L에 대해 i번째 Transformer 블록의 출력을 갖는 [batch_size, seq_length, 1024]모양의 Tensor입니다.
#                   목록의 마지막 값은 sequence_output과 같습니다.
# 미세 조정을 위해, pooled_output 배열을 사용합니다.


# 전처리 모델, 선택한 BERT 모델에 한하여 하나의 Dense 및 Dropout 레이어를 사용하여 매우 간단한 '미세 조정 모델'을 생성하는 부분입니다.
def build_classifier_model():
  text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
  preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
  encoder_inputs = preprocessing_layer(text_input)
  encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
  outputs = encoder(encoder_inputs)
  net = outputs['pooled_output']
  net = tf.keras.layers.Dropout(0.1)(net)
  net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
  return tf.keras.Model(text_input, net)

# 모델이 전처리 모델의 출력으로 실행되는지 확인하는 구간입니다.
classifier_model = build_classifier_model()
bert_raw_result = classifier_model(tf.constant(text_test))
print(tf.sigmoid(bert_raw_result))

# 모델 구조는, 게시물에서 확인 바랍니다.
# 이제 전처리 모듈, BERT 인코더, 데이터 및 분류기를 포함하여 모델을 학습하는데 필요한 모든 부분이 준비되었습니다.
# 손실 기능을 거쳐 옵티마이저 및 모델로드 그리고 훈련 순서대로 가게됩니다.

# 손실 기능
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
metrics = tf.metrics.BinaryAccuracy()

# 옵티마이저
epochs = 20
steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
num_train_steps = steps_per_epoch * epochs 
num_warmup_steps = int(0.1*num_train_steps)

init_lr = 3e-5
optimizer = optimization.create_optimizer(init_lr=init_lr,
                                          num_train_steps=num_train_steps,
                                          num_warmup_steps=num_warmup_steps,
                                          optimizer_type='adamw')

# 모델 로드 그리고 훈련
# 이전에 생성한 classifier_model을 사용하여, 손실, 측정 항목 및 최적화 프로그램으로 모델을 컴파일 할 수 있습니다.
# 학습 시간은 선택한 BERT 모델의 복잡성에 따라서 달라집니다.

print(f'Training model with {tfhub_handle_encoder}')
history = classifier_model.fit(x=train_ds,
                               validation_data=val_ds,
                               epochs=epochs)

# 모델의 성능을 살펴보는 구간입니다, 두개의 값이 반환될 것입니다(손실(값이 낮을수록 좋습니다.), 정확도)
loss, accuracy = classifier_model.evaluate(test_ds)

print(f'Loss: {loss}')
print(f'Accuracy: {accuracy}')

# 시간 경과에 따른 정확성과 손실 도표
# model.fit()에 의해 반환된 History 객체를 기반으로 합니다, 비교를 위해 훈련 및 검증 손실과 훈련 및 검증 정확도를 플로팅 할 수 있습니다.
history_dict = history.history
print(history_dict.keys())

acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)
fig = plt.figure(figsize=(10, 6))
fig.tight_layout()

plt.subplot(2, 1, 1)
# "bo" is for "blue dot"
plt.plot(epochs, loss, 'r', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
# plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

# 실행하면, 그림이 '뿅'하고 나타날 것입니다.
# 그림에서 나타내는 바로, 빨간색 선은 훈련 손실 및 정확도를 나타내고, 파란색 선은 검증 및 손실 정확도를 나타냅니다.
# 이제 나중에(언제든) 사용하기 위해 미세 조정된 모델을 저장해주는 작업을 수행하면 됩니다, 방법은 아래와 같습니다.
dataset_name = 'imdb'
saved_model_path = './{}_bert'.format(dataset_name.replace('/', '_'))

classifier_model.save(saved_model_path, include_optimizer=False)

# 이제 메모리에 있는 모델을 다시 로드하는 부분입니다.
reloaded_model = tf.saved_model.load(saved_model_path)

# 이제는 원하는 문장을, 모델을 이용하여 테스트 할 수 있습니다.
# 아래 예제 변수에 추가하기만 하면 됩니다.
# 모두 여기까지 따라오시느라 고생하셨습니다!
def print_my_examples(inputs, results):
  result_for_printing = \
    [f'input: {inputs[i]:<30} : score: {results[i][0]:.6f}'
                         for i in range(len(inputs))]
  print(*result_for_printing, sep='\n')
  print()


examples = [
    'this is such an amazing movie!',  # 이것은, 앞에서 시행한 문장과 동일한 문장입니다 헤헤
    'The movie was great!',
    'The movie was meh.',
    'The movie was okish.',
    'The movie was terrible...'
]

reloaded_results = tf.sigmoid(reloaded_model(tf.constant(examples)))
original_results = tf.sigmoid(classifier_model(tf.constant(examples)))

print('Results from the saved model:')
print_my_examples(examples, reloaded_results)
print('Results from the model in memory:')
print_my_examples(examples, original_results)

# Trying to access resource using the wrong type. Expected class tensorflow::lookup::LookupInterface got class tensorflow::lookup::LookupInterface
# 상위의 멘트와 함께, 텐서플로우 버전에 관계없이, 윈도우 10 사용자의 유저와 맥북 유저 그리고 리눅스 사용자까지 모두 발견되는 공통된 에러입니다.
# 따라서, Colab에서의 구동을 추천드립니다.

# 제 기준에서, 239번째 라인에서 저는 위와 같은 문제가 발생하였습니다.
# 에러의 뜻은 다음과 같습니다 : 'Trying to access resource using the wrong type. : 잘못된 유형을 사용하여 리소스에 엑세스하려고 합니다.'
# https://www.tensorflow.org/tutorials/text/classify_text_with_bert 을 참고하여 오류의 문제점을 찾아보려 하였으나 크게 차이가 나는 부분을 발견하지 못했고
# 따라서 구글링해본 결과, 해외의 수많은 개발자분들도 동일한 문제가 발생되고 있음을 확인하였습니다.