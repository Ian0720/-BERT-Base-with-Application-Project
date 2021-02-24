## BERT 학습 파일 설명 
- 공부 겸 다음 개발 프로젝트에 응용하고 싶어서, 제가 공부한 부분들을 ipynb파일에 텍스트를 추가하는 형식으로 bar를 만들어 수록해놓았습니다.
- 혹시 오답이나 오역이 발견되었을 시, 알려주시면 감사하겠습니다

## 공유 동기
- 이전에는 Transformer를 활용하여 챗봇을 만들어 보았습니다.
- 따라서 한층 업그레이드 된 형태인 'BERT'를 이용하고자 이렇게 올리게 되었습니다.
- BERT가 어떻게 구동이되고 어떤 파일들이 로드가되어 서로가 협응해서 하나의 작업이 이루어지는지 궁금했고<br/>
  또한 한눈에 보기 쉽게 정리해놓은 게시물들이 (제 생각에는) 부족하다고 생각되어서 직접 이런저런 부분들, 이해가 가지 않거나 혹은 그럴 법한 부분들에 대해<br/>
  발품 팔아가며 적어보았습니다.<br/> 
<br/>

## BERT 파일 구조
<img src="https://user-images.githubusercontent.com/79067558/108457590-46a89600-72b6-11eb-9326-1ed8e6cfb535.png" width="70%" height="70%"><br/>

## 핵심 파일 설명
|Title|Explanation|From|
|:----:|:---------:|:----------:|
|BERT Base|Main Base File From Google|https://github.com/google-research/bert|
|wordpiece.py|Saltlux AI labs @ AIR Group|https://github.com/lovit/WordPieceModel|
|WikiExtractor.py|Author: Giuseppe Attardi, extracts and cleans text from a Wikipedia database dump|https://github.com/attardi/wikiextractor|<br/>
<br/>

## 추가된 파일 설명(BERTText.py)

## BERT를 응용하여 텍스트를 분류하는 프로젝트를 수행해 보았습니다.
- 각 파일의 실행 경과 및 코드가 각 해당 라인에 작성된 이유 또한 작성해 놓았으니 참고하시면 됩니다.
- 예제의 용량은 커서 업로드 할 수 없었음을 양해 부탁드립니다.
- 저 같은 경우는, 인터넷 영화 데이터베이스에서 가져온 50,000개의 영화 리뷰 텍스트가 포함된 대형 영화 리뷰 데이터 세트를 참고하였으며
- 이를 통하여 영화리뷰를 긍정적 또는 부정적으로 분류하는 감정 분석 모델을 구축해보았습니다.
- 아 참, 마지막에 올린 부분은 다국어 지원을 위한 파일입니다, 또한 ipynb는 원본이니 참고하셔서 둘다 구동해보시는 것도 좋을 것 같아서 올렸습니다!

## BERT MODEL ARCHITECTURE
<img src="https://user-images.githubusercontent.com/79067558/108948232-70dac900-76a5-11eb-80c9-a1b8b55bf200.png" width="70%" height="70%"><br/>

## Training and Validation Loss (시간 경과에 따른 정확성과 손실 도표)
<img src="https://user-images.githubusercontent.com/79067558/108948326-a1226780-76a5-11eb-95fc-9dc4b8e082ea.png" width=70% height="70%"><br/>

## 참고한 부분
|Title|Explanation|From|
|:-------:|:----------:|:---------:|
|BERT Project|About Text Classification by 'BERT'|https://www.tensorflow.org/tutorials/text/classify_text_with_bert (Tensorflow Official)|
