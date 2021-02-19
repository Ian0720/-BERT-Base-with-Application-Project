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
