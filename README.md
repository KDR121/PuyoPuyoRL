# 「ぷよぷよ」のゲームAI

## 作品説明

SEGA社が制作・コンパイルを行っている「ぷよぷよ」シリーズに対して、自動でゲームを解くAIを深層強化学習によって作成した。

### 制作目的

老若男女問わず瞬時にゲームの内容やルールが理解できるゲームでありながらも、eSportsとして競技性・戦略性があり、知れば知るほど奥深さを感じるゲームである「ぷよぷよ」に対して、私自身がこれまで勉強sてきた機械学習や深層学習の知識を活かしてAIを作成したいと考えたため。


### 制作概要

「ぷよぷよ通」の一人用モードできる限り高いスコアを獲得することに対してアプローチを行った。

### 使用手法
##### 強化学習
![rl](https://user-images.githubusercontent.com/59335458/152766417-1a71d0ea-6ae0-4e6d-b60b-481d2846d386.PNG)

### 開発環境
python 3.7

RL:

  enviroment : [frostburn](https://github.com/frostburn/gym_puyopuyo)
  
  Agent : Deep Q-Network

### 結果・考察

学習初期
![学習初期](https://user-images.githubusercontent.com/59335458/152766887-85bee738-2afc-43ef-ab64-c81a19b005cf.PNG)

学習中期
![学習中記](https://user-images.githubusercontent.com/59335458/152766892-1fec58dc-07d8-4fb1-8814-e58b906422f2.PNG)

学習最終
![学習終盤](https://user-images.githubusercontent.com/59335458/152766901-6841d048-820d-4413-aaef-26cd8ca7607b.PNG)



