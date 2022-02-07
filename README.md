# 「ぷよぷよ」のゲームAI

## 作品説明

SEGA社が制作・コンパイルを行っている「ぷよぷよ」シリーズに対して、自動でゲームを解くAIを深層強化学習によって作成した。

## 制作目的

老若男女問わず瞬時にゲームの内容やルールが理解できるゲームでありながらも、eSportsとして競技性・戦略性があり、知れば知るほど奥深さを感じるゲームである「ぷよぷよ」に対して、私自身がこれまで勉強sてきた機械学習や深層学習の知識を活かしてAIを作成したいと考えたため。


## 制作概要

「ぷよぷよ通」の一人用モードできる限り高いスコアを獲得することに対してアプローチを行った。

## 使用手法
##### 強化学習
![rl](https://user-images.githubusercontent.com/59335458/152766417-1a71d0ea-6ae0-4e6d-b60b-481d2846d386.PNG)

## 開発環境
python 3.7
Reinforcement Learning  
enviroment : [frostburn](https://github.com/frostburn/gym_puyopuyo)  
Agent : Deep Q-Network  
pytorch:  
OpenAI gym:  

## 結果・考察
#### 学習初期  
一定の箇所にのみ積み上がり、連鎖はおろかは最低数でゲームオーバーになってしまう。  
<img src = "https://user-images.githubusercontent.com/59335458/152766887-85bee738-2afc-43ef-ab64-c81a19b005cf.PNG" width = 200px>

#### 学習中期  
一連鎖や二連鎖は確認できるが、三連鎖以上の連鎖は確認されず、まだ埋められるマス目があってもゲームオーバーになってしまう。  
<img src = "https://user-images.githubusercontent.com/59335458/152766892-1fec58dc-07d8-4fb1-8814-e58b906422f2.PNG" width = 200px>

#### 学習終盤  
序盤に右端二列に積み上げて、何連鎖も行う。300回ぷよを落としてもゲームオーバーにならないことがある。6連鎖や7連鎖もたまに出る。  
<img src = "https://user-images.githubusercontent.com/59335458/152766901-6841d048-820d-4413-aaef-26cd8ca7607b.PNG" width = 200px>



