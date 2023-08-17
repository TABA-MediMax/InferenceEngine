# InferenceEngine
MediMax 프로젝트의 Inference Engine 개발 repository

2023/8/14(월)
Triton Server 사용법 공부 및 실행
    - config.pbtxt 작성방법 학습함
    - directory 구조가 어떻게 되어야하는지 학습함
    - 문제: 현재 갖고있는 학습 모델의 입출력 값이 우리가 원하는 Triton Server API가 아님

2023/8/15(화)
    - 현재 모델을 다시 학습시키기 위해 시간이 너무 많이 필요하므로, 현재 모델을 손대지 않고 입출력 값을 변경할 방법 탐색
    - Triton Ensembel Scheduling 활용 가능
    - Triton Ensembel Scheduling 사용 방법 공부 및 적용
    - 문제: 모델 파일의 분석에 어려움 발생(.pt 파일)

2023/8/16(수)
    - .pt 파일 분석 후, 입출력 형태 정확히 확인
    
    The model expects an input shape of (1,3,224,224)
     - 1 is the batch size.
     - 3 represents the RGB channels of the image.
     - 224×224 is the height and width of the image.
    
    The output shape is (1,5000)
     - 1 for the batch size.
     - 5000 for the probability scores of each of the 5000 pill classes.
So, the model takes in a 224×224 RGB image and produces a 5000-dimensional vector representing the probability distribution over the 5000 pill classes.
    - 확인된 입출력 형태에 맞춰, preprocessing model, postprocessing model 구현
    - Triton을 위한 config.pbtxt 작성
    - 문제: .pt 파일 분류 결과로 나오는 확률분포에 맞는 classification list가 없음.
    - 해결: 내일 새로운 모델을 학습시켜서 사용해보기.

2023/8/17(목)
    - 






